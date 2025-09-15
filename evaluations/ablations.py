# Load in multiple policy configurations and corresponding checkpoints, run tests for
# input scenario/env, and save rewards & actions to CSV.


import sys, os, glob
import yaml
import torch
from pathlib import Path
from torchrl.modules import ProbabilisticActor
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import (ExplorationType, set_exploration_type)
from tensordict.tensordict import TensorDict
from collections import defaultdict

import csv

# Add the parent directory itself to sys.path so imports like 'envs' work
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
print("Parent dir:", parent_dir)
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir)+"\\envs")

from envs.scenarios.explore_comms_tasks import Scenario
from experiment_vec import load_yaml_to_kwargs, init_device, create_env, create_actor

def get_by_cycle(xs, i):
    """Safely pick an item even if lists are length-1; cycles when needed."""
    return xs[i % len(xs)]


def pair_model_and_checkpoints(model_dir: str, ckpt_dir: str):
    """
    Return sorted list of (model_yaml_path, checkpoint_pt_path) pairs
    where the basenames (without extension) match across folders.
    """
    model_dir = Path(model_dir)
    ckpt_dir = Path(ckpt_dir)

    yaml_paths = {p.stem: p for p in sorted(model_dir.glob("*.yaml"))}
    # Strip any prefix before the actual model name in .pt filenames
    pt_paths = {}
    for p in sorted(ckpt_dir.glob("*.pt")):
        # Example: dense_fullTF.pt -> fullTF
        name = p.stem
        if "_" in name:
            name = name.split("_")[-1]
        pt_paths[name] = p

    common = sorted(set(yaml_paths.keys()) & set(pt_paths.keys()))
    missing_models = sorted(set(pt_paths.keys()) - set(yaml_paths.keys()))
    missing_ckpts = sorted(set(yaml_paths.keys()) - set(pt_paths.keys()))


    print("yaml paths:", yaml_paths)
    print("pt paths:", pt_paths)
    print("Common:", common)
    print("Miss models:", missing_models)
    print("Miss ckpts:", missing_ckpts)


    if missing_models:
        print("[warn] Checkpoints with no matching YAML:", ", ".join(missing_models))
    if missing_ckpts:
        print("[warn] YAMLs with no matching checkpoint:", ", ".join(missing_ckpts))
    if not common:
        raise FileNotFoundError("No matching model/checkpoint pairs found.")

    return [(str(yaml_paths[k]), str(pt_paths[k])) for k in common]


def test_setup(test_index: int,
               scenario_configs,
               env_configs,
               model_config_fp: str,
               checkpoint_fp: str,
               scenario_obj: Scenario,
               ) -> tuple[TransformedEnv, ProbabilisticActor, defaultdict]:
    """Prepare env and policy for a single test/model pair."""

    # LOAD CONFIGS #
    scenario_config = load_yaml_to_kwargs(get_by_cycle(scenario_configs, test_index))
    print(f"Loading model config {model_config_fp} & checkpoint {checkpoint_fp}...")
    env_config = load_yaml_to_kwargs(get_by_cycle(env_configs, test_index))
    model_config = load_yaml_to_kwargs(model_config_fp)

    # LOAD MODEL WEIGHTS #
    torch.serialization.add_safe_globals([defaultdict])
    torch.serialization.add_safe_globals([list])
    checkpt_data = torch.load(checkpoint_fp, weights_only=True)

    device = init_device()

    # CREATE SIM ENVIRONMENT #
    print("Creating sim environment...")
    env = create_env(scenario_obj, device, env_config, scenario_config)

    # ACTOR POLICY MODULES, LOAD WEIGHTS #
    print("Loading model weights...")
    num_features = model_config["num_features"]
    num_heuristics = model_config["num_heuristics"]
    d_feedforward = model_config["d_feedforward"]
    d_model = model_config["d_model"]
    agent_attn = model_config["agent_attn"]
    cell_pos_as_features = model_config["cell_pos_as_features"]
    agent_id_enc = model_config.get("agent_id_enc", True)
    use_encoder = model_config.get("use_encoder", True)
    use_decoder = model_config.get("use_decoder", True)
    rob_pos_enc = model_config.get("rob_pos_enc", True)
    no_transformer = model_config.get("no_transformer", False)
    if no_transformer:
        use_encoder = False
        use_decoder = False
        print(f"No transformer policy; use_encoder={use_encoder}, use_decoder={use_decoder}")

    action_softmax = model_config.get("action_softmax", False)
    action_max = model_config.get("action_max", False)
    print(f"Action max: {action_max}")
    if action_softmax == True:
        print("Using action softmax")
        env.base_env.use_softmax = True
    elif action_max == True:
        print("Using action max")
        env.base_env.use_max = True

    tf_act, policy_module = create_actor(
        env, 
        num_features, 
        num_heuristics, 
        d_feedforward, 
        d_model,
        agent_attn, 
        cell_pos_as_features,
        agent_id_enc,
        use_encoder,
        use_decoder,
        rob_pos_enc,
        device
    )
    tf_act.load_state_dict(checkpt_data['actor_state_dict'])
    tf_act.eval()

    logs = defaultdict(list)

    return env, policy_module, logs


def run_policy(env: TransformedEnv, logs: defaultdict, policy, rollout_steps: int):
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        print("Running policy...")
        env.reset()
        policy_rollout = env.rollout(rollout_steps, policy, return_contiguous=False)
        rewards = policy_rollout["next", "reward"][0][1:]  # Skip first step (startup)
        actions = policy_rollout["action"][0][1:]
        policy_reward_mean = rewards.mean().item()
        policy_reward_sum = rewards.sum().item()
        logs["policy reward (mean)"].append(policy_reward_mean)
        logs["policy reward (sum)"].append(policy_reward_sum)
        logs["rewards"].append(rewards.cpu().numpy())
        logs["actions"].append(actions.cpu().numpy())
    return policy_reward_mean, policy_reward_sum, rewards.cpu().numpy(), actions.cpu().numpy()


def run_tests(scenario_configs,
              env_configs,
              test_configs,
              model_ckpt_pairs,  # list of (model_yaml_fp, ckpt_fp)
              folder_path="test"):
    """Run tests to evaluate each model vs comparison method."""

    data_dir = os.path.join(folder_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_fp = os.path.join(data_dir, "results.csv")
    csv_header = ["model_config", "test", "run", "step", "reward", "action"]
    write_header = not os.path.exists(csv_fp)

    # Load (single) test config once unless multiple provided
    # We still support cycling for completeness
    scenario_obj = Scenario()

    for i, (model_config_fp, ckpt_fp) in enumerate(model_ckpt_pairs):
        model_config_name = os.path.basename(model_config_fp)
        checkpt_name = os.path.basename(ckpt_fp)
        # Choose a test index (cycle if multiple)
        test_index = i  # cycles via get_by_cycle inside test_setup / below
        test_config = load_yaml_to_kwargs(get_by_cycle(test_configs, test_index))

        print(f"Test {test_index} setup for model {model_config_name}...")
        env, policy, logs = test_setup(
            test_index,
            scenario_configs,
            env_configs,
            model_config_fp,
            ckpt_fp,
            scenario_obj,
        )

        num_runs = test_config["num_runs"]
        rollout_steps = test_config["rollout_steps"]
        create_gifs = test_config.get("create_gifs", False)
        
        # Enable rendering
        if hasattr(env, "base_env") and create_gifs:
            env.base_env.render = True

        for run in range(num_runs):
            torch.manual_seed(run)  # Reproducibility

            print(f"Preparing to run test {test_index} with policy, model {model_config_name}, run {run}...")

            # Configure rendering fp
            if create_gifs:
                render_name = f"test{test_index}_run{run}.gif"
                render_fp = os.path.join(f"{folder_path}/gif/{checkpt_name}/", render_name)
                if hasattr(env, "base_env"):
                    env.base_env.render_fp = render_fp
                    env.base_env.count = run * rollout_steps
                os.makedirs(os.path.dirname(render_fp), exist_ok=True)

            # Run test & log
            policy_reward_mean, policy_reward_sum, rewards, actions = run_policy(env, logs, policy, rollout_steps)
            print("Mean reward policy:", policy_reward_mean)

            # ---- Save per-step results to CSV ----
            with open(csv_fp, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(csv_header)
                    write_header = False
                for step, (reward, action) in enumerate(zip(rewards, actions)):
                    writer.writerow([
                        model_config_name, test_index, run, step,
                        float(reward),
                        action.tolist() if hasattr(action, 'tolist') else action
                    ])
        print("Done with model", model_config_name, "test", test_index)


if __name__ == "__main__":
    # Usage: python run_tests.py <scenario_fp> <env_fp> <test_fp> <model_configs_dir> <checkpts_dir> <output_folder>
    if len(sys.argv) != 7:
        print("Usage: python ablations.py <scenario_fp> <env_fp> <test_fp> <model_configs_dir> <checkpts_dir> <output_folder>")
        sys.exit(1)

    scenario_fp = sys.argv[1]
    env_fp = sys.argv[2]
    test_fp = sys.argv[3]
    model_configs_dir = sys.argv[4]
    checkpts_dir = sys.argv[5]
    test_folder_name = sys.argv[6]

    # Pair YAMLs and checkpoints by basename
    model_ckpt_pairs = pair_model_and_checkpoints(model_configs_dir, checkpts_dir)

    # Env, Scenario & params
    scenario_configs = [scenario_fp]
    env_configs = [env_fp]
    test_configs = [test_fp]

    run_tests(
        scenario_configs=scenario_configs,
        env_configs=env_configs,
        test_configs=test_configs,
        model_ckpt_pairs=model_ckpt_pairs,
        folder_path=test_folder_name
    )

# python evaluations\ablations.py "conf\scenarios\comms_5.yaml" "conf\envs\planning_env_explore_5_1env.yaml" "evaluations\tests\trial1.yaml"  "evaluations\configs_weights\policy_configs" "evaluations\configs_weights\policy_weights" evaluations