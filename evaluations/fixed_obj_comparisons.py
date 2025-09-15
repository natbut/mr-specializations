# Load in best policy configurations and corresponding checkpoint alongside HybDec planner
# config and fixed agent objectives & run tests for input scenario/env. 
# Save rewards to CSV.

import sys, os
import yaml
import torch
from torchrl.modules import ProbabilisticActor
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import (ExplorationType,
                                set_exploration_type)
from tensordict.tensordict import TensorDict
from collections import defaultdict

import csv
from pathlib import Path

# Get the parent directory of the current file
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.append(str(parent_dir))

from envs.scenarios.explore_comms_tasks import Scenario
from envs import planning_env_vec
from experiment_vec import load_yaml_to_kwargs, init_device, create_env, create_actor

def test_setup(test,
               scenario_configs,
               env_configs,
               model_configs,
               checkpoint,
               ):

    # LOAD CONFIGS #
    print("Loading configs & checkpoint...")
    scenario_config = load_yaml_to_kwargs(scenario_configs[test])
    env_config = load_yaml_to_kwargs(env_configs[test])
    model_config = load_yaml_to_kwargs(model_configs[test])
    # LOAD MODEL WEIGHTS #
    torch.serialization.add_safe_globals([defaultdict])
    torch.serialization.add_safe_globals([list])
    checkpt_data = torch.load(checkpoint, weights_only=True)

    device = init_device()

    # CREATE SIM ENVIRONMENT #
    print("Creating sim environment...")
    env = create_env(scenario, device, env_config, scenario_config)

    # ACTOR POLICY MODULES, LOAD WEIGHTS #
    print("Loading model weights...")
    # ACTOR POLICY MODULES, LOAD WEIGHTS #
    num_features = model_config["num_features"]
    num_heuristics = model_config["num_heuristics"]
    d_feedforward = model_config["d_feedforward"]
    d_model = model_config["d_model"]
    agent_attn=model_config["agent_attn"]
    cell_pos_as_features=model_config["cell_pos_as_features"]
    agent_id_enc = model_config.get("agent_id_enc", True)
    use_encoder = model_config.get("use_encoder", True)
    use_decoder = model_config.get("use_decoder", True)
    rob_pos_enc = model_config.get("rob_pos_enc", True)
    no_transformer = model_config.get("no_transformer", False)
    if no_transformer:
        use_encoder = False
        use_decoder = False
    action_softmax = model_config.get("action_softmax", False)
    action_max = model_config.get("action_max", False)
    if action_softmax == True:
        print("Using action softmax")
        env.base_env.use_softmax = True
    elif action_max == True:
        print("Using action max")
        env.base_env.use_max = True
    tf_act, policy_module = create_actor(env,
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


def run_tests(scenario_configs,
              env_configs,
              test_configs,
              model_configs,
              checkpoint,
              folder_path="test",
              ):
    """Run tests to evaluate model vs comparison method."""

    TASKS_ONLY = [1, 0, 0, 0, 0, 0]
    COMMS_ONLY = [0, 0, 0, 0, 1, 0]
    EXPLORE_ONLY = [0, 0, 1, 0, 0, 0]

    ### CYCLE THROUGH TEST CONFIGS ###
    for test in range(max(len(scenario_configs), len(env_configs), len(model_configs))):
        print(f"Test {test} setup...")
        test_config = load_yaml_to_kwargs(test_configs[test])
                                              
        # Set up environments, policy, planner, and logs details
        env, policy, logs  = test_setup(test,
                                            scenario_configs, 
                                            env_configs, 
                                            model_configs, 
                                            checkpoint, 
                                            )
        
        env.base_env.render = True
        
        # Load in test parameters
        num_runs = test_config["num_runs"]
        rollout_steps = test_config["rollout_steps"]
        agent_assignments = test_config["agent_assignments"]
        planning_iters = test_config["planning_iters"]
        
        plan_heuristic_fns = ["goto_goal", "neediest_comms_midpt", "nearest_agent"]

        scenario_name = os.path.splitext(os.path.basename(scenario_configs[test]))[0]
        data_dir = os.path.join(folder_path, "data_"+str(scenario_name))
        os.makedirs(data_dir, exist_ok=True)
        csv_fp = os.path.join(data_dir, "results.csv")
        csv_header = [
            "test", "run",
            "policy_reward_mean", "policy_reward_sum", "policy_actions",
            "tasks_reward_mean", "tasks_reward_sum",
            "taskcomms_reward_mean", "taskcomms_reward_sum",
            "taskexp_reward_mean", "taskexp_reward_sum",
            "taskcommsexpA_reward_mean", "taskcommsexpA_reward_sum",
            "taskcommsexpB_reward_mean", "taskcommsexpB_reward_sum",
            "taskcommsexpC_reward_mean", "taskcommsexpC_reward_sum",
        ]

        for run in range(num_runs):

            seed = TensorDict({"seed": torch.tensor([run,])}, batch_size=[1])

            # Toggle rendering
            if run % 5 == 0:
                # Configure render
                env.base_env.render = True
                render_name = f"render_policy"
                render_fp = os.path.join(f"{folder_path}/gif_{scenario_name}/policy/", render_name)
                env.base_env.render_fp = render_fp
                env.base_env.count = run*rollout_steps
                os.makedirs(os.path.dirname(render_fp), exist_ok=True)
            else:
                env.base_env.render = False
                

            # ---- Run test with policy ----
            print(f"Preparing to run test {test} with policy...")

            # Set heuristics for policy
            env.base_env.reset_heuristic_eval_fns()
            
            # Run test & log   
            policy_reward_mean, policy_reward_sum, policy_actions = run_policy(env, logs, policy, rollout_steps, seed=seed)
            print("Mean reward policy:", logs["policy reward (mean)"])


            # ---- Run test with h_tasks only ----
            # Fix agent actions
            agents = env.base_env.sim_env.scenario.active_agents
            actions = []
            for _ in agents:
                actions += TASKS_ONLY
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            tasks_rew_mean, tasks_rew_sum = run_fixed_heuristics(env, logs, actions, rollout_steps, name="all_tasks", scenario_name=scenario_name, seed=run, folder_path=folder_path)            
            print("Mean all tasks:", logs["all_tasks reward (mean)"])


            # ---- Run test with split h_tasks, h_comms ----
            # Fix agent actions
            agents = env.base_env.sim_env.scenario.active_agents
            actions = []
            for i, _ in enumerate(agents):
                if i % 2 == 0:
                    actions += TASKS_ONLY
                else:
                    actions += COMMS_ONLY
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            taskcomms_rew_mean, taskcomms_rew_sum = run_fixed_heuristics(env, logs, actions, rollout_steps, name="taskcomms", scenario_name=scenario_name, seed=run, folder_path=folder_path)      
            print("Mean tasks/comms:", logs["task_comms reward (mean)"])


            # ---- Run test with split tasks, explore ----
            # Fix agent actions
            agents = env.base_env.sim_env.scenario.active_agents
            actions = []
            for i, _ in enumerate(agents):
                if i % 2 == 0:
                    actions += TASKS_ONLY
                else:
                    actions += EXPLORE_ONLY
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            taskexp_rew_mean, taskexp_rew_sum = run_fixed_heuristics(env, logs, actions, rollout_steps, name="taskexp", scenario_name=scenario_name, seed=run, folder_path=folder_path)      
            print("Mean tasks/explore:", logs["tasks_explore reward (mean)"])


            # ---- Run test with 2 tasks, 1 comms, 1 explore ----
            # Fix agent actions
            agents = env.base_env.sim_env.scenario.active_agents
            actions = []
            for i, _ in enumerate(agents):
                if i < 2:
                    actions += TASKS_ONLY
                elif i == 2:
                    actions += COMMS_ONLY
                elif i == 3:
                    actions += EXPLORE_ONLY
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            taskcommexpA_rew_mean, taskcommexpA_rew_sum = run_fixed_heuristics(env, logs, actions, rollout_steps, name="taskcommexpA", scenario_name=scenario_name, seed=run, folder_path=folder_path)      
            print("Mean 2task, 1comm, 1exp:", logs["taskcommexpA reward (mean)"])


            # ---- Run test with 1 tasks, 2 comms, 1 explore ----
            # Fix agent actions
            agents = env.base_env.sim_env.scenario.active_agents
            actions = []
            for i, _ in enumerate(agents):
                if i < 2:
                    actions += COMMS_ONLY
                elif i == 2:
                    actions += EXPLORE_ONLY
                elif i == 3:
                    actions += TASKS_ONLY
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            taskcommexpB_rew_mean, taskcommexpB_rew_sum = run_fixed_heuristics(env, logs, actions, rollout_steps, name="taskcommexpB", scenario_name=scenario_name, seed=run, folder_path=folder_path)      
            print("Mean 1task, 2comm, 1exp:", logs["taskcommexpB reward (mean)"])


            # ---- Run test with 1 tasks, 1 comms, 2 explore ----
            # Fix agent actions
            agents = env.base_env.sim_env.scenario.active_agents
            actions = []
            for i, _ in enumerate(agents):
                if i < 2:
                    actions += EXPLORE_ONLY
                elif i == 2:
                    actions += COMMS_ONLY
                elif i == 3:
                    actions += TASKS_ONLY
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            taskcommexpC_rew_mean, taskcommexpC_rew_sum = run_fixed_heuristics(env, logs, actions, rollout_steps, name="taskcommexpC", scenario_name=scenario_name, seed=run, folder_path=folder_path)      
            print("Mean 1task, 1comm, 2exp:", logs["taskcommexpC reward (mean)"])


            # ---- Save results to CSV ----
            write_header = not os.path.exists(csv_fp)
            with open(csv_fp, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(csv_header)
                writer.writerow([
                    test, run,
                    policy_reward_mean, policy_reward_sum, policy_actions,
                    tasks_rew_mean, tasks_rew_sum,
                    taskcomms_rew_mean, taskcomms_rew_sum,
                    taskexp_rew_mean, taskexp_rew_sum,
                    taskcommexpA_rew_mean, taskcommexpA_rew_sum,
                    taskcommexpB_rew_mean, taskcommexpB_rew_sum,
                    taskcommexpC_rew_mean, taskcommexpC_rew_sum,
                ])
        
        print("Done!")

def run_fixed_heuristics(env: TransformedEnv, logs: defaultdict, actions, rollout_steps: int, name: str, scenario_name: str, seed, folder_path="test"):
    print("Running fixed heuristics...")
    # render_name = f"render_{name}"
    # render_fp = os.path.join(f"{folder_path}/gif/{name}/", render_name)
    # env.base_env.render_fp = render_fp
    # env.base_env.count = seed["seed"][0]*rollout_steps
    # os.makedirs(os.path.dirname(render_fp), exist_ok=True)

    # Toggle rendering
    if seed % 5 == 0:
        # Configure render
        env.base_env.render = True
        render_name = f"render_{name}"
        render_fp = os.path.join(f"{folder_path}/gif_{scenario_name}/{name}/", render_name)
        env.base_env.render_fp = render_fp
        env.base_env.count = seed*rollout_steps
        os.makedirs(os.path.dirname(render_fp), exist_ok=True)
    else:
        env.base_env.render = False

    env.reset() # TODO seed
    # env.base_env._set_seed(seed)
    fixed_rollout_rewards = rollout_fixed_actions(env, actions, rollout_steps)
    reward_sum = sum(fixed_rollout_rewards[1:])
    reward_mean = reward_sum / len(fixed_rollout_rewards[1:])
    logs[name + " reward (mean)"].append(reward_mean)
    logs[name + " reward (sum)"].append(reward_sum)

    return reward_mean, reward_sum

def run_policy(env: TransformedEnv, logs: defaultdict, policy, rollout_steps: int, seed):
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # execute a rollout with the trained policy
        print("Running policy...")
        env.reset() # TODO seed
        # env.base_env._set_seed(seed)
        policy_rollout = env.rollout(rollout_steps, policy, return_contiguous=False)
        print("Policy reward shape:", policy_rollout["next", "reward"].shape)
        policy_reward_mean = policy_rollout["next", "reward"][0][1:].mean().item()  # Skip first step (startup)
        policy_reward_sum = policy_rollout["next", "reward"][0][1:].sum().item()  # Skip first step (startup)
        logs["policy reward (mean)"].append(policy_reward_mean)
        logs["policy reward (sum)"].append(policy_reward_sum)
        logs["action"] = policy_rollout["action"]

    return policy_reward_mean, policy_reward_sum, policy_rollout["action"]


def rollout_fixed_actions(env: TransformedEnv,
                            actions,
                            rollout_steps: int,
                            ):
    logs = {}
    logs["reward"] = 0.0
    rewards = []
    # For step in rollout_steps:
    for i in range(rollout_steps):

        # Roll out plan
        tdict = env.base_env._step(actions)
        rewards.append(tdict["reward"].item())

    
    # logs["reward"] += tdict["reward"].item()

    return rewards



if __name__ == "__main__":

    if len(sys.argv) < 6:
        print("Usage: python run_tests.py <scenario_fp> <env_fp> <test_fp> <model_fp> <checkpt_fp> <comp_fp> <folder_name>")
        sys.exit(1)

    scenario_fp = sys.argv[1]
    env_fp = sys.argv[2]
    test_fp = sys.argv[3]
    model_fp = sys.argv[4]
    checkpt_fp = sys.argv[5]
    test_folder_name = sys.argv[6]

    # Env, Scenario & params
    scenario = Scenario()  
    scenario_configs = [scenario_fp]
        # "conf/scenarios/exploring_0.yaml", #SR_tasks_5.yaml",
    # ]
    env_configs = [env_fp]
        # "conf/envs/planning_env_explore_1.yaml", #planning_env_vec_4.yaml",
    # ]

    test_configs = [test_fp]
        # "conf/envs/planning_env_explore_1.yaml", #planning_env_vec_4.yaml",
    # ]

    # Model Params
    model_configs = [model_fp]
        # "conf/models/mat_9.yaml",
    # ]

    # Checkpoint
    checkpoint = checkpt_fp


    run_tests(scenario_configs,
              env_configs,
              test_configs,
              model_configs,
              checkpoint,
              folder_path=test_folder_name
              )
    


# python evaluations/comparisons.py "conf/scenarios/comms_5.yaml" "conf/envs/planning_env_explore_5_1env.yaml"  "evaluations/tests/trial1.yaml" "evaluations/configs_weights/policy_configs/fullTF.yaml" "evaluations/configs_weights/policy_weights/dense_fullTF.pt" "evaluations/fixed_comparisons"