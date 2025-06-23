
import multiprocessing
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import yaml
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (CatTensors, Compose, DoubleToFloat, ObservationNorm,
                          ParallelEnv, StepCounter, TransformedEnv)
from torchrl.envs.utils import (ExplorationType, check_env_specs,
                                set_exploration_type)
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

import wandb
from envs.planning_env_vec import VMASPlanningEnv
from models.transformer import (EnvironmentCriticTransformer,
                                EnvironmentTransformer)


def load_yaml_to_kwargs(file_path: str) -> None:
    """
    Load parameters from a YAML file and pass them as keyword arguments to a function.

    Args:
        file_path (str): Relative or absolute path to the YAML file.
        func (callable): The function to which the YAML parameters will be passed as kwargs.

    Returns:
        None
    """
    with open(file_path, 'r') as file:
        try:
            params = yaml.safe_load(file)
            if not isinstance(params, dict):
                raise ValueError("YAML file must contain a dictionary at the top level.")
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file {file_path}: {exc}")
        except ValueError as exc:
            print(f"Error: {exc}")

    return params


def sweep(scenario,
            scenario_configs,
            env_configs,
            model_configs,
            sweep_configs,
            project_name=None,
            entity=None,
            conf_name=None
            ):

    sweep_id = wandb.sweep(sweep=sweep_configs,
                           entity=entity,
                           project=project_name+"-sweep"
                           )
    
    wandb.agent(sweep_id,
                function=lambda: run_sweep(
                    scenario,
                    scenario_configs,
                    env_configs,
                    model_configs,
                    project_name,
                    entity,
                    conf_name,
                ),
                count=10
                )
    
def run_sweep(scenario,
         scenario_configs,
         env_configs,
         model_configs,
         project_name,
         entity,
         conf_name,
         ):
    
    wandb.init(entity=entity,
               project=project_name+"-sweep",
               name=conf_name
               )
    train_PPO_sweep(scenario,
                    scenario_configs,
                    env_configs,
                    model_configs,
                    wandb.config,
                    )

    
def train_PPO_sweep(scenario,
                    scenario_configs,
                    env_configs,
                    model_configs,
                    sweep_configs,
                    ):

    # SET UP DATA LOGS #
    scen_name = scenario_configs[0].split('/')[-1].split('.')[0]
    env_name = env_configs[0].split('/')[-1].split('.')[0]
    model_name = model_configs[0].split('/')[-1].split('.')[0]
    file_name = f"{scen_name}_{env_name}_{model_name}"
    test_name = "_".join(list(sweep_configs.keys()))
    test_folder_path = os.path.join("runs", file_name, test_name)
    os.makedirs(os.path.dirname(test_folder_path), exist_ok=True)

    # LOAD CONFIGS #
    scenario_config = load_yaml_to_kwargs(scenario_configs[0])
    env_config = load_yaml_to_kwargs(env_configs[0])
    model_config = load_yaml_to_kwargs(model_configs[0])

    # PPO PARAMS #
    lr = sweep_configs.lr
    max_grad_norm = sweep_configs.max_grad_norm
    frames_per_batch = sweep_configs.frames_per_batch # training batch size
    total_frames = sweep_configs.total_frames # total frames collected
    sub_batch_size = sweep_configs.sub_batch_size  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = sweep_configs.num_epochs  # optimization steps per batch of data collected
    clip_epsilon = (sweep_configs.clip_epsilon) # clip value for PPO loss: see the equation in the intro for more context.
    gamma = sweep_configs.gamma
    lmbda = sweep_configs.lmbda
    entropy_eps = sweep_configs.entropy_eps

    train_PPO(scenario,
              scenario_config,
              env_config,
              model_config,
              lr,
              total_frames,
              frames_per_batch,
              sub_batch_size,
              num_epochs,
              gamma,
              lmbda,
              max_grad_norm,
              entropy_eps,
              clip_epsilon,
              test_folder_path,
              test_name,
              wandb_mode="SWEEP"
              )
    

def train(scenario,
          scenario_configs,
          env_configs,
          rl_configs,
          model_configs,
          checkpt_fp=None,
          wandb_mode="TRAIN",
          project_name=None
          ):
    
    ### CYCLE THROUGH TEST CONFIGS ###
    for test in range(max(len(scenario_configs), len(env_configs), len(rl_configs), len(model_configs))):

        # LOAD CONFIGS #
        scenario_config = load_yaml_to_kwargs(scenario_configs[test])
        env_config = load_yaml_to_kwargs(env_configs[test])
        rl_config = load_yaml_to_kwargs(rl_configs[test])
        model_config = load_yaml_to_kwargs(model_configs[test])
        if checkpt_fp:
            torch.serialization.add_safe_globals([defaultdict])
            torch.serialization.add_safe_globals([list])
            checkpt_data = torch.load(checkpt_fp, weights_only=True)
        else:
            checkpt_data = None

        # PPO PARAMS #
        lr = rl_config["lr"]
        max_grad_norm = rl_config["max_grad_norm"]
        frames_per_batch = rl_config["frames_per_batch"] # training batch size
        total_frames = rl_config["total_frames"] # total frames collected
        sub_batch_size = rl_config["sub_batch_size"]  # 64 # cardinality of the sub-samples gathered from the current data in the inner loop
        num_epochs = rl_config["num_epochs"]  # optimization steps per batch of data collected
        clip_epsilon = (rl_config["clip_epsilon"]) # clip value for PPO loss: see the equation in the intro for more context.
        gamma = rl_config["gamma"]
        lmbda = rl_config["lambda"]
        entropy_eps = rl_config["entropy_eps"]

        # SET UP DATA LOGS #
        scen_name = scenario_configs[test].split('/')[-1].split('.')[0]
        env_name = env_configs[test].split('/')[-1].split('.')[0]
        rl_name = rl_configs[test].split('/')[-1].split('.')[0]
        model_name = model_configs[test].split('/')[-1].split('.')[0]
        test_name = f"{scen_name}_{env_name}_{rl_name}_{model_name}"
        test_folder_path = os.path.join("runs", test_name)
        os.makedirs(os.path.dirname(test_folder_path), exist_ok=True)


        train_PPO(scenario,
                  scenario_config,
                  env_config,
                  model_config,
                  lr,
                  total_frames,
                  frames_per_batch,
                  sub_batch_size,
                  num_epochs,
                  gamma,
                  lmbda,
                  max_grad_norm,
                  entropy_eps,
                  clip_epsilon,
                  test_folder_path,
                  test_name,
                  wandb_mode=wandb_mode,
                  project_name=project_name,
                  checkpt_data=checkpt_data,
                  )



def train_PPO(scenario,
              scenario_config,
              env_config,
              model_config,
              lr,
              total_frames,
              frames_per_batch,
              sub_batch_size,
              num_epochs,
              gamma,
              lmbda,
              max_grad_norm,
              entropy_eps,
              clip_epsilon,
              test_folder_path,
              test_name,
              wandb_mode=None,
              project_name=None,
              checkpt_data=None,
              ):
    
    ### INIT WANDB ###
    if wandb_mode=="TRAIN":
        resuming = None
        if checkpt_data is not None:
            resuming = "allow"
        run = wandb.init(
            entity="nlbutler18-oregon-state-university",
            project=project_name,
            name=test_name,
            id=test_name,
            resume=resuming
        )
    

    ### HYPERPARAMS ###
    torch.set_num_threads(1) # NOTE Recommended by Manuel to accelerate training
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    # DEFINE ENVIRONMENT #
    num_envs = env_config["num_envs"]
    base_env = VMASPlanningEnv(scenario,
                                device=device,
                                env_kwargs=env_config,
                                scenario_kwargs=scenario_config,
                                )

    env = TransformedEnv(
        base_env,
        Compose(
            # ObservationNorm(in_keys=["cell_feats"]),
            # ObservationNorm(in_keys=["cell_pos"]),
            # ObservationNorm(in_keys=["rob_pos"]),
            DoubleToFloat(in_keys=["cell_feats", "cell_pos", "rob_pos"]),
            StepCounter(),
        ),
        device=device,
    )

    # Initialize observation norm stats
    # for t in env.transform:
    #     if isinstance(t, ObservationNorm):
    #         print("Normalizing obs", t.in_keys)
    #         t.init_stats(num_iter=10*num_envs, reduce_dim=[0,1], cat_dim=0) # num_iter should be divisible (?) or match (?) horizon in env

    check_env_specs(env)

    ### ACTOR & CRITIC POLICY MODULES ###
    num_features = model_config["num_features"]
    num_heuristics = model_config["num_heuristics"]
    d_feedforward = model_config["d_feedforward"]
    d_model = model_config["d_model"]

    tf_act = EnvironmentTransformer(num_features=num_features,
                                        num_heuristics=num_heuristics,
                                        d_feedforward=d_feedforward,
                                        d_model=d_model,
                                        ).to(device)
    
    tf_crit = EnvironmentCriticTransformer(num_features=num_features,
                                            d_model=d_model,
                                            use_attention_pool=True,
                                            ).to(device)

    policy_module = TensorDictModule(
        tf_act,
        in_keys=[("cell_feats"), ("cell_pos"), ("num_cells"), ("rob_pos"), ("num_robs")],
        out_keys=["loc","scale"]
    )

    policy_module = ProbabilisticActor( # Actions sampled from dist during exploration
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec_unbatched.space.low,
            "high": env.action_spec_unbatched.space.high,
        },
        return_log_prob=True,
        # NOTE we need the log-prob for the numerator of the importance weights
    )

    value_module = ValueOperator(
        module=tf_crit,
        in_keys=[("cell_feats"), ("cell_pos"), ("num_cells"),],
        out_keys=["state_value"]
    )

    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset())) # NOTE leave this in to init lazy modules

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # split_trajs=True,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(shuffle=False),
    )

    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        # average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        # critic_coef=1.0, #0.01,
        loss_critic_type="smooth_l1",
        # normalize_advantage=True,
        # clip_value=False,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    if checkpt_data is not None:
        tf_act.load_state_dict(checkpt_data['actor_state_dict'])
        tf_crit.load_state_dict(checkpt_data['actor_state_dict'])
        optim.load_state_dict(checkpt_data['optimizer_state_dict'])
        scheduler.load_state_dict(checkpt_data['scheduler_state_dict'])
        env.load_state_dict(checkpt_data['env_state_dict'])
        logs = checkpt_data['logs']
    else:
        logs = defaultdict(list)
    
    ### TRAINING LOOP ###
    pbar = tqdm(total=total_frames)
    eval_str = ""
    best_reward = float("-inf")
    # Iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # Learn from collected data batch
        for ep in range(num_epochs):
            
            # Initial batch data prep for vectorized environment
            data = tensordict_data.flatten(0,1) # Flatten trajectory to fix batching from collector
            data["sample_log_prob"] = data["sample_log_prob"].sum(dim=-1)#.unsqueeze(-1) # Sum log prob for each agent

            # We re-compute advantage at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(data)

            # Print debug stats
            if ep % 8 == 0:
                print("\n EP:", ep)
                print("\nTraj IDs:\n", data["collector", "traj_ids"][:10].cpu().tolist())
                print("\nSample log prob:\n", data["sample_log_prob"][:10].cpu().tolist())
                print("\nAdvantage:\n", data["advantage"][:10].cpu().tolist())
                print("\nstate_value:\n", data["state_value"][:10].cpu().tolist())
                print("\nvalue_target:\n", data["value_target"][:10].cpu().tolist())
                print("\nreward:\n", data["next", "reward"][:10].cpu().tolist())
                print("\ndone:\n", data["next", "done"][:10].cpu().tolist())
                print("\nterminated:\n", data["next", "terminated"][:10].cpu().tolist())

            assert ~data["advantage"].isnan().any(), "NaN detected! Terminating..."

            replay_buffer.extend(data.to(device))

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                # print("SUBDATA:\n", subdata)
                # print("Subdata Advantage:", subdata["advantage"])
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                logs["loss_objective"].append(loss_vals["loss_objective"].item())
                logs["loss_critic"].append(loss_vals["loss_critic"].item())
                logs["loss_entropy"].append(loss_vals["loss_entropy"].item())

                if wandb_mode != None:
                    wandb.log({"train/loss_objective": loss_vals["loss_objective"].item()})
                    wandb.log({"train/loss_critic": loss_vals["loss_critic"].item()})
                    wandb.log({"train/loss_entropy": loss_vals["loss_entropy"].item()})
                # wandb.log({"train/value_clip_fraction": loss_vals["value_clip_fraction"].item()})
                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()


        logs["reward"].append(data["next", "reward"].mean().item())
        pbar.update(data.numel())
        cum_reward_str = (
            f"| Training average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(data["step_count"].max().item())
        logs["lr"].append(optim.param_groups[0]["lr"])

        if wandb_mode != None:
            wandb.log({"train/mean_reward": data["next", "reward"].mean().item()})
            wandb.log({"train/step_count": data["step_count"].max().item()})
            wandb.log({"train/lr": optim.param_groups[0]["lr"]})
            wandb.log({"train/advantage_mean_abs": data["advantage"].abs().mean().item()})
            wandb.log({"train/advantage_std": data["advantage"].std().item()})
            wandb.log({"train/state_value_mean": data["state_value"].mean().item()}) 
            wandb.log({"train/value_target_mean": data["value_target"].mean().item()})
            wandb.log({f"actions/rob{j}_action{i}_mean": data["action"][:, j, i].mean().item() for i in range(num_heuristics) for j in range(base_env.sim_env.n_agents)})

        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # Run evaluation
            env.base_env.render = True
            env.base_env.count = 0
            render_name = f"render_{i}"
            render_fp = os.path.join(f"{test_folder_path}/gif/", render_name)
            env.base_env.render_fp = render_fp
            os.makedirs(os.path.dirname(render_fp), exist_ok=True)
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(16, policy_module, return_contiguous=False)
                env.reset()
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f})"
                )

                if wandb_mode != None:
                    wandb.log({"eval/mean_reward": eval_rollout["next", "reward"].mean().item()})
                    wandb.log({"eval/cum_reward": eval_rollout["next", "reward"].sum().item()})
                    wandb.log({"eval/step_count": eval_rollout["step_count"].max().item()})
                    wandb.log({f"eval/env0_rob{j}_action{i}": eval_rollout["action"][0, j, i] for i in range(num_heuristics) for j in range(base_env.sim_env.n_agents)})

                del eval_rollout
            env.base_env.render = False

        # Save checkpoints & best-performing models, including environment state
        if data["next", "reward"].mean().item() > best_reward or i % 5 == 0:
            checkpt_data = {
                'step': i,
                'actor_state_dict': tf_act.state_dict(),
                'critic_state_dict': tf_crit.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'logs': logs,
                'env_state_dict': env.state_dict() if hasattr(env, "state_dict") else None,
                }
            checkpoint_name = f"checkpt_{i}.pt"
            checkpoint_path = os.path.join(f"{test_folder_path}/checkpoints/", checkpoint_name)
            save_checkpt(checkpoint_path, checkpt_data)

            if data["next", "reward"].mean().item() > best_reward:
                best_reward = data["next", "reward"].mean().item()
                checkpoint_name = f"best.pt"
                checkpoint_path = os.path.join(f"{test_folder_path}/checkpoints/", checkpoint_name)
                save_checkpt(checkpoint_path, checkpt_data)
            
        pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str])) #stepcount_str,

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    if wandb_mode == "TRAIN":
        run.finish()    


def save_checkpt(checkpt_path, checkpt_data):
    os.makedirs(os.path.dirname(checkpt_path), exist_ok=True)
    torch.save(checkpt_data, checkpt_path)
