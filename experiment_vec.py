
import multiprocessing
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import wandb
import yaml
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv, CatTensors)
from torchrl.envs.utils import (ExplorationType, check_env_specs,
                                set_exploration_type)
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from envs.planning_env_vec import VMASPlanningEnv
from models.transformer import EnvironmentTransformer


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



def train_PPO(scenario,
              scenario_configs,
              env_configs,
              rl_configs,
              model_configs,
              use_wandb=False
              ):
    
    ### HYPERPARAMS ###
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    ### CYCLE THROUGH TEST CONFIGS ###
    for test in range(max(len(scenario_configs), len(env_configs), len(rl_configs), len(model_configs))):

        # SET UP DATA LOGS #
        scen_name = scenario_configs[test].split('/')[-1].split('.')[0]
        env_name = env_configs[test].split('/')[-1].split('.')[0]
        rl_name = rl_configs[test].split('/')[-1].split('.')[0]
        model_name = model_configs[test].split('/')[-1].split('.')[0]
        test_name = f"{scen_name}_{env_name}_{rl_name}_{model_name}"
        test_folder_path = os.path.join("runs", test_name)
        os.makedirs(os.path.dirname(test_folder_path), exist_ok=True)

        ### INIT WANDB ###
        if use_wandb:
            run = wandb.init(
                entity="nlbutler18-oregon-state-university",
                project="mothership",
                name=test_name,
            )

        # LOAD CONFIGS #
        scenario_config = load_yaml_to_kwargs(scenario_configs[test])
        env_config = load_yaml_to_kwargs(env_configs[test])
        rl_config = load_yaml_to_kwargs(rl_configs[test])
        model_config = load_yaml_to_kwargs(model_configs[test])

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

        # DEFINE ENVIRONMENT #
        base_env = VMASPlanningEnv(scenario,
                                    device=device,
                                    env_kwargs=env_config,
                                    scenario_kwargs=scenario_config,
                                    )
        
        env = TransformedEnv(
            base_env,
            Compose(
                # normalize observations
                # ObservationNorm(in_keys=[("obs", "cell_feats")]),
                # ObservationNorm(in_keys=[("obs", "cell_pos")]),
                # ObservationNorm(in_keys=[("obs", "rob_pos")]),
                DoubleToFloat(in_keys=[("obs", "cell_feats"), ("obs", "cell_pos"), ("obs", "rob_pos")]),
                StepCounter(),
            ),
            device=device
        )
        # env.append_transform(
        #     CatTensors(in_keys=[
        #         ("obs", "cell_feats"),
        #         ("obs", "cell_pos"),
        #         ("obs", "rob_pos"),
        #         ],
        #                out_keys=["observation"]
        #     )
        # )
        # env.append_transform(StepCounter())

        # Initialize stats
        # env.transform[0].init_stats(num_iter=10)#, reduce_dim=0, cat_dim=0, keep_dims=False)
        # print("normalization constant shape:\n", env.transform[0].loc.shape)
        # env.transform[1].init_stats(num_iter=10)#, reduce_dim=0, cat_dim=0, keep_dims=False)
        # print("normalization constant shape:\n", env.transform[1].loc.shape)
        # env.transform[2].init_stats(num_iter=10)#, reduce_dim=0, cat_dim=0, keep_dims=False)
        # print("normalization constant shape:\n", env.transform[2].loc.shape)

        # # Evaluate environment initialization
        # print("normalization constant shape:\n", env.transform[0].loc.shape)

        print("observation_spec:\n", env.observation_spec)
        print("reward_spec:\n", env.reward_spec)
        print("input_spec:\n", env.input_spec)
        print("action_spec (as defined by input_spec):\n", env.action_spec)

        check_env_specs(env)

        rollout = env.rollout(3)
        print("rollout of three steps:", rollout)
        print("Shape of the rollout TensorDict:", rollout.batch_size)

        ### ACTOR & CRITIC POLICY MODULES ###

        # num_cells = model_config["num_cells"]
        num_features = model_config["num_features"]
        num_heuristics = model_config["num_heuristics"]
        d_model = model_config["d_model"]
        #num_heads
        #num_layers
        
        mat = EnvironmentTransformer(num_features=num_features,
                                           num_heuristics=num_heuristics,
                                           d_model=d_model,
                                           ).to(device)

        policy_module = TensorDictModule(
            mat,
            in_keys=[("obs", "cell_feats"), ("obs", "cell_pos"), ("obs", "rob_pos")],
            out_keys=["foo","loc","scale"]
        )

        # print("LOW:", env.action_spec_unbatched.space.low)
        # print("HIGH:", env.action_spec_unbatched.space.high)

        # break

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
            module=mat,
            in_keys=[("obs", "cell_feats"), ("obs", "cell_pos"), ("obs", "rob_pos")],
            out_keys=["state_value","foo","bar"]
        )

        print("Running policy:", policy_module(env.reset()))
        print("Running value:", value_module(env.reset())) # NOTE leave this in to init lazy modules

        collector = SyncDataCollector(
            env,
            policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        advantage_module = GAE(
            gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
        )

        loss_module = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )

        optim = torch.optim.Adam(loss_module.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // frames_per_batch, 0.0
        )
        

        ### TRAINING LOOP ###
        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        eval_str = ""
        best_reward = float("-inf")
        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(collector):
            # we now have a batch of data to work with. Let's learn something from it.
            # print("\n!! Running Training")
            for ep in range(num_epochs):
                # print("=== EPOCH:", ep, "===")
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.

                # print("!! TDict Data:\n", tensordict_data)
                # Compute advantage values and add to tdict
                advantage_module(tensordict_data)

                # print("Sample log prob:", tensordict_data["sample_log_prob"])
                # print("Advantage:", tensordict_data["advantage"])
                
                data_view = tensordict_data.reshape(-1)

                replay_buffer.extend(data_view.to(device))

                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size)
                    # print("\nSUBDATA:", subdata)
                    # print("Sample log prob:", subdata["sample_log_prob"])
                    # print("Advantage:", subdata["advantage"])
                    loss_vals = loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )
                    # print("\nLOSS Obj:", loss_vals["loss_objective"])
                    # print("LOSS Crit:", loss_vals["loss_critic"])
                    # print("LOSS Entr:", loss_vals["loss_entropy"])
                    logs["loss_objective"].append(loss_vals["loss_objective"].item())
                    logs["loss_critic"].append(loss_vals["loss_critic"].item())
                    logs["loss_entropy"].append(loss_vals["loss_entropy"].item())
                    if use_wandb:
                        run.log({"train/loss_objective": loss_vals["loss_objective"].item()})
                        run.log({"train/loss_critic": loss_vals["loss_critic"].item()})
                        run.log({"train/loss_entropy": loss_vals["loss_entropy"].item()})
                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()


            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                f"| Training average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            logs["lr"].append(optim.param_groups[0]["lr"])
            # stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            if use_wandb:
                run.log({"train/mean_reward": tensordict_data["next", "reward"].mean().item()})
                run.log({"train/step_count": tensordict_data["step_count"].max().item()})
                run.log({"train/lr": optim.param_groups[0]["lr"]})

            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # print("\n!! Running Evaluation")
                env.base_env.render = True
                env.base_env.count = 0
                render_name = f"render_{i}"
                render_fp = os.path.join(f"{test_folder_path}/gif/", render_name)
                env.base_env.render_fp = render_fp
                os.makedirs(os.path.dirname(render_fp), exist_ok=True)
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = env.rollout(3, policy_module)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {logs['eval reward (sum)'][0]: 4.4f})"
                        # f"eval step-count: {logs['eval step_count'][-1]}"
                    )
                    if use_wandb:
                        run.log({"eval/mean_reward": eval_rollout["next", "reward"].mean().item()})
                        run.log({"eval/cum_reward": eval_rollout["next", "reward"].sum().item()})
                        run.log({"eval/step_count": eval_rollout["step_count"].max().item()})

                    # Save best models
                    if eval_rollout["next", "reward"].mean().item() > best_reward:
                        best_reward = eval_rollout["next", "reward"].mean().item()
                        checkpoint_name = f"best.pt"
                        checkpoint_path = os.path.join(f"{test_folder_path}/checkpoints/", checkpoint_name)
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        torch.save({
                            'step': i,
                            'tf_state_dict': mat.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'logs': logs,
                        }, checkpoint_path)

                    del eval_rollout
                env.base_env.render = False

                # Save model checkpoints
                checkpoint_name = f"checkpoint_step_{i}.pt"
                checkpoint_path = os.path.join(f"{test_folder_path}/checkpoints/", checkpoint_name)
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'step': i,
                    'tf_state_dict': mat.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'logs': logs,
                }, checkpoint_path)
                # print(f"Saved checkpoint to {checkpoint_path}")

            pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str])) #stepcount_str,

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            scheduler.step()

        if use_wandb:
            run.finish()

        ### SHOW RESULTS ###
        # plt.figure(figsize=(10, 10))
        # plt.subplot(2, 2, 1)
        # plt.plot(logs["reward"])
        # plt.title("training rewards (average)")
        # plt.subplot(2, 2, 2)
        # plt.plot(logs["loss_objective"])
        # plt.title("Obj loss (training)")
        # plt.subplot(2, 2, 3)
        # plt.plot(logs["eval reward (sum)"])
        # plt.title("Return (test)")
        # plt.subplot(2, 2, 4)
        # plt.plot(logs["loss_critic"])
        # plt.title("Critic loss (training)")
        # plt.show()
