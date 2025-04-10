import copy
import multiprocessing
import time
from typing import Union
import os

import numpy as np
import torch
from moviepy import ImageSequenceClip
from envs.scenarios.SR_tasks import Scenario
from vmas import make_env
# from vmas.simulator.scenario import BaseScenario
from envs.planning_env import VMASPlanningEnv

# TORCHRL
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from torch import nn

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, SliceSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    EnvBase,
    ParallelEnv
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm


def train_PPO(scenario):
    ### HYPERPARAMS ###
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    num_cells = 16
    batch_size = None # num_envs
    node_dim = 4
    lr = 1e-3
    max_grad_norm = 1.0
    # For a complete training, bring the number of frames up to 1M
    frames_per_batch = 256 # training batch size
    total_frames = 10*1024 # total frames collected

    ### PPO PARAMS ###
    sub_batch_size = 64  # 64 # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 16  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    ### DEFINE ENVIRONMENT ###
    base_env = VMASPlanningEnv(scenario,
                          num_envs=batch_size,
                          device=device,
                          node_dim=node_dim
                          )
    
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["x"]), # Change to "observation"
            DoubleToFloat(),
            StepCounter(),
        ),
        device=device
    )

    # env = ParallelEnv(
    #     num_envs=2,
    #     create_env_fn=base_env,
    #     device=device
    # )

    env.transform[0].init_stats(num_iter=2, reduce_dim=0, cat_dim=0)

    # Evaluate environment initialization
    print("normalization constant shape:\n", env.transform[0].loc.shape)

    print("observation_spec:\n", env.observation_spec)
    print("reward_spec:\n", env.reward_spec)
    print("input_spec:\n", env.input_spec)
    print("action_spec (as defined by input_spec):\n", env.action_spec)

    check_env_specs(env)

    rollout = env.rollout(3)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)

    ### DEFINE ACTOR POLICY & POLICY MODULE ###

    actor_net = nn.Sequential(
        nn.LazyLinear(2*num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        actor_net, in_keys=["x"], out_keys=["loc", "scale"]
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
        # we'll need the log-prob for the numerator of the importance weights
    )


    value_net = nn.Sequential(

        nn.LazyLinear(2*num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["x"],
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

    # env.base_env.render = True

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        print("\n!! Running Training")
        for ep in range(num_epochs):
            # print("=== EPOCH:", ep, "===")
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.

            # print("!! TDict Data:\n", tensordict_data)
            # Compute advantage values and add to tdict
            advantage_module(tensordict_data)
            
            data_view = tensordict_data.reshape(-1)

            replay_buffer.extend(data_view.to(device))

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                logs["loss_objective"].append(loss_vals["loss_objective"].item())
                logs["loss_critic"].append(loss_vals["loss_critic"].item())
                logs["loss_entropy"].append(loss_vals["loss_entropy"].item())
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
        # logs["step_count"].append(tensordict_data["step_count"].max().item())
        # stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # Save model checkpoints
            checkpoint_path = f"checkpoints/checkpoint_step_{i}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'step': i,
                'actor_state_dict': actor_net.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'logs': logs,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            print("\n!! Running Evaluation")
            env.base_env.render = True
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
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    # f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
            env.base_env.render = False
        pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str])) #stepcount_str,

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()


    ### SHOW RESULTS ###
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["loss_objective"])
    plt.title("Obj loss (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["loss_critic"])
    plt.title("Critic loss (training)")
    plt.show()




if __name__ == "__main__":

    ### List test config files here ###

    # Scenario & params
    scenario = Scenario()


    # RL Hyperparams


    # Model Params


    
    train_PPO(scenario,
              scenario_configs,
              rl_configs,
              model_configs,
              )


    # == Problem/Project Notes ==

    # 1. We have a set of "candidate behaviors/specializations" for passenger
    # 2. Mothership/Agent learns to prioritize these behaviors/specializations given
    # state of environment to maximize reward G.

    # Why not have Mothership/Agent use a sampling-based planner to find best behavior?
    # 1. Can be expensive/slow (especially for multiagent)
    # 2. Communication overhead (especially for multiagent)
    # 3 Information mismatch: Mothership/Agent priorization observation space is different
    # from the operating agent's observation space.