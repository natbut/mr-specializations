import copy
import multiprocessing
import time
from typing import Union

import numpy as np
import torch
from moviepy import ImageSequenceClip
from scenarios.SR_tasks import Scenario
from vmas import make_env
# from vmas.simulator.scenario import BaseScenario
from planning_env import VMASPlanningEnv

# TORCHRL
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

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
    EnvBase
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from torch_geometric.nn import Sequential, GATv2Conv, GCNConv, global_mean_pool
from torch_geometric.data import Batch, Data

def train_PPO(scenario):
    ### HYPERPARAMS ###
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    batch_size = 2
    node_dim = 4
    lr = 3e-4
    max_grad_norm = 1.0
    # For a complete training, bring the number of frames up to 1M
    frames_per_batch = 4
    total_frames = 1_024

    ### PPO PARAMS ###
    sub_batch_size = 32 # 64 # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    ### DEFINE ENVIRONMENT ###
    env = VMASPlanningEnv(scenario,
                          batch_size=batch_size,
                          device=device,
                          node_dim=node_dim
                          )

    ### DEFINE ACTOR POLICY & POLICY MODULE ###
    hidden = 16
    in_channels = env.action_spec.shape[-1] # NOTE this is num of node features, which FOR NOW is equal to action shape
    actor_net = Sequential(
        'x, edge_index', 
        [
            # Transform TorchRL obs to PyG input
            (lambda x, edge_index: (
                [Data(x=x_i, edge_index=edges_i) for x_i, edges_i in zip(x, edge_index)]
            ), 'x, edge_index -> graphs'),
            (lambda graphs: (
                Batch.from_data_list(graphs)
            ), 'graphs -> batch'),
            (lambda batch: (
                batch.x,
                batch.edge_index
            ), 'batch -> x0, edge_index0'),
            # Use GNN
            (GCNConv(in_channels, hidden), 'x0, edge_index0 -> x1'),
            nn.ReLU(inplace=True,),
            (GCNConv(hidden, hidden), 'x1, edge_index0 -> x2'),
            nn.ReLU(inplace=True),
            (nn.Linear(hidden, 2 * env.action_spec.shape[-1], device=device), 'x2 -> x3'),  # should reduce to 2*n_features(*n_nodes?)
            (NormalParamExtractor(), 'x3 -> loc, scale'),
            # Tansform back to TorchRL format
            (lambda loc, scale: (loc.view(-1, node_dim**2, env.action_spec.shape[-1]),
                                 scale.view(-1, node_dim**2, env.action_spec.shape[-1])),
                                 'loc, scale -> loc, scale'
                                 ),  # Reshape
        ],
    ).to(device)

    policy_module = TensorDictModule(
        actor_net, in_keys=["x", "edge_index"], out_keys=["loc", "scale"],
    )

    actor = ProbabilisticActor(
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
    ).to(device)

    
    print("!!! Running policy:", actor(env.reset()))

    ### DEFINE VALUE NET ###
    value_net = Sequential(
        'x, edge_index',
        [
            # Transform TorchRL obs to PyG input
            (lambda x, edge_index: (
                ([
                    Data(x=x_i, edge_index=edges_i) for x_i, edges_i in zip(x, edge_index)
                ])# if x.dim() == 3 else
                # ([
                #     Data(x=x_i, edge_index=edges_i) for x_i, edges_i in zip(x.view(-1, x.shape[2], x.shape[3]),
                #                                                             edge_index.view(-1, edge_index.shape[2], edge_index.shape[3]))
                # ])
            ), 'x, edge_index -> graphs'),
            (lambda graphs: (
                Batch.from_data_list(graphs)
            ), 'graphs -> batch'),
            (lambda batch: (
                batch.x,
                batch.edge_index,
                batch.batch
            ), 'batch -> x0, edge_index0, batch_size'),
            # Use GNN
            (GCNConv(in_channels, hidden), 'x0, edge_index0 -> x1'),
            nn.ReLU(inplace=True),
            (GCNConv(hidden, hidden), 'x1, edge_index0 -> x2'),
            nn.ReLU(inplace=True),
            # Pooling to aggregate node features into a graph-level representation
            # NOTE consider other pooling methods
            (global_mean_pool, 'x2, batch_size -> x3'),  # Global mean pooling
            (nn.Linear(hidden, 1), 'x3 -> state_value'),
        ],
    ).to(device)
    
    value_module = ValueOperator(
        module=value_net,
        in_keys=["x", "edge_index"],
        # out_keys=["state_value"]
    ).to(device)

    # print("!!! Running value:", value_module(env.reset()))

    ### COLLECTOR ###
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # split_trajs=False,
        device=device,
    )

    ### REPLAY BUFFER ###
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        # sampler=SliceSampler(num_slices=4, traj_key=("collector", "traj_ids"))
    )

    ### LOSS FUNCTION ###
    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
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

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for ep in range(num_epochs):
            print("\n === EPOCH:", ep, "===")
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.

            # print("!! TDict Data:\n", tensordict_data)
            # TODO Compute advantage values and add to tdict
            reshaped_data = tensordict_data.reshape(-1)
            print("Reshaped TDict Data:\n", reshaped_data)
            print("!! Computing advantage...")
            # advantage_module(tensordict_data)

            # TODO Append outputs to tensordict_data for logging

            print("!! Adding experience to buffer")
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                print("!! Backprop loss ...")
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        # logs["step_count"].append(tensordict_data["step_count"].max().item())
        # stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, actor)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                # logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    # f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
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
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()




if __name__ == "__main__":
    
    train_PPO(Scenario())
