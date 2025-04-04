import copy
import multiprocessing
import time
from typing import Union
import os

import numpy as np
import torch
from moviepy import ImageSequenceClip
from scenarios.SR_tasks import Scenario
from vmas import make_env
# from vmas.simulator.scenario import BaseScenarion 
from planning_env import VMASPlanningEnv

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
from torch.distributions import Normal


def train_PPO(scenario):
    ### HYPERPARAMS ###
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    batch_size = 4 # num_envs
    node_dim = 4
    lr = 3e-2
    max_grad_norm = 1.0
    # For a complete training, bring the number of frames up to 1M
    frames_per_batch = 8 # training batch size
    total_frames = 10*1024 # total frames collected

    ### PPO PARAMS ###
    sub_batch_size = 16  # 64 # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 32  # optimization steps per batch of data collected
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
    in_channels = env.action_spec.shape[-1] # NOTE this is num of node features, which FOR NOW is equal to action shape
    hidden = env.action_spec.shape[-1]
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

    ### DEFINE VALUE NET ###
    value_net = Sequential(
        'x, edge_index',
        [
            (lambda x, edge_index: (
                [Data(x=x_i, edge_index=edges_i) for x_i, edges_i in zip(x,edge_index)]
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

    ### COLLECTOR ###
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        # split_trajs=False,
        reset_at_each_iter=True,
        device=device,
    )

    ### REPLAY BUFFER ###
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        # sampler=SamplerWithoutReplacement(),
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
    td = env.reset()

    for i in range(total_frames // frames_per_batch):
        rollout = []
        for t in range(frames_per_batch // batch_size): # Each env step returns batch_size samples
            with torch.no_grad():
                action_td = actor(td)
                td.update(action_td)
                td = env.step(td)
            rollout.append(td.clone())

        # Stack rollout into one TensorDict with shape [T, B, ...]
        data = torch.stack(rollout, dim=0)  # shape: [T, B, ...]

        # Optionally reshape to [T*B, ...] if needed by loss functions
        data = data.reshape(-1, *data.shape[2:])  # shape: [T*B, ...]

        print("Data:\n", data)
        print("Data x[0]:\n", data["x"][0])
        print("Data edge_index[0]:\n", data["edge_index"][0])
        
        # Compute advantage and loss
        advantage_module(data)
        loss_vals = loss_module(data)
        
        # Optimize
        loss = loss_vals["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
        optim.step()
        optim.zero_grad()

        scheduler.step()




if __name__ == "__main__":
    
    train_PPO(Scenario())
