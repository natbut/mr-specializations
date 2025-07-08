#  Nathan Butler

import typing
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario

try:
    from agents.planning_agent import PlanningAgent
except:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join('agents', '..')))
    print("\n",sys.path)
    from agents.planning_agent import PlanningAgent

from vmas.simulator.utils import (ANGULAR_FRICTION, DRAG, LINEAR_FRICTION,
                                  Color, ScenarioUtils)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        ################
        # Scenario configuration
        ################
        self.plot_grid = False  # You can use this to plot a grid under the rendering for visualization purposes

        self.max_n_agents_holonomic = kwargs.pop(
            "max_n_agents_holonomic", 0
        )  # Number of agents with holonomic dynamics
        self.max_n_agents_diff_drive = kwargs.pop(
            "max_n_agents_diff_drive", 3
        )  # Number of agents with differential drive dynamics
        self.max_n_agents_car = kwargs.pop(
            "max_n_agents_car", 0
        )  # Number of agents with car dynamics
        self.max_n_agents = (
            self.max_n_agents_holonomic + self.max_n_agents_diff_drive + self.max_n_agents_car
        )
        self.max_n_tasks = kwargs.pop("max_n_tasks", 10) # EQUAL TO MAX TASKS
        self.max_n_obstacles = kwargs.pop("max_n_obstacles", 7)

        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 1
        )  # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 1
        )  # Y-coordinate limit for entities spawning

        self.static_env = kwargs.pop(
            "static_env", False
        )

        self.comms_rendering_range = kwargs.pop(
            "comms_rendering_range", 0
        )  # Used for rendering communication lines between agents (just visual)

        self._agents_per_task = kwargs.pop(
            "agents_per_task", 1
            )

        self.shared_rew = kwargs.pop(
            "shared_rew", False
        )  # Whether the agents get a global or local reward for going to their goals
        
        self.task_comp_range = kwargs.pop(
            "task_comp_range", 0.03
        )
        self.tasks_respawn_rate = kwargs.pop(
            "tasks_respawn_rate", 0.001
        )
        self.complete_task_coeff = kwargs.pop(
            "task_reward", 0.1
        )
        self.time_penalty = kwargs.pop(
            "time_penalty", -0.11
        )

        self.agent_radius = kwargs.pop(
            "agent_radius", 0.025
            )
        self.task_radius = kwargs.pop(
            "task_radius", 0.025
            )
        
        self.discrete_resolution = kwargs.pop(
            "discrete_resolution", 0.2)
        
        self.num_feats = kwargs.pop(
            "num_feats", 5
            ) # TODO compute
        
        self.min_distance_between_entities = kwargs.pop(
            "min_distance_between_entities", 0.05
            ) # Minimum distance between entities at spawning time
        
        self.comms_rew_decay_drop = kwargs.pop(
            "comms_rew_decay_drop", None #10.0
        )
        self.comms_rew_decay_max = kwargs.pop(
            "comms_rew_decay_max", None #0.5
        )
        
        self.variable_team_size = kwargs.pop(
            "variable_team_size", False
        )

        self.rich_cell_features = kwargs.pop(
            "rich_cell_features", False
        )

        self.spawn_tasks_burst = kwargs.pop(
            "spawn_tasks_burst", False
        )
        
        self.min_collision_distance = (
            0.005  # Minimum distance between entities for collision trigger
        )
        
        self.agent_spawn_radius = 0.2
        self.agent_disable_prob = 0.5

        ScenarioUtils.check_kwargs_consumed(kwargs) # Warn is not all kwargs have been consumed

        ################
        # Make world
        ################
        world = World(
            batch_dim,  # Number of environments simulated
            device,  # Device for simulation
            substeps=1,  # Number of physical substeps (more yields more accurate but more expensive physics)
            collision_force=50,  # Paramneter to tune for collisions
            dt=0.1,  # Simulation timestep
            gravity=(0.0, 0.0),  # Customizable gravity
            drag=DRAG,  # Physics parameters
            linear_friction=LINEAR_FRICTION,  # Physics parameters
            angular_friction=ANGULAR_FRICTION,  # Physics parameters
            x_semidim=self.world_spawning_x,
            y_semidim=self.world_spawning_y
            # There are many more....
        )

        ################
        # Add agents
        ################
        known_colors = [
            Color.BLUE,
            Color.ORANGE,
            Color.GREEN,
            Color.PINK,
            Color.PURPLE,
            Color.YELLOW,
            Color.RED,
        ]  # Colors for first 7
        colors = torch.randn(
            (max(self.max_n_agents, self.max_n_tasks), 3), device=device
        )  # Other colors if we have more elements are random

        self.agents = []
        for i in range(self.max_n_agents):
            # color = self.agent_color
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )  # Get color for agent


            if i < self.max_n_agents_holonomic:
                agent = PlanningAgent(
                    name=f"holonomic_{i}",
                    collide=True,
                    color=color,
                    render_action=True,
                    max_speed=0.25,
                    shape=Sphere(radius=self.agent_radius),
                    u_range=[1, 1],  # Ranges for actions
                    u_multiplier=[3, 3],  # Action multipliers
                    dynamics=Holonomic(),  # If you go to its class you can see it has 2 actions: force_x, and force_y
                )
                
            elif i < self.max_n_agents_holonomic + self.max_n_agents_diff_drive:
                agent = PlanningAgent(
                    name=f"diff_drive_{i - self.max_n_agents_holonomic}",
                    collide=True,
                    color=color,
                    render_action=True,
                    max_speed=0.25,
                    shape=Sphere(radius=self.agent_radius),
                    u_range=[1, 1],  # Ranges for actions
                    u_multiplier=[0.5, 1],  # Action multipliers
                    dynamics=DiffDrive(
                        world
                    ),  # If you go to its class you can see it has 2 actions: forward velocity and angular velocity
                )
            else:
                max_steering_angle = torch.pi / 4
                width = self.agent_radius
                agent = PlanningAgent(
                    name=f"car_{i-self.max_n_agents_holonomic-self.max_n_agents_diff_drive}",
                    collide=True,
                    color=color,
                    render_action=True,
                    max_speed=0.25,
                    shape=Box(length=self.agent_radius * 2, width=width),
                    u_range=[1, max_steering_angle],
                    u_multiplier=[0.5, 1],
                    dynamics=KinematicBicycle(
                        world,
                        width=width,
                        l_f=self.agent_radius,  # Distance between the front axle and the center of gravity
                        l_r=self.agent_radius,  # Distance between the rear axle and the center of gravity
                        max_steering_angle=max_steering_angle,
                    ),  # If you go to its class you can see it has 2 actions: forward velocity and steering angle
                )
                
            agent.tasks_rew = torch.zeros(batch_dim, device=device)
            agent.null_action = torch.zeros((batch_dim, 2), device=device)

            world.add_agent(agent)  # Add the agent to the world
            self.agents.append(agent)

        ################
        # Add tasks
        ################
        self.tasks = []
        for i in range(self.max_n_tasks):
            # color = known_colors[i]
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            ) 
            task = Landmark(
                name=f"goal_{i}",
                collide=False,
                color=color,
                shape=Sphere(radius=self.task_radius),
            )
            world.add_landmark(task)
            self.tasks.append(task)

        ################
        # Add obstacles
        ################
        self.obstacles = (
            []
        )  # We will store obstacles here for easy access
        for i in range(self.max_n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                color=Color.BLACK,
                shape=Sphere(radius=self.agent_radius * 2 / 3),
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)

        ################
        # Add Mothership
        ################
        self.base = Landmark(name=f"base",
                             collide=True,
                             color=Color.GREEN,
                             shape=Sphere(radius=self.agent_radius*1.5),
                             )
        world.add_landmark(self.base)

        ################
        # Init Reward Tracking
        ################        
        self.completed_tasks = torch.zeros((batch_dim, self.max_n_tasks), device=device)
        self.stored_tasks = torch.full(self.completed_tasks.shape, True, device=device)
        self.shared_tasks_rew = torch.zeros(batch_dim, device=device)
        
        ################
        # Init High-Level Graph Details
        ################
        cell_centers = []
        for x_pos in torch.arange(-self.world_spawning_x+(self.discrete_resolution/2), self.world_spawning_x, self.discrete_resolution):
            for y_pos in torch.arange(-self.world_spawning_y+(self.discrete_resolution/2), self.world_spawning_y, self.discrete_resolution):
                cell_centers.append([x_pos, y_pos])
        cell_centers = torch.tensor(cell_centers, device=device)
        self.discrete_cell_centers = torch.stack([cell_centers for _ in range(batch_dim)])
        self.discrete_cell_features = torch.zeros(self.discrete_cell_centers.shape[:-1] + (self.num_feats,), device=device)
        self.discrete_cell_explored = torch.full(self.discrete_cell_centers.shape[:-1], False, device=device)
        
        self.explored_cell_ids = torch.zeros((batch_dim, 1), device=device)

        self.stored_explored_cell_centers = torch.full(self.discrete_cell_centers.shape, 2.0*world.x_semidim, device=device)

        ################
        # Init Heuristic Details
        ################
        self.candidate_frontiers = torch.zeros((self.discrete_cell_centers.shape[:-1] + (4,2,)), device=device)
        self.frontiers = [self.candidate_frontiers[b].clone() for b in range(self.candidate_frontiers.shape[0])]
        self.comms_pts = torch.zeros((batch_dim, self.max_n_agents), device=device) # TODO Should be derived from number of agents

        ## ELEMENT STORAGE ##
        self.storage_pos = torch.full((batch_dim, 2), 2, device=device, dtype=torch.float32)

        return world

    def reset_world_at(self, env_index: int = None):
        # TODO update to reset at specific env_index
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents
            + self.obstacles
            + [self.base],
            # + self.tasks,  # List of entities to spawn
            self.world,
            env_index,  # Pass the env_index so we only reset what needs resetting
            self.min_distance_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )

        # RESET AGENTS
        self.agents = []
        for i, agent in enumerate(self.world.agents):
            if i <= 1 or not self.variable_team_size: # Mandates that we always have min 2 agents
                agent.is_active = True
            else:
                agent.is_active = (torch.rand(1,).item() < self.agent_disable_prob)
            
            # If active, compute new position and clamp within world bounds
            if agent.is_active:
                self.agents.append(agent)
                offset = torch.randn_like(self.base.state.pos)  # Random direction
                offset = offset / torch.norm(offset, dim=-1, keepdim=True) * torch.rand_like(self.base.state.pos) * self.agent_spawn_radius
                new_pos = self.base.state.pos + offset
                new_pos[..., 0] = torch.clamp(new_pos[..., 0], -self.world.x_semidim, self.world.x_semidim)
                new_pos[..., 1] = torch.clamp(new_pos[..., 1], -self.world.y_semidim, self.world.y_semidim)
            # else, move to storage pos
            else:
                agent.trajs = -1*self.storage_pos[:]
                new_pos = -1*self.storage_pos[:]
                
            # print("Agent active:", agent.is_active)
            # print("New pos:", new_pos)
            agent.state.pos = new_pos
            
        # print("!! RESET ENV AGENTS:", self.agents, "shape:", len(self.agents))

        # RESET TASKS
        for task in self.tasks:
            # print("TASK POS SHAPE", task.state.pos[:].shape)
            # print("STORAGE SHAPE:", self.storage_pos.shape)
            task.state.pos[:] = self.storage_pos
        self.stored_tasks.fill_(True)

        # RESET EXPLORED REGIONS & REGION FEATURES
        self.discrete_cell_explored.fill_(False)
        self.discrete_cell_features.fill_(0)
        self.stored_explored_cell_centers.fill_(2.0*self.world.x_semidim)
        self.explored_cell_ids = [] #torch.zeros((self.world.batch_dim, 1), device=self.world.device)

        # RESET ADDITIONAL HEURISTIC OBS
        self.candidate_frontiers.fill_(0)
        self.frontiers = [self.candidate_frontiers[b].clone() for b in range(self.candidate_frontiers.shape[0])]
        self.comms_pts.fill_(0)
        self._compute_frontier_pts()
        
        # INIT EXPLORED CELLS
        self.agents_pos = torch.stack(
                [a.state.pos for a in self.world.agents], dim=1)
        self.tasks_pos = torch.stack(
            [t.state.pos for t in self.tasks], dim=1)
        self._update_exploration()
        
        # UPDATE FRONTIERS
        self.frontiers = self._get_frontier_pts()
        
        # UPDATE DISCRETE CELL FEATURES
        if self.rich_cell_features:
            self._update_cell_features_rich()
        else:
            self._update_cell_features_sparse() 

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-2]

        if is_first:
            # We can compute rewards when the first agent is called such that we do not have to recompute global components
            self.time_rew = torch.full(
                (self.world.batch_dim,),
                self.time_penalty,
                device=self.world.device,
            )

            # Check completed tasks based on passenger locations
            self.agents_pos = torch.stack(
                [a.state.pos for a in self.world.agents], dim=1
            )
            self.tasks_pos = torch.stack([t.state.pos for t in self.tasks], dim=1)
            self.agents_tasks_dists = torch.cdist(self.agents_pos, self.tasks_pos)
            self.agents_per_task = torch.sum(
                (self.agents_tasks_dists < self.task_comp_range).type(torch.int),
                dim=1,
            )
            self.completed_tasks = self.agents_per_task >= self._agents_per_task
            if self.completed_tasks.any():
                self.stored_tasks = torch.where(self.completed_tasks, self.completed_tasks, self.stored_tasks)

            # Allocate completion credit to passengers
            self.shared_tasks_rew[:] = 0
            for a in self.world.agents:
                if a.is_active:
                    self.shared_tasks_rew += self.agent_tasks_reward(a)
        
        # Process environment updates
        if is_last:
            # == TOGGLE NEWLY-EXPLORED REGIONS ==
            # print("Agents pos shape:", self.agents_pos.shape, " Cell centers shape:", self.discrete_cell_centers.shape)
            # agents_cell_dists = torch.min(torch.cdist(self.discrete_cell_centers, self.agents_pos), dim=-1).values # for each cell, dist to each agent
            # print("\nAgents cell dists:", agents_cell_dists, " Shape:", agents_cell_dists.shape)

            # == UPDATE CELL EXPLORATION STATUS == 
            self._update_exploration()

            # == SPAWN IN STORED TASKS (random) ==
            if not self.spawn_tasks_burst:
                self.spawn_tasks()

            # == UPDATE FRONTIERS ==
            self.frontiers = self._get_frontier_pts()
            
            # == UPDATE DISCRETE CELL FEATURES ==
            if self.rich_cell_features:
                self._update_cell_features_rich()
            else:
                self._update_cell_features_sparse()    

        tasks_reward = (
            self.shared_tasks_rew if self.shared_rew else agent.tasks_rew
        )  # Choose global or local reward based on configuration

        rews = tasks_reward + self.time_rew

        return rews.unsqueeze(-1) # [B,1]
    
    def _update_exploration(self):
        # Mark cells as explored if any agent is within the square bounds of the cell
        # For each cell, check if any agent is within half the cell width/height in both x and y
        cell_centers = self.discrete_cell_centers  # [B, N_cells, 2]
        agent_pos = self.agents_pos  # [B, N_agents, 2]
        # Expand dims for broadcasting: [B, N_cells, 1, 2] - [B, 1, N_agents, 2]
        diff = (cell_centers.unsqueeze(2) - agent_pos.unsqueeze(1)).abs()
        # Add a small epsilon to include boundary cases due to floating point precision
        epsilon = 1e-6 # NOTE: added in small epsilon to avoid precision issues for robot cell placement.
        in_cell = (diff[..., 0] <= self.discrete_resolution / 2 + epsilon) & (diff[..., 1] <= self.discrete_resolution / 2 + epsilon)
        in_cell = in_cell.any(dim = -1) # Collapse to [B, N_cells]
        # print("In cell:", in_cell, "shape:", in_cell.shape, "Discrete explore shape:", self.discrete_cell_explored.shape)
        # If any agent is in the cell, mark as explored
        self.discrete_cell_explored = torch.where(in_cell, in_cell, self.discrete_cell_explored)
        # print("EXPLORE STATUS:", self.discrete_cell_explored)
        B = self.world.batch_dim
        self.explored_cell_ids = [torch.nonzero(self.discrete_cell_explored[b], as_tuple=False).squeeze(1) for b in range(B)]
        # print("EXPLORED IDS", self.explored_cell_ids)
        # Efficiently update explored_cell_centers for all batches
        for b, ids in enumerate(self.explored_cell_ids):
            self.stored_explored_cell_centers[b, ids] = self.discrete_cell_centers[b, ids]
        

    def _update_cell_features_sparse(self):
        # For each cell, compute features:
        # 0: tasks per cell
        # 1: obstacles per cell
        # 2: agents per cell
        # 3: frontiers per cell (number of unexplored neighbors)
        # 4: explored (bool)

        # [B, N_cells, 2]
        cell_centers = self.discrete_cell_centers
        B, N_cells, _ = cell_centers.shape
        
        # Base per cell
        base_pos = torch.stack([self.base.state.pos], dim=1)
        base_in_cell = (
            (cell_centers.unsqueeze(2) - base_pos.unsqueeze(1)).abs() <= self.discrete_resolution / 2
        ).all(dim=-1)  # [B, N_cells, N_tasks]
        base_per_cell = base_in_cell.sum(dim=-1).float()  # [B, N_cells]

        # Tasks per cell
        tasks_pos = self.tasks_pos  # [B, N_tasks, 2]
        tasks_in_cell = (
            (cell_centers.unsqueeze(2) - tasks_pos.unsqueeze(1)).abs() <= self.discrete_resolution / 2
        ).all(dim=-1)  # [B, N_cells, N_tasks]
        tasks_per_cell = tasks_in_cell.sum(dim=-1).float()  # [B, N_cells]

        # print("Tasks per cell (0):", tasks_per_cell[0], "\n shape:", tasks_per_cell.shape)

        # Obstacles per cell
        obstacles_pos = torch.stack([o.state.pos for o in self.obstacles], dim=1)  # [B, N_obstacles, 2]
        obstacles_in_cell = (
            (cell_centers.unsqueeze(2) - obstacles_pos.unsqueeze(1)).abs() <= self.discrete_resolution / 2
        ).all(dim=-1)  # [B, N_cells, N_obstacles]
        obstacles_per_cell = obstacles_in_cell.sum(dim=-1).float()  # [B, N_cells]

        # print("Obs per cell (0):", obstacles_per_cell[0], "\n shape:", obstacles_per_cell.shape)

        # Agents per cell
        agents_pos = self.agents_pos  # [B, N_agents, 2]
        agents_in_cell = (
            (cell_centers.unsqueeze(2) - agents_pos.unsqueeze(1)).abs() <= self.discrete_resolution / 2
        ).all(dim=-1)  # [B, N_cells, N_agents]
        agents_per_cell = agents_in_cell.sum(dim=-1).float()  # [B, N_cells]

        # print("Agents per cell (0):", agents_per_cell[0], "\n shape:", agents_per_cell.shape)

        # Frontiers per cell: number of neighbors that are not explored
        # Use candidate_frontiers and discrete_cell_explored
        frontiers_per_cell = torch.zeros((B, N_cells), device=cell_centers.device)
        for b in range(B):
            # For each cell, check if each neighbor is explored
            neighbors = self.candidate_frontiers[b]  # [N_cells, 4, 2]
            # For each neighbor, check if its center is in the explored set
            explored_centers = self.discrete_cell_centers[b][self.discrete_cell_explored[b]]  # [N_explored, 2]
            if explored_centers.shape[0] == 0:
                unexplored_neighbors = torch.ones((N_cells, 4), device=cell_centers.device)
            else:
                # [N_cells, 4, 2] - [N_explored, 2] -> [N_cells, 4, N_explored, 2]
                diff = neighbors.unsqueeze(2) - explored_centers.unsqueeze(0).unsqueeze(0)
                is_close = torch.isclose(diff, torch.zeros_like(diff), atol=1e-6)
                is_explored = torch.all(is_close, dim=-1).any(dim=-1)  # [N_cells, 4]
                unexplored_neighbors = (~is_explored).float()
            frontiers_per_cell[b] = unexplored_neighbors.sum(dim=-1)

        # print("Frontiers per cell (0):", frontiers_per_cell[0], "\n shape:", frontiers_per_cell.shape)

        # Explored (bool)
        explored = self.discrete_cell_explored.float()  # [B, N_cells]

        # print("Explored (0):", explored[0], "\n shape:", explored.shape)

        # Stack features into last dimension
        features = [
                tasks_per_cell,
                obstacles_per_cell,
                agents_per_cell,
                frontiers_per_cell,
                explored,
            ]
        if self.comms_rew_decay_drop != None:
            features.append(base_per_cell)
        self.discrete_cell_features = torch.stack(
            features,
            dim=-1,
        )  # [B, N_cells, 5]
        
        # print("Discrete features:", self.discrete_cell_features)

    def _update_cell_features_rich(self):
        # For each cell, compute features:
        # 0: 1/Dist to base (or 1 if base in cell)
        # 1: 1/Dist to nearest task (or 1 if task(s) are in cell)
        # 2: 1/Dist to nearest obstacle (or 1 if obstacle(s) are in cell)
        # 3: 1/Dist to nearest agent (or 1 if agent(s) in cell)
        # 4: % frontiers per cell (0.0 to 1.0)
        # 5: explored (bool)

        # [B, N_cells, 2]
        cell_centers = self.discrete_cell_centers
        B, N_cells, _ = cell_centers.shape
        max_dist = 2.0 * self.world.x_semidim # Max dist for norm
        
        # Base per cell
        base_pos = torch.stack([self.base.state.pos], dim=1)  # [B, 1, 2]
        # Compute Euclidean distance from each cell center to base position
        dists = torch.norm(cell_centers - base_pos.squeeze(1).unsqueeze(1), dim=-1)  # [B, N_cells]
        # Normalize so that 0 distance -> 1, distance >= 1 -> 0 (clip to [0,1])
        base_dist_per_cell = 1.0 - (torch.clamp(dists, 0, max_dist) / max_dist)
        # If base is in cell, set the value to 1
        base_in_cell = (
            (cell_centers.unsqueeze(2) - base_pos.unsqueeze(1)).abs() <= self.discrete_resolution / 2
        ).all(dim=-1).squeeze(-1)  # [B, N_cells]
        base_dist_per_cell = torch.where(base_in_cell, torch.ones_like(base_dist_per_cell), base_dist_per_cell)

        # print("Base dist per cell (0):", base_dist_per_cell[0], "\n shape:", base_dist_per_cell.shape)


        # Tasks per cell (rich feature: 1-dist to nearest active task, or 1.0 if task is in cell)
        tasks_pos = self.tasks_pos  # [B, N_tasks, 2]
        active_tasks_mask = ~self.stored_tasks  # [B, N_tasks]

        # If there are no tasks or no active tasks, set to 0
        if tasks_pos.shape[1] == 0 or not active_tasks_mask.any():
            task_dist_per_cell = torch.zeros((B, N_cells), device=cell_centers.device)
        else:
            # Mask out stored tasks by setting their positions far away
            masked_tasks_pos = tasks_pos.clone()
            far_away = 10.0
            for b in range(B):
                # Set stored (inactive) tasks to far away so they don't affect min distance
                masked_tasks_pos[b][~active_tasks_mask[b]] = far_away

                # Compute distances to all (masked) tasks
                dists = torch.norm(cell_centers.unsqueeze(2) - masked_tasks_pos.unsqueeze(1), dim=-1)  # [B, N_cells, N_tasks]
                min_dists, _ = dists.min(dim=-1)  # [B, N_cells]

                # Check if any active task is in the cell (distance <= half cell width)
                tasks_in_cell = (
                (cell_centers.unsqueeze(2) - masked_tasks_pos.unsqueeze(1)).abs() <= self.discrete_resolution / 2
                ).all(dim=-1)  # [B, N_cells, N_tasks]
                any_task_in_cell = tasks_in_cell.any(dim=-1)  # [B, N_cells]

                # Assign 1.0 if a task is in the cell, else 1.0 - min_dist (clipped to [0,1])
                task_dist_per_cell = torch.where(
                any_task_in_cell,
                torch.ones_like(min_dists),
                torch.clamp(1.0 - (min_dists/max_dist), min=0.0, max=max_dist)
                )

        # print("Task dist per cell (0):", task_dist_per_cell[0], "\n shape:", task_dist_per_cell.shape)

        # Obstacles per cell
        obstacles_pos = torch.stack([o.state.pos for o in self.obstacles], dim=1)  # [B, N_obstacles, 2]
        dists = torch.norm(cell_centers.unsqueeze(2) - obstacles_pos.unsqueeze(1), dim=-1)
        min_dists, _ = dists.min(dim=-1)  # [B, N_cells]
        obstacles_in_cell = (
            (cell_centers.unsqueeze(2) - obstacles_pos.unsqueeze(1)).abs() <= self.discrete_resolution / 2
        ).all(dim=-1)  # [B, N_cells, N_obstacles]
        any_obstacle_in_cell = obstacles_in_cell.any(dim=-1)  # [B, N_cells]
        obstacles_dist_per_cell = torch.where(
            any_obstacle_in_cell,
            torch.ones_like(min_dists),
            torch.clamp(1.0 - (min_dists/max_dist), min=0.0, max=max_dist)
        )  # [B, N_cells]

        # print("Obs  dist per cell (0):", obstacles_dist_per_cell[0], "\n shape:", obstacles_dist_per_cell.shape)

        # Agents per cell
        agents_pos = self.agents_pos  # [B, N_agents, 2]
        dists = torch.norm(cell_centers.unsqueeze(2) - agents_pos.unsqueeze(1), dim=-1)
        min_dists, _ = dists.min(dim=-1)  # [B, N_cells]
        agents_in_cell = (
            (cell_centers.unsqueeze(2) - agents_pos.unsqueeze(1)).abs() <= self.discrete_resolution / 2
        ).all(dim=-1)  # [B, N_cells, N_obstacles]
        any_agent_in_cell = agents_in_cell.any(dim=-1)  # [B, N_cells]
        agents_dist_per_cell = torch.where(
            any_agent_in_cell,
            torch.ones_like(min_dists),
            torch.clamp(1.0 - (min_dists/max_dist), min=0.0, max=max_dist)
        )  # [B, N_cells]

        # print("Agents dist per cell (0):", agents_dist_per_cell[0], "\n shape:", agents_dist_per_cell.shape)

        # Frontiers per cell: number of neighbors that are not explored
        # Use candidate_frontiers and discrete_cell_explored
        # TODO
        frontiers_per_cell = torch.zeros((B, N_cells), device=cell_centers.device)
        for b in range(B):
            # For each cell, check if each neighbor is explored
            neighbors = self.candidate_frontiers[b]  # [N_cells, 4, 2]
            # For each neighbor, check if its center is in the explored set
            explored_centers = self.discrete_cell_centers[b][self.discrete_cell_explored[b]]  # [N_explored, 2]
            if explored_centers.shape[0] == 0:
                unexplored_neighbors = torch.ones((N_cells, 4), device=cell_centers.device)
            else:
                # [N_cells, 4, 2] - [N_explored, 2] -> [N_cells, 4, N_explored, 2]
                diff = neighbors.unsqueeze(2) - explored_centers.unsqueeze(0).unsqueeze(0)
                is_close = torch.isclose(diff, torch.zeros_like(diff), atol=1e-6)
                is_explored = torch.all(is_close, dim=-1).any(dim=-1)  # [N_cells, 4]
                unexplored_neighbors = (~is_explored).float()
            frontiers_per_cell[b] = unexplored_neighbors.sum(dim=-1) / 4.0 # Normalize by max possible frontiers (4 neighbors)

        # print("Frontiers per cell (0):", frontiers_per_cell[0], "\n shape:", frontiers_per_cell.shape)

        # Explored (bool)
        explored = self.discrete_cell_explored.float()  # [B, N_cells]

        # print("Explored (0):", explored[0], "\n shape:", explored.shape)

        # Stack features into last dimension
        features = [
                task_dist_per_cell,
                obstacles_dist_per_cell,
                agents_dist_per_cell,
                frontiers_per_cell,
                explored,
            ]
        # Add base distance feature if base is relevant to scenario config
        if self.comms_rew_decay_drop != None:
            features.append(base_dist_per_cell)
        self.discrete_cell_features = torch.stack(
            features,
            dim=-1,
        )  # [B, N_cells, 5]
        
        # print("Discrete features:", self.discrete_cell_features)


    def _compute_frontier_pts(self):
        # For each cell, store the centerpoints of its 4 neighbors (left, right, down, up)
        # If a neighbor does not exist (out of bounds), fill with NaN
        B, N_cells, _ = self.discrete_cell_centers.shape
        device = self.discrete_cell_centers.device
        neighbor_offsets = torch.tensor([
            [-self.discrete_resolution, 0],  # left
            [self.discrete_resolution, 0],   # right
            [0, -self.discrete_resolution],  # down
            [0, self.discrete_resolution],   # up
        ], device=device)

        # Build a mapping from cell position to index for each batch
        for b in range(B):
            centers = self.discrete_cell_centers[b]  # [N_cells, 2]
            for idx, pos in enumerate(centers):
                for n, offset in enumerate(neighbor_offsets):
                    neighbor_pos = pos + offset
                    # Check if neighbor is within world bounds
                    if (
                        -self.world_spawning_x <= neighbor_pos[0] <= self.world_spawning_x
                        and -self.world_spawning_y <= neighbor_pos[1] <= self.world_spawning_y
                    ):
                        self.candidate_frontiers[b, idx, n] = neighbor_pos
                    else:
                        # print(f"neighbor {neighbor_pos} out of bounds")
                        self.candidate_frontiers[b, idx, n] = torch.tensor([100.0, 100.0], device=device)


    def _get_frontier_pts(self):
        # Apply the exploration mask across all worlds at once
        # self.frontiers: [B, N_cells, 4, 2], self.discrete_cell_explored: [B, N_cells]
        # This will give a list of [N_explored_cells, 4, 2] for each batch/world
        # Vectorized masked frontiers computation for efficiency
        B, N_cells, N_neighbors, _ = self.candidate_frontiers.shape
        device = self.candidate_frontiers.device

        # Get explored mask and explored centers for all batches
        explored_centers = [
            self.discrete_cell_centers[b][self.discrete_cell_explored[b]] for b in range(B)
        ]  # List of [N_explored, 2] per batch

        # Gather frontiers for explored cells in each batch
        masked_frontiers = [
            self.candidate_frontiers[b][self.discrete_cell_explored[b]].clone() for b in range(B)
        ]  # List of [N_explored, 4, 2] per batch

        # For each batch, mask out neighbors that are already explored (set to large value)
        for b in range(B):
            if explored_centers[b].shape[0] == 0:
                continue
            # [N_explored, 4, 2] -> [N_explored*4, 2]
            neighbors = masked_frontiers[b].reshape(-1, 2)
            # [N_explored*4, N_explored, 2]
            diff = neighbors.unsqueeze(1) - explored_centers[b].unsqueeze(0)
            is_close = torch.isclose(diff, torch.zeros_like(diff), atol=1e-6)
            is_explored = torch.all(is_close, dim=2).any(dim=1)  # [N_explored*4]
            # Set already-explored neighbors to large value
            neighbors[is_explored] = torch.tensor([100.0, 100.0], device=device)
            masked_frontiers[b] = neighbors.reshape(-1, N_neighbors, 2)

        # Pad masked_frontiers so all entries have the same length for torch.stack
        max_len = max(f.shape[0] for f in masked_frontiers)
        for b in range(B):
            n = masked_frontiers[b].shape[0]
            if n < max_len:
                pad_shape = (max_len - n, N_neighbors, 2)
                pad = torch.full(pad_shape, 100.0, device=device)
                masked_frontiers[b] = torch.cat([masked_frontiers[b], pad], dim=0)

        return masked_frontiers

    
    def agent_tasks_reward(self, agent: Agent):
        """Reward for covering targets"""
        agent_index = self.world.agents.index(agent)

        agent.tasks_rew[:] = 0
        targets_covered_by_agent = (
            self.agents_tasks_dists[:, agent_index] < self.task_comp_range
        )
        num_covered_targets_covered_by_agent = (
            targets_covered_by_agent * self.completed_tasks
        ).sum(dim=-1)
        agent.tasks_rew += (
            num_covered_targets_covered_by_agent * self.complete_task_coeff
        )
        
        if self.comms_rew_decay_drop is not None and agent.tasks_rew.any() != 0.0:
            decay = self._get_reward_decay(agent)
            agent.tasks_rew *= decay
            # print("Decayed rew:", agent.tasks_rew)
        return agent.tasks_rew
            
    def _get_reward_decay(self, agent: Agent):
        # Decay reward value using multi-hop comms distance
        max_min_dists = torch.zeros((self.world.batch_dim,1), dtype=agent.state.pos.dtype, device=agent.state.pos.device)
        cur_pos = agent.state.pos.clone().unsqueeze(1)
        visited_pos = agent.state.pos.clone().unsqueeze(1)
        base_pos = self.base.state.pos.unsqueeze(1)
        other_agents_pos = torch.stack(
            [a.state.pos for a in self.world.agents if a.name != agent.name],
            dim=1
        )
        other_agents_pos = torch.cat((other_agents_pos, base_pos), dim=1)
        num_pts = other_agents_pos.shape[1]
        
        while not torch.allclose(cur_pos, base_pos):
            # Get distance to nearest agents
            dists = torch.norm(cur_pos - other_agents_pos, dim=2)
            nearest_pos_idx = torch.argmin(dists, dim=1)
            min_dists = torch.min(dists, dim=1).values.unsqueeze(-1)
            
            # Track already-visited positions & update current position
            visited_pos = torch.cat((visited_pos, cur_pos), dim=1)
            cur_pos = other_agents_pos[torch.arange(other_agents_pos.shape[0]), nearest_pos_idx].unsqueeze(1)
            # print(f"Updated visited pos: {visited_pos}\n New cur pos: {cur_pos}")
            # print(f"New max min dists: {max_min_dists}")

            # Remove the current position from agents_pos for the next iteration, but do not remove entries that are at base_pos
            mask = ~torch.all(torch.isclose(other_agents_pos, cur_pos), dim=-1)
            # Filter agents_pos batch-wise to avoid shape errors
            filtered_agents_pos = []
            for b in range(other_agents_pos.shape[0]):
                filtered = other_agents_pos[b][mask[b]]
                while filtered.shape[0] < num_pts:
                    filtered = torch.cat((filtered, base_pos[b]), dim=0)
                filtered_agents_pos.append(filtered)
            other_agents_pos = torch.stack(filtered_agents_pos, dim=0)
            # print(f"New agents pos: {agents_pos}\nbase_pos: {base_pos}")
            
            max_min_dists = torch.where(min_dists > max_min_dists, min_dists, max_min_dists)
        
        # Compute decay
        # print("Tasks rew shape", agent.tasks_rew.shape)
        # print("comp1:", self.comms_rew_decay_drop*max_min_dists.squeeze())
        # print("comp2:", self.comms_rew_decay_drop*torch.full(agent.tasks_rew.shape, self.comms_rew_decay_max, device=agent.device))
        exponent = self.comms_rew_decay_drop*max_min_dists.squeeze() - self.comms_rew_decay_drop*torch.full(agent.tasks_rew.shape, self.comms_rew_decay_max, device=agent.device)
        # print("Exponent:", exponent)
        decay = torch.exp(exponent)
        # print("Decay:", decay)
        offset = torch.ones(agent.tasks_rew.shape, device=agent.device) - decay
        # print("With offset,", offset)
            
        return torch.clip(offset, min=0.0)
        

    def observation(self, agent: Agent):
        # Global observation for learner and local agent observation for planner
        local_obs = {
            "obs_tasks": torch.stack(
                [task.state.pos for task in self.tasks],
                dim=1
            ),
            "obs_agents": torch.stack(
                [a.state.pos for a in self.world.agents if a.name != agent.name and a.is_active],
                dim=1
            ),
            "obs_obstacles": torch.stack(
                [obstacle.state.pos for obstacle in self.obstacles],
                dim=1
            ),
            "obs_frontiers": torch.stack(self.frontiers, dim=0),
            "obs_base": self.base.state.pos,
            "pos": agent.state.pos,
            "vel": agent.state.vel,
        }
        if not isinstance(agent.dynamics, Holonomic):
            # Non-holonomic agents need to know angular states
            local_obs.update(
                {
                    "rot": agent.state.rot,
                    "ang_vel": agent.state.ang_vel,
                }
            )

        agent.obs = local_obs # Agent obs for local planning
        # print(local_obs)
        
        cell_features = torch.nested.as_nested_tensor([self.discrete_cell_features[b, ids] for b, ids in enumerate(self.explored_cell_ids)], device=self.world.device)
        cell_centers = torch.nested.as_nested_tensor([self.discrete_cell_centers[b, ids] for b, ids in enumerate(self.explored_cell_ids)], device=self.world.device)
        
        cell_features = torch.nested.to_padded_tensor(cell_features, 
                                                      padding=0.0, 
                                                      output_size=(self.world.batch_dim, 100, self.num_feats))

        cell_centers = torch.nested.to_padded_tensor(cell_centers, 
                                                     padding=0.0, 
                                                     output_size=(self.world.batch_dim, 100, 2))
        
        rob_pos = torch.stack([agent.state.pos for agent in self.agents], dim=1)
        # print("Env rob pos shape:", rob_pos.shape)
        # Pad stacked_pos to [B, max_n_agents, 2]
        B, R, D = rob_pos.shape
        if R < self.max_n_agents:
            pad_shape = (B, self.max_n_agents - R, D)
            pad = torch.zeros(pad_shape, device=rob_pos.device, dtype=rob_pos.dtype)
            rob_pos = torch.cat([rob_pos, pad], dim=1)
        # print("Padded rob pos:", rob_pos)
        
        # print("\nCell features sample:", cell_features[0][:4])
        # torch.stack(
        #         [agent.state.pos for agent in self.agents],
        #         dim=1
        #     )
        
        global_obs = {
            "cell_feats": cell_features, # torch.nested.to_padded_tensor(cell_features, padding=0.0),

            "cell_pos": cell_centers, # torch.nested.to_padded_tensor(cell_centers, padding=0.0),
                #self.stored_explored_cell_centers,

            "num_cells": torch.tensor([len(ids) for ids in self.explored_cell_ids], dtype=cell_features.dtype, device=self.world.device),

            "rob_pos": rob_pos,
            
            "num_robs": torch.tensor([len(self.agents) for _ in range(self.world.batch_dim)], dtype=torch.float32, device=self.world.device)
        }

        return global_obs
    
    def spawn_tasks(self, attempts=1, verbose=False):
        explored_cell_centers = [self.discrete_cell_centers[b, ids] for b, ids in enumerate(self.explored_cell_ids)]
        occupied_positions_agents = [self.agents_pos]

        for _ in range(attempts):
            for i, task in enumerate(self.tasks):
                
                occupied_positions_tasks = [
                                            o.state.pos.unsqueeze(1)
                                            for o in self.tasks
                                            if o is not task
                                            ]
                occupied_positions = torch.cat(
                    occupied_positions_agents + occupied_positions_tasks,
                    dim=1,
                )

                # == MOVE COMPLETED TASKS OUT OF BOUNDS (TO STORAGE) ==
                if self.completed_tasks[:, i].any():
                    task.state.pos[self.completed_tasks[:, i]] = self.storage_pos[0].clone()
                
                # == SPAWN IN TASKS TO EXPLORED REGIONS (OCCASIONALLY) ==
                    # 1) Grab random explored cell. 2) Use cell dims for x_bounds and y_bounds
                for idx in range(self.world.batch_dim):
                    spawn_prob = self.tasks_respawn_rate * len(explored_cell_centers[idx])
                    if verbose: print("Spawn prob:", spawn_prob)
                    if self.stored_tasks[idx,i].any() and np.random.random() < spawn_prob:
                        if verbose: print("Spawning task", i, " in world", idx)
                        spawn_pos = []
                        rand_cell_idx = torch.randint(explored_cell_centers[idx].shape[0], (1,)).item()
                        rand_cell_center = explored_cell_centers[idx][rand_cell_idx].tolist()
                        if verbose: print("\nRand cell center:", rand_cell_center)

                        rand_pos = ScenarioUtils.find_random_pos_for_entity(
                            occupied_positions[idx],
                            env_index=idx,
                            world=self.world,
                            min_dist_between_entities=self.min_distance_between_entities,
                            x_bounds=(rand_cell_center[0]-self.discrete_resolution/2,
                                    rand_cell_center[0]+self.discrete_resolution/2),
                            y_bounds=(rand_cell_center[1]-self.discrete_resolution/2,
                                    rand_cell_center[1]+self.discrete_resolution/2),
                        )
                        spawn_pos.append(rand_pos)

                        spawn_pos = torch.stack(spawn_pos)
                        if verbose: print("Spawn pos:", spawn_pos)
                        task.state.pos[idx] = spawn_pos
                        self.stored_tasks[idx, i] = False


    def done(self) -> Tensor:
        # print("Completed tasks:", self.completed_tasks)
        # return self.completed_tasks.all(dim=-1) #self.all_goal_reached
        return torch.full(
            (self.world.batch_dim,), False, device=self.world.device)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            # "pos_rew": self.tasks_rew if self.shared_rew else agent.pos_rew,
            # "final_rew": self.final_rew,
            # "agent_collision_rew": agent.agent_collision_rew,
            "completed_tasks": self.completed_tasks,
            "agent_tasks": agent.tasks_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = [
            ScenarioUtils.plot_entity_rotation(agent, env_index)
            for agent in self.world.agents
            if not isinstance(agent.dynamics, Holonomic)
        ]  # Plot the rotation for non-holonomic agents

        # Plot explored regions
        side_len = self.discrete_resolution/2
        square = [[-side_len, -side_len],
                  [-side_len, side_len],
                  [side_len, side_len],
                  [side_len, -side_len]]
        
        for center in self.stored_explored_cell_centers[env_index]:
            cell = rendering.make_polygon(square, filled=True)
            xform = rendering.Transform()
            xform.set_translation(*center)
            cell.add_attr(xform)
            cell.set_color(0.95,0.95,0.95)
            geoms.append(cell)

        # Plot Task ranges
        for target in self.tasks:
            range_circle = rendering.make_circle(self.task_comp_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(0,0,0)
            geoms.append(range_circle)

        # Plot agents traj
        if len(self.world.agents[0].trajs) > 0:
            for agent in self.world.agents:
                if not agent.is_active:
                    continue
                traj_pts = agent.trajs[env_index][agent.traj_idx[env_index]:]
                if len(traj_pts) > 0:
                    for pt in traj_pts:
                        pt_circle = rendering.make_circle(0.01, filled=True)
                        xform = rendering.Transform()
                        xform.set_translation(*pt)
                        pt_circle.add_attr(xform)
                        pt_circle.set_color(*agent.color)
                        geoms.append(pt_circle)

        # Plot communication lines
        if self.comms_rendering_range > 0:
            for i, agent1 in enumerate(self.world.agents):
                for j, agent2 in enumerate(self.world.agents):
                    if j <= i:
                        continue
                    agent_dist = torch.linalg.vector_norm(
                        agent1.state.pos - agent2.state.pos, dim=-1
                    )
                    if agent_dist[env_index] <= self.comms_rew_decay_max:
                        color = Color.BLACK.value
                        line = rendering.Line(
                            (agent1.state.pos[env_index]),
                            (agent2.state.pos[env_index]),
                            width=1,
                        )
                        line.set_color(*color)
                        geoms.append(line)
        return geoms
    
    

if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(scenario, display_info=False)