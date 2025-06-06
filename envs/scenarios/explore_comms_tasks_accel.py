# Nathan Butler

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
    from agents.planning_agent_accel import PlanningAgent
except:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join('agents', '..')))
    print("\n",sys.path)
    from agents.planning_agent_accel import PlanningAgent

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
            "num_feats", 5) # TODO compute dynamically?
        
        self.min_distance_between_entities = kwargs.pop(
            "min_distance_between_entities", 0.05) # Minimum distance between entities at spawning time
        self.min_collision_distance = (
            0.005   # Minimum distance between entities for collision trigger
        )

        ScenarioUtils.check_kwargs_consumed(kwargs) # Warn is not all kwargs have been consumed

        ################
        # Make world
        ################
        world = World(
            batch_dim,  # Number of environments simulated
            device,  # Device for simulation
            substeps=1,  # Number of physical substeps (more yields more accurate but more expensive physics)
            collision_force=500,  # Paramneter to tune for collisions
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
        ]   # Colors for first 7
        colors = torch.randn(
            (max(self.max_n_agents, self.max_n_tasks), 3), device=device
        )   # Other colors if we have more elements are random

        self.agents = []
        for i in range(self.max_n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )   # Get color for agent


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

            world.add_agent(agent)  # Add the agent to the world
            self.agents.append(agent)

        ################
        # Add tasks
        ################
        self.tasks = []
        for i in range(self.max_n_tasks):
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
        )   # We will store obstacles here for easy access
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
        # Vectorized generation of cell_centers
        x_coords = torch.arange(
            -self.world_spawning_x + (self.discrete_resolution / 2),
            self.world_spawning_x,
            self.discrete_resolution,
            device=device
        )
        y_coords = torch.arange(
            -self.world_spawning_y + (self.discrete_resolution / 2),
            self.world_spawning_y,
            self.discrete_resolution,
            device=device
        )
        # Create a meshgrid of all combinations of x and y coordinates
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Stack x and y coordinates to form [N_x * N_y, 2]
        cell_centers_2d = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        
        # Stack for batch dimension: [batch_dim, N_x * N_y, 2]
        self.discrete_cell_centers = cell_centers_2d.unsqueeze(0).expand(batch_dim, -1, -1)

        self.discrete_cell_features = torch.zeros(self.discrete_cell_centers.shape[:-1] + (self.num_feats,), device=device)
        self.discrete_cell_explored = torch.full(self.discrete_cell_centers.shape[:-1], False, device=device)

        self.stored_explored_cell_centers = torch.full(self.discrete_cell_centers.shape, 2.0*world.x_semidim, device=device)

        ################
        # Init Heuristic Details
        ################
        self.candidate_frontiers = torch.zeros((self.discrete_cell_centers.shape[:-1] + (4,2,)), device=device)
        self.frontiers = [self.candidate_frontiers[b].clone() for b in range(self.candidate_frontiers.shape[0])] # Initial placeholder
        self.comms_pts = torch.zeros((batch_dim, self.max_n_agents), device=device) # TODO Should be derived from number of agents

        ## ELEMENT STORAGE ##
        self.storage_pos = torch.full((batch_dim, 2), 2, device=device, dtype=torch.float32)
        
        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents
            + self.obstacles
            + [self.base],
            self.world,
            env_index,  # Pass the env_index so we only reset what needs resetting
            self.min_distance_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )

        # RESET AGENTS - Vectorized
        spawn_radius = 0.2
        # Generate random offsets for all agents in all batches
        offsets = torch.randn(
            self.world.batch_dim, len(self.world.agents), 2,
            device=self.world.device,
            dtype=torch.float32
        )
        # Normalize direction and apply random magnitude within spawn_radius
        offsets = offsets / torch.norm(offsets, dim=-1, keepdim=True) * (
            torch.rand(self.world.batch_dim, len(self.world.agents), 1, device=self.world.device) * spawn_radius
        )
        
        # Add base position (expanded for broadcasting) to all offsets
        new_agent_positions = self.base.state.pos.unsqueeze(1) + offsets # [B, N_agents, 2]

        # Assign back to each agent's state
        for k, agent in enumerate(self.world.agents):
            agent.state.pos = new_agent_positions[:, k, :]

        # RESET TASKS
        for task in self.tasks:
            task.state.pos[:] = self.storage_pos # Assuming self.storage_pos can broadcast or is of appropriate shape
        self.stored_tasks.fill_(True)

        # RESET EXPLORED REGIONS & REGION FEATURES
        self.discrete_cell_explored.fill_(False)
        self.discrete_cell_features.fill_(0)
        self.stored_explored_cell_centers.fill_(2.0*self.world.x_semidim)

        # RESET ADDITIONAL HEURISTIC OBS
        self.candidate_frontiers.fill_(0) # This is filled by _compute_frontier_pts at each step, so ok to zero here
        # Re-compute candidate frontiers (neighbors) for all cells in a vectorized manner
        self._compute_frontier_pts() 
        self.frontiers = self._get_frontier_pts() # Get the initial frontiers after reset
        self.comms_pts.fill_(0)

    def reward(self, agent: Agent):
        # Using -1 here assuming self.world.agents only contains agents and not other entities
        # If there's a reason for -2 (e.g. some internal VMAS artifact at end), it should be specified.
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1] 

        if is_first:
            # We can compute rewards when the first agent is called such that we do not have to recompute global components
            self.time_rew = torch.full(
                (self.world.batch_dim,),
                self.time_penalty,
                device=self.world.device,
            )

            self.agents_pos = torch.stack(
                [a.state.pos for a in self.world.agents], dim=1
            ) # [B, N_agents, 2]
            self.tasks_pos = torch.stack([t.state.pos for t in self.tasks], dim=1) # [B, N_tasks, 2]
            self.agents_tasks_dists = torch.cdist(self.agents_pos, self.tasks_pos) # [B, N_agents, N_tasks]
            
            # For each task, check if enough agents are within range to complete it
            agents_near_tasks = (self.agents_tasks_dists < self.task_comp_range).sum(dim=1) # [B, N_tasks]
            
            # Determine newly completed tasks
            newly_completed_tasks = (agents_near_tasks >= self._agents_per_task) # [B, N_tasks]
            
            # Update `self.stored_tasks`: once a task is completed, it stays completed for reward purposes
            self.stored_tasks = torch.where(newly_completed_tasks, newly_completed_tasks, self.stored_tasks)

            # Compute rewards for all agents and shared reward in one vectorized operation
            # `(self.agents_tasks_dists < self.task_comp_range)`: [B, N_agents, N_tasks] - Which agent is near which task (boolean)
            # `self.stored_tasks.unsqueeze(1)`: [B, 1, N_tasks] - Which tasks are globally completed (boolean)
            # Element-wise AND gives [B, N_agents, N_tasks] where True means agent is near AND task is completed
            completed_tasks_covered_by_agents = (self.agents_tasks_dists < self.task_comp_range) & self.stored_tasks.unsqueeze(1)
            
            # Sum over tasks dimension to get total completed tasks covered by each agent
            num_completed_tasks_per_agent = completed_tasks_covered_by_agents.sum(dim=-1).float() # [B, N_agents]
            
            # Assign local tasks_rew to each agent
            for k, ag in enumerate(self.world.agents):
                ag.tasks_rew = num_completed_tasks_per_agent[:, k] * self.complete_task_coeff # [B]
            
            # Calculate shared reward (sum of completed tasks across all agents, scaled)
            self.shared_tasks_rew = num_completed_tasks_per_agent.sum(dim=-1) * self.complete_task_coeff # [B]
        
        # Process environment updates
        if is_last:
            # == TOGGLE NEWLY-EXPLORED REGIONS ==
            cell_centers = self.discrete_cell_centers  # [B, N_cells, 2]
            agent_pos = self.agents_pos  # [B, N_agents, 2]
            
            # Calculate absolute difference between cell centers and agent positions
            # [B, N_cells, 1, 2] - [B, 1, N_agents, 2] -> [B, N_cells, N_agents, 2]
            diff = (cell_centers.unsqueeze(2) - agent_pos.unsqueeze(1)).abs()
            
            # Check if both x and y diffs are within half_res
            in_cell = (diff[..., 0] <= self.discrete_resolution / 2) & (diff[..., 1] <= self.discrete_resolution / 2)
            in_cell = in_cell.any(dim = -1) # Collapse over agents: [B, N_cells]
            
            # If any agent is in the cell, mark as explored (torch.where ensures it only sets to True)
            self.discrete_cell_explored = torch.where(in_cell, in_cell, self.discrete_cell_explored)
            
            # Efficiently update stored_explored_cell_centers for all batches
            # Mask self.discrete_cell_centers with self.discrete_cell_explored
            # Then, for the non-explored ones, assign the out-of-bounds value
            out_of_bounds_value_for_stored = torch.full((2,), 2.0 * self.world.x_semidim, device=self.world.device)
            self.stored_explored_cell_centers = torch.where(
                self.discrete_cell_explored.unsqueeze(-1).expand_as(self.discrete_cell_centers),
                self.discrete_cell_centers,
                out_of_bounds_value_for_stored
            )

            # == MOVE COMPLETED TASKS OUT OF BOUNDS (TO STORAGE) ==
            # `self.completed_tasks` is [B, N_tasks] boolean
            for i, task in enumerate(self.tasks):
                # task.state.pos is [B, 2]
                # self.storage_pos is [B, 2]
                task.state.pos[self.stored_tasks[:, i]] = self.storage_pos[self.stored_tasks[:, i]]

            # == SPAWN IN TASKS TO EXPLORED REGIONS (OCCASIONALLY) ==
            B = self.world.batch_dim
            # Calculate number of explored cells for each batch
            num_explored_cells = self.discrete_cell_explored.sum(dim=1) # [B]
            
            # Calculate spawning probability for each batch. Avoid division by zero.
            spawn_probs = torch.where(
                num_explored_cells > 0, 
                self.tasks_respawn_rate * num_explored_cells.float(), 
                torch.tensor(0.0, device=self.world.device)
            ) # [B]
            
            # Determine which tasks should respawn for each batch
            random_chance = torch.rand(B, self.max_n_tasks, device=self.world.device) # [B, N_tasks]
            should_respawn_task = self.stored_tasks & (random_chance < spawn_probs.unsqueeze(1)) # [B, N_tasks]

            for i, task in enumerate(self.tasks):
                for b_idx in range(B):
                    if should_respawn_task[b_idx, i]:
                        # Rebuild occupied_positions for the current batch and task
                        current_agents_pos_b = self.agents_pos[b_idx].unsqueeze(0) # [1, N_agents, 2]

                        # Collect positions of other *active* tasks for this specific batch, excluding current task `i`
                        # and any tasks that have already been moved to storage in this step
                        active_other_tasks_pos_b_filtered = []
                        for j in range(self.max_n_tasks):
                            if j != i:
                                task_j_pos_b = self.tasks[j].state.pos[b_idx].unsqueeze(0) # [1, 2]
                                # Check if task_j_pos_b is NOT equal to storage_pos[b_idx]
                                if not torch.isclose(task_j_pos_b, self.storage_pos[b_idx], atol=1e-6).all():
                                    active_other_tasks_pos_b_filtered.append(task_j_pos_b.unsqueeze(1)) # [1,1,2]
                        
                        if active_other_tasks_pos_b_filtered:
                            occupied_positions_for_b_idx = torch.cat(
                                [current_agents_pos_b] + active_other_tasks_pos_b_filtered,
                                dim=1,
                            ) # [1, NumOccupied, 2]
                        else:
                            occupied_positions_for_b_idx = current_agents_pos_b

                        # Select a random explored cell for spawning for this batch
                        explored_cell_indices_b = torch.nonzero(self.discrete_cell_explored[b_idx], as_tuple=False).squeeze(1)
                        if explored_cell_indices_b.numel() == 0:
                            # No explored cells in this batch, cannot spawn this task.
                            continue

                        # Randomly pick one of the *valid* explored cell indices
                        rand_cell_idx_in_explored_indices = torch.randint(explored_cell_indices_b.shape[0], (1,)).item()
                        actual_cell_idx = explored_cell_indices_b[rand_cell_idx_in_explored_indices]
                        rand_cell_center = self.discrete_cell_centers[b_idx, actual_cell_idx].tolist()

                        # Find a random position within the selected explored cell
                        spawn_pos_for_task_b = ScenarioUtils.find_random_pos_for_entity(
                            occupied_positions_for_b_idx[0], # Pass the single batch item
                            env_index=b_idx,
                            world=self.world,
                            min_dist_between_entities=self.min_distance_between_entities,
                            x_bounds=(rand_cell_center[0]-self.discrete_resolution/2,
                                      rand_cell_center[0]+self.discrete_resolution/2),
                            y_bounds=(rand_cell_center[1]-self.discrete_resolution/2,
                                      rand_cell_center[1]+self.discrete_resolution/2),
                        )
                        task.state.pos[b_idx] = spawn_pos_for_task_b
                        self.stored_tasks[b_idx, i] = False # Mark as not stored anymore (i.e., active)


            # == UPDATE FRONTIERS & COMMS ==
            self.frontiers = self._get_frontier_pts()
            
            # == UPDATE DISCRETE CELL FEATURES ==
            self._updated_discrete_cell_features()    

        tasks_reward = (
            self.shared_tasks_rew if self.shared_rew else agent.tasks_rew
        )   # Choose global or local reward based on configuration

        rews = tasks_reward + self.time_rew

        return rews.unsqueeze(-1) # [B,1]

    def _updated_discrete_cell_features(self):
        # For each cell, compute features:
        # 0: tasks per cell
        # 1: obstacles per cell
        # 2: agents per cell
        # 3: frontiers per cell (number of unexplored neighbors)
        # 4: explored (bool)

        # [B, N_cells, 2]
        cell_centers = self.discrete_cell_centers
        B, N_cells, _ = cell_centers.shape
        device = cell_centers.device

        # Tasks per cell (already vectorized)
        tasks_pos = self.tasks_pos  # [B, N_tasks, 2]
        # Expand dims for broadcasting: [B, N_cells, 1, 2] - [B, 1, N_tasks, 2]
        tasks_in_cell_diff = (cell_centers.unsqueeze(2) - tasks_pos.unsqueeze(1)).abs()
        tasks_in_cell = (tasks_in_cell_diff <= self.discrete_resolution / 2).all(dim=-1)  # [B, N_cells, N_tasks]
        tasks_per_cell = tasks_in_cell.sum(dim=-1).float()  # [B, N_cells]

        # Obstacles per cell (already vectorized)
        obstacles_pos = torch.stack([o.state.pos for o in self.obstacles], dim=1)  # [B, N_obstacles, 2]
        obstacles_in_cell_diff = (cell_centers.unsqueeze(2) - obstacles_pos.unsqueeze(1)).abs()
        obstacles_in_cell = (obstacles_in_cell_diff <= self.discrete_resolution / 2).all(dim=-1)  # [B, N_cells, N_obstacles]
        obstacles_per_cell = obstacles_in_cell.sum(dim=-1).float()  # [B, N_cells]

        # Agents per cell (already vectorized)
        agents_pos = self.agents_pos  # [B, N_agents, 2]
        agents_in_cell_diff = (cell_centers.unsqueeze(2) - agents_pos.unsqueeze(1)).abs()
        agents_in_cell = (agents_in_cell_diff <= self.discrete_resolution / 2).all(dim=-1)  # [B, N_cells, N_agents]
        agents_per_cell = agents_in_cell.sum(dim=-1).float()  # [B, N_cells]

        # Frontiers per cell: number of neighbors that are not explored (Vectorized)
        frontiers_per_cell = torch.zeros((B, N_cells), device=device)

        max_explored_cells = self.discrete_cell_explored.sum(dim=1).max().item() # Max number of explored cells in any batch
        
        if max_explored_cells == 0:
            # If no cells are explored in any batch, all existing neighbors are unexplored
            # A "neighbor" is considered valid if it's not the 100.0 out-of-bounds marker
            is_valid_neighbor = ~torch.isclose(self.candidate_frontiers, torch.tensor([100.0, 100.0], device=device), atol=1e-6).all(dim=-1)
            frontiers_per_cell = is_valid_neighbor.sum(dim=-1).float() # [B, N_cells]
        else:
            # Create a padded tensor for all explored cell centers to compare against
            padded_explored_centers = torch.full((B, max_explored_cells, 2), float('nan'), device=device)
            explored_mask_for_padded = torch.zeros((B, max_explored_cells), dtype=torch.bool, device=device)

            for b in range(B):
                explored_indices = torch.nonzero(self.discrete_cell_explored[b], as_tuple=False).squeeze(1)
                num_explored = explored_indices.shape[0]
                if num_explored > 0:
                    padded_explored_centers[b, :num_explored] = self.discrete_cell_centers[b, explored_indices]
                    explored_mask_for_padded[b, :num_explored] = True

            # candidate_frontiers: [B, N_cells, 4, 2] (current cell's neighbors)
            # padded_explored_centers: [B, max_explored_cells, 2] (all explored cells for each batch, padded)

            # Calculate difference between each candidate neighbor and each explored center
            # [B, N_cells, 4, 1, 2] - [B, 1, 1, max_explored_cells, 2]
            diff = self.candidate_frontiers.unsqueeze(3) - padded_explored_centers.unsqueeze(1).unsqueeze(1)
            # Resulting shape: [B, N_cells, 4, max_explored_cells, 2]

            # Check if differences are close to zero (meaning a neighbor's center matches an explored cell's center)
            is_close = torch.isclose(diff, torch.zeros_like(diff), atol=1e-6)
            # is_close shape: [B, N_cells, 4, max_explored_cells, 2]

            # A neighbor is explored if all its coordinates are close to a *valid* explored center's coordinates
            is_explored_per_neighbor_and_explored_center = torch.all(is_close, dim=-1) # [B, N_cells, 4, max_explored_cells]
            is_explored_per_neighbor = torch.any(is_explored_per_neighbor_and_explored_center & explored_mask_for_padded.unsqueeze(1).unsqueeze(1), dim=-1) # [B, N_cells, 4]

            # A neighbor is "valid" if it's not the 100.0, 100.0 out-of-bounds marker
            out_of_bounds_marker = torch.tensor([100.0, 100.0], device=device)
            is_valid_neighbor = ~torch.isclose(self.candidate_frontiers, out_of_bounds_marker, atol=1e-6).all(dim=-1)
            # is_valid_neighbor shape: [B, N_cells, 4]

            # A neighbor is unexplored if it's valid AND not explored
            unexplored_neighbors_count = (~is_explored_per_neighbor).float() * is_valid_neighbor.float()
            frontiers_per_cell = unexplored_neighbors_count.sum(dim=-1) # [B, N_cells]

        # Explored (bool)
        explored = self.discrete_cell_explored.float()  # [B, N_cells]

        # Stack features into last dimension
        self.discrete_cell_features = torch.stack(
            [
                tasks_per_cell,
                obstacles_per_cell,
                agents_per_cell,
                frontiers_per_cell,
                explored,
            ],
            dim=-1,
        )  # [B, N_cells, 5]


    def _compute_frontier_pts(self):
        # This function computes the 4 potential neighbors for each cell in the discrete grid,
        # storing them in `self.candidate_frontiers`.
        # Neighbors outside the world bounds are marked with a large value (100.0, 100.0).

        B, N_cells, _ = self.discrete_cell_centers.shape
        device = self.discrete_cell_centers.device
        
        # Define offsets for the 4 direct neighbors (left, right, down, up)
        neighbor_offsets = torch.tensor([
            [-self.discrete_resolution, 0],   # left
            [self.discrete_resolution, 0],    # right
            [0, -self.discrete_resolution],   # down
            [0, self.discrete_resolution],    # up
        ], device=device, dtype=torch.float32)

        # Broadcast addition to get all candidate neighbor positions for all cells in all batches
        # self.discrete_cell_centers: [B, N_cells, 2]
        # neighbor_offsets: [4, 2] -> unsqueeze to [1, 1, 4, 2] for broadcasting
        candidate_neighbors = self.discrete_cell_centers.unsqueeze(2) + neighbor_offsets.unsqueeze(0).unsqueeze(0)
        # Resulting shape: [B, N_cells, 4, 2]

        # Define world bounds
        x_min, x_max = -self.world_spawning_x, self.world_spawning_x
        y_min, y_max = -self.world_spawning_y, self.world_spawning_y

        # Check if candidate neighbor positions are within world bounds for both x and y coordinates
        is_in_bounds_x = (candidate_neighbors[..., 0] >= x_min - 1e-6) & (candidate_neighbors[..., 0] <= x_max + 1e-6)
        is_in_bounds_y = (candidate_neighbors[..., 1] >= y_min - 1e-6) & (candidate_neighbors[..., 1] <= y_max + 1e-6)
        is_in_bounds = is_in_bounds_x & is_in_bounds_y # [B, N_cells, 4]

        # Create a tensor for out-of-bounds positions, using the same shape as candidate_neighbors
        out_of_bounds_val = torch.tensor([100.0, 100.0], device=device, dtype=torch.float32)
        out_of_bounds_positions_tensor = out_of_bounds_val.expand_as(candidate_neighbors)

        # Use torch.where to assign candidate_neighbors if in bounds, else out_of_bounds_positions
        self.candidate_frontiers = torch.where(is_in_bounds.unsqueeze(-1), candidate_neighbors, out_of_bounds_positions_tensor)

    def _get_frontier_pts(self):
        # This function identifies true "frontier" points (neighbors of explored cells that are themselves unexplored).
        # It returns a padded tensor suitable for observation.

        B, N_cells, N_neighbors, _ = self.candidate_frontiers.shape
        device = self.candidate_frontiers.device

        # Get the subset of candidate_frontiers that originate from explored cells
        # `frontiers_from_explored_cells_masked`: [B, N_cells], True if cell is explored.
        # This will select rows from `self.candidate_frontiers` where the corresponding cell is explored.
        # The result will be a list of tensors, as the number of explored cells can vary per batch.
        frontiers_from_explored_cells_list = [
            self.candidate_frontiers[b][self.discrete_cell_explored[b]] for b in range(B)
        ]

        # Pad these lists to a common maximum length to create a single tensor for all batches.
        # This is necessary for `torch.stack` in the observation function.
        max_num_frontiers_per_batch = max([f.shape[0] if f.shape[0] > 0 else 0 for f in frontiers_from_explored_cells_list])

        if max_num_frontiers_per_batch == 0:
            # If no cells are explored or no valid frontiers, return a tensor of out-of-bounds markers
            return torch.full((B, N_cells, N_neighbors, 2), 100.0, device=device) # Return a minimal tensor, to be stacked

        padded_frontiers_list = []
        for b in range(B):
            current_frontiers = frontiers_from_explored_cells_list[b]
            num_current_frontiers = current_frontiers.shape[0]
            if num_current_frontiers < max_num_frontiers_per_batch:
                padding_shape = (max_num_frontiers_per_batch - num_current_frontiers, N_neighbors, 2)
                padding = torch.full(padding_shape, 100.0, device=device) # Use out-of-bounds marker for padding
                current_frontiers = torch.cat([current_frontiers, padding], dim=0)
            padded_frontiers_list.append(current_frontiers)
        
        # Stack the padded list into a single tensor: [B, MaxNumFrontiers, 4, 2]
        frontiers_tensor = torch.stack(padded_frontiers_list, dim=0) 

        # Now, for each point in `frontiers_tensor`, check if it's an *already explored* cell itself.
        # If it is, it's not a true frontier. Also filter out initial out-of-bounds markers.

        # Create a padded tensor for all explored cell centers to compare against
        # This is the same logic as in `_updated_discrete_cell_features`
        max_explored_cells_overall = self.discrete_cell_explored.sum(dim=1).max().item()
        
        if max_explored_cells_overall == 0:
            # If no cells are explored anywhere, then any neighbor that's not initially out-of-bounds is a frontier.
            # The `frontiers_tensor` already contains initial out-of-bounds values, so it's ready.
            return frontiers_tensor

        padded_all_explored_centers = torch.full((B, max_explored_cells_overall, 2), float('nan'), device=device)
        all_explored_mask = torch.zeros((B, max_explored_cells_overall), dtype=torch.bool, device=device)

        for b in range(B):
            explored_indices = torch.nonzero(self.discrete_cell_explored[b], as_tuple=False).squeeze(1)
            num_explored = explored_indices.shape[0]
            if num_explored > 0:
                padded_all_explored_centers[b, :num_explored] = self.discrete_cell_centers[b, explored_indices]
                all_explored_mask[b, :num_explored] = True
        
        # Reshape frontiers_tensor to [B, TotalNumCandidateFrontiers, 2] for easier comparison
        reshaped_frontiers = frontiers_tensor.view(B, -1, 2)
        
        # Calculate difference between each reshaped_frontier point and each explored center
        # [B, TotalNumCandidateFrontiers, 1, 2] - [B, 1, NumExp, 2]
        diff = reshaped_frontiers.unsqueeze(2) - padded_all_explored_centers.unsqueeze(1)
        # Result: [B, TotalNumCandidateFrontiers, NumExp, 2]

        is_close = torch.isclose(diff, torch.zeros_like(diff), atol=1e-6)
        # Result: [B, TotalNumCandidateFrontiers, NumExp, 2]

        # Check if a frontier point matches any *valid* explored center (within tolerance)
        is_explored_neighbor_per_center = torch.all(is_close, dim=-1) # [B, TotalNumCandidateFrontiers, NumExp]
        is_explored_neighbor = torch.any(is_explored_neighbor_per_center & all_explored_mask.unsqueeze(1), dim=-1) # [B, TotalNumCandidateFrontiers]
        
        # Identify points that were initially marked as out-of-bounds (from _compute_frontier_pts)
        out_of_bounds_marker = torch.tensor([100.0, 100.0], device=device)
        is_initial_oob = torch.isclose(reshaped_frontiers, out_of_bounds_marker, atol=1e-6).all(dim=-1) # [B, TotalNumCandidateFrontiers]

        # A point is a valid true frontier if it's NOT an already explored cell AND NOT an initial out-of-bounds marker
        is_true_frontier_mask = (~is_explored_neighbor) & (~is_initial_oob) # [B, TotalNumCandidateFrontiers]

        # Apply the mask: set non-frontier points to the out-of-bounds marker
        final_frontiers_reshaped = torch.where(is_true_frontier_mask.unsqueeze(-1), reshaped_frontiers, out_of_bounds_marker)

        # Reshape back to original structure: [B, MaxNumFrontiers, 4, 2]
        return final_frontiers_reshaped.view(B, max_num_frontiers_per_batch, N_neighbors, 2)

    
    def observation(self, agent: Agent):
        # Global observation for learner and local agent observation for planner
        local_obs = {
            "obs_tasks": torch.stack(
                [task.state.pos for task in self.tasks],
                dim=1
            ),
            "obs_agents": torch.stack(
                [a.state.pos for a in self.world.agents if a.name != agent.name],
                dim=1
            ),
            "obs_obstacles": torch.stack(
                [obstacle.state.pos for obstacle in self.obstacles],
                dim=1
            ),
            "obs_frontiers": self.frontiers,
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

        # Collect mothership global obs for role assignments
        global_obs = {
            "cell_feats": self.discrete_cell_features,
            "cell_pos": self.stored_explored_cell_centers,
            "rob_pos": torch.stack(
                [agent.state.pos for agent in self.world.agents],
                dim=1
            ),
        }

        return global_obs

    def done(self) -> Tensor:
        return torch.full(
            (self.world.batch_dim,), False, device=self.world.device)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "completed_tasks": self.completed_tasks,
            "agent_tasks": agent.tasks_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms = [
            ScenarioUtils.plot_entity_rotation(agent, env_index)
            for agent in self.world.agents
            if not isinstance(agent.dynamics, Holonomic)
        ]   # Plot the rotation for non-holonomic agents

        # Plot explored regions
        side_len = self.discrete_resolution/2
        square = [[-side_len, -side_len],
                  [-side_len, side_len],
                  [side_len, side_len],
                  [side_len, -side_len]]
        
        # Only render the cells that are marked as explored for the given env_index
        explored_cell_centers_for_render = self.discrete_cell_centers[env_index][self.discrete_cell_explored[env_index]]

        for center in explored_cell_centers_for_render:
            cell = rendering.make_polygon(square, filled=True)
            xform = rendering.Transform()
            xform.set_translation(center[0].item(), center[1].item()) # .item() for rendering positions
            cell.add_attr(xform)
            cell.set_color(0.95,0.95,0.95, 0.2) # Light gray with transparency
            geoms.append(cell)
        
        return geoms


if __name__ == "__main__":
    scenario = Scenario()
    render_interactively(scenario, display_info=False)