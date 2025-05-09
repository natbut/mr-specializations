#  Copyright (c) 2025.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import Dict, List

import torch
import numpy as np

from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar

try:
    from agents.planning_agent import PlanningAgent
except:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('agents', '..')))
    print("\n",sys.path)
    from agents.planning_agent import PlanningAgent

from vmas.simulator.utils import (
    ANGULAR_FRICTION,
    Color,
    DRAG,
    LINEAR_FRICTION,
    ScenarioUtils,
)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        ################
        # Scenario configuration
        ################
        self.plot_grid = False  # You can use this to plot a grid under the rendering for visualization purposes

        self.n_agents_holonomic = kwargs.pop(
            "n_agents_holonomic", 1
        )  # Number of agents with holonomic dynamics
        self.n_agents_diff_drive = kwargs.pop(
            "n_agents_diff_drive", 0
        )  # Number of agents with differential drive dynamics
        self.n_agents_car = kwargs.pop(
            "n_agents_car", 0
        )  # Number of agents with car dynamics
        self.n_agents = (
            self.n_agents_holonomic + self.n_agents_diff_drive + self.n_agents_car
        )
        self.n_tasks = kwargs.pop("n_tasks", 2)
        self.n_obstacles = kwargs.pop("n_obstacles", 2)

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
        self.lidar_range = kwargs.pop("lidar_range", 0.3)  # Range of the LIDAR sensor
        self.n_lidar_rays = kwargs.pop(
            "n_lidar_rays", 12
        )  # Number of LIDAR rays around the agent, each ray gives an observation between 0 and lidar_range

        self._agents_per_task = kwargs.pop(
            "agents_per_task", 1
            )

        self.shared_rew = kwargs.pop(
            "shared_rew", False
        )  # Whether the agents get a global or local reward for going to their goals
        
        self.task_comp_range = kwargs.pop(
            "task_comp_range", 0.25
        )
        self.tasks_respawn = kwargs.pop(
            "tasks_respawn", False
        )
        self.complete_task_coeff = kwargs.pop(
            "task_reward", 1
        )
        self.time_penalty = kwargs.pop(
            "time_penalty", -0.1
        )

        self.agent_radius = kwargs.pop(
            "agent_radius", 0.1
            )
        
        self.min_distance_between_entities = (
            self.agent_radius * 4 + 0.05
        )  # Minimum distance between entities at spawning time
        self.min_collision_distance = (
            0.005  # Minimum distance between entities for collision trigger
        )

        ScenarioUtils.check_kwargs_consumed(kwargs) # Warn is not all kwargs have been consumed


        ################
        # Make world
        ################
        world = World(
            batch_dim,  # Number of environments simulated
            device,  # Device for simulation
            substeps=5,  # Number of physical substeps (more yields more accurate but more expensive physics)
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
        ]  # Colors for the first 7 agents
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )  # Other colors if we have more agents are random

        self.agent_color = Color.BLUE
        self.task_color = Color.ORANGE

        self.agents = []
        for i in range(self.n_agents):
            color = self.agent_color
            # color = (
            #     known_colors[i]
            #     if i < len(known_colors)
            #     else colors[i - len(known_colors)]
            # )  # Get color for agent

            sensors = [
                Lidar(
                    world,
                    n_rays=self.n_lidar_rays,
                    max_range=self.lidar_range,
                    entity_filter=lambda e: isinstance(
                        e, Agent
                    ),  # This makes sure that this lidar only percieves other agents
                    angle_start=0.0,  # LIDAR angular ranges (we sense 360 degrees)
                    angle_end=2
                    * torch.pi,  # LIDAR angular ranges (we sense 360 degrees)
                )
            ]  # Agent LIDAR sensor

            if i < self.n_agents_holonomic:
                agent = PlanningAgent(
                    name=f"holonomic_{i}",
                    collide=True,
                    color=color,
                    render_action=True,
                    sensors=sensors,
                    shape=Sphere(radius=self.agent_radius),
                    u_range=[1, 1],  # Ranges for actions
                    u_multiplier=[3, 3],  # Action multipliers
                    dynamics=Holonomic(),  # If you go to its class you can see it has 2 actions: force_x, and force_y
                )
            elif i < self.n_agents_holonomic + self.n_agents_diff_drive:
                agent = PlanningAgent(
                    name=f"diff_drive_{i - self.n_agents_holonomic}",
                    collide=True,
                    color=color,
                    render_action=True,
                    sensors=sensors,
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
                    name=f"car_{i-self.n_agents_holonomic-self.n_agents_diff_drive}",
                    collide=True,
                    color=color,
                    render_action=True,
                    sensors=sensors,
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
        for i in range(self.n_tasks):
            color = self.task_color
            task = Landmark(
                name=f"goal_{i}",
                collide=False,
                color=color,
            )
            world.add_landmark(task)
            self.tasks.append(task)


        ################
        # Add obstacles
        ################
        self.obstacles = (
            []
        )  # We will store obstacles here for easy access
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                color=Color.BLACK,
                shape=Sphere(radius=self.agent_radius * 2 / 3),
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)


        ################
        # Init Reward Tracking
        ################        
        self.completed_tasks = torch.zeros((batch_dim, self.n_tasks), device=device)
        self.shared_tasks_rew = torch.zeros(batch_dim, device=device)

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents
            + self.obstacles
            + self.tasks,  # List of entities to spawn
            self.world,
            env_index,  # Pass the env_index so we only reset what needs resetting
            self.min_distance_between_entities,
            x_bounds=(-self.world.x_semidim, self.world.x_semidim),
            y_bounds=(-self.world.y_semidim, self.world.y_semidim),
        )

        if self.static_env:
            for i, agent in enumerate(self.agents):
                agent.set_pos(
                    torch.tensor(
                        [0, 0],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

            for i, task in enumerate(self.tasks):
                i += 1
                loc = 0.35*i*(-1)**(i)
                coords = [-loc, loc]
                task.set_pos(
                    torch.tensor(
                        [coords[0], coords[1]],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

            for i, obs in enumerate(self.obstacles):
                i += 1
                loc = 0.35*i*(-1)**(i)
                coords = [-loc, -loc]
                obs.set_pos(
                    torch.tensor(
                        [coords[0], coords[1]],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )


    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

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

            # Allocate completion credit to passengers
            self.shared_tasks_rew[:] = 0
            for a in self.world.agents:
                self.shared_tasks_rew += self.agent_tasks_reward(a)
        
        # Respawn tasks
        if is_last:
            if self.tasks_respawn:
                occupied_positions_agents = [self.agents_pos]
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
                    pos = ScenarioUtils.find_random_pos_for_entity(
                        occupied_positions,
                        env_index=None,
                        world=self.world,
                        min_dist_between_entities=self.min_distance_between_entities,
                        x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                        y_bounds=(-self.world.y_semidim, self.world.y_semidim),
                    )

                    task.state.pos[self.completed_tasks[:, i]] = pos[
                        self.completed_tasks[:, i]
                    ].squeeze(1)
            # else:
                # self.all_time_covered_targets += self.completed_tasks
            #     for i, task in enumerate(self.tasks):
            #         task.state.pos[self.completed_tasks[:, i]] = self.get_outside_pos(
            #             None
            #         )[self.completed_tasks[:, i]]

        tasks_reward = (
            self.shared_tasks_rew if self.shared_rew else agent.tasks_rew
        )  # Choose global or local reward based on configuration

        return tasks_reward + self.time_rew
    
    def agent_tasks_reward(self, agent):
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
        return agent.tasks_rew

    def observation(self, agent: Agent):
        obs = {
            
            "obs_tasks": torch.stack(
                [task.state.pos for task in self.tasks],
                # dim=-1
            ),
            "obs_agents": torch.stack(
                [agent.state.pos for agent in self.world.agents],
                # dim=-1
            ),
            "obs_obstacles": torch.stack(
                [obstacle.state.pos for obstacle in self.obstacles],
                # dim=-1
            ),
            # : torch.cat(
            #     [
            #         task.state.pos for task in self.tasks
            #     ]
            #     + [
            #         obstacle.state.pos for obstacle in self.obstacles
            #     ],
            #     dim=-1,
            # ),
            "pos": agent.state.pos,
            "vel": agent.state.vel,
        }
        if not isinstance(agent.dynamics, Holonomic):
            # Non-holonomic agents need to know angular states
            obs.update(
                {
                    "rot": agent.state.rot,
                    "ang_vel": agent.state.ang_vel,
                }
            )
        return obs

    def done(self) -> Tensor:
        # print("Completed tasks:", self.completed_tasks)
        return self.completed_tasks.any(dim=-1) #self.all_goal_reached

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

        # Task ranges
        for target in self.tasks:
            range_circle = rendering.make_circle(self.task_comp_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*self.task_color.value)
            geoms.append(range_circle)

        # Plot communication lines
        if self.comms_rendering_range > 0:
            for i, agent1 in enumerate(self.world.agents):
                for j, agent2 in enumerate(self.world.agents):
                    if j <= i:
                        continue
                    agent_dist = torch.linalg.vector_norm(
                        agent1.state.pos - agent2.state.pos, dim=-1
                    )
                    if agent_dist[env_index] <= self.comms_rendering_range:
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
    render_interactively(scenario)