from typing import Union

import numpy as np
import torch
from vmas.simulator.core import *


class PlanningAgent(Agent):
    
    def __init__(
        self,
        name: str,
        heuristic_funcs: List = None,
        sim_action_func = None,
        sim_velocity: float = 0.05,
        shape: Shape = None,
        movable: bool = True,
        rotatable: bool = True,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        f_range: float = None,
        max_f: float = None,
        t_range: float = None,
        max_t: float = None,
        v_range: float = None,
        max_speed: float = None,
        color=Color.BLUE,
        alpha: float = 0.5,
        obs_range: float = None,
        obs_noise: float = None,
        u_noise: Union[float, Sequence[float]] = 0.0,
        u_range: Union[float, Sequence[float]] = 1.0,
        u_multiplier: Union[float, Sequence[float]] = 1.0,
        action_script: Callable[[Agent, World], None] = None,
        sensors: List[Sensor] = None,
        c_noise: float = 0.0,
        silent: bool = True,
        adversary: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
        render_action: bool = False,
        dynamics: Dynamics = None,  # Defaults to holonomic
        action_size: int = None,  # Defaults to what required by the dynamics
        discrete_action_nvec: List[
            int
        ] = None,
        
        ):
        
        self.obs = []
        self.heuristic_funcs = heuristic_funcs
        self.sim_velocity = sim_velocity
        self.control_action_dict = {0: [0.0,0.0],
                        1: [0.0,-self.sim_velocity],
                        2: [0.0,self.sim_velocity],
                        3: [-self.sim_velocity,0.0],
                        4: [-self.sim_velocity,-self.sim_velocity],
                        5: [-self.sim_velocity,self.sim_velocity],
                        6: [self.sim_velocity,0.0],
                        7: [self.sim_velocity,-self.sim_velocity],
                        8: [self.sim_velocity,self.sim_velocity],                   
                        }
        
        super().__init__(name, shape, movable, rotatable, collide, density, mass, f_range, max_f, t_range, max_t, v_range, max_speed, color, alpha, obs_range, obs_noise, u_noise, u_range, u_multiplier, action_script, sensors, c_noise, silent, adversary, drag, linear_friction, angular_friction, gravity, collision_filter, render_action, dynamics, action_size, discrete_action_nvec)



    def _compute_trajectory(self, graph_batch, heuristic_weights, horizon, verbose=False):
        """
        Use a planner to compute min cost path through each graph in batch
        """
        graphs = graph_batch.to_data_list()
        start_raw = self.state.pos

        # Get nearest starting node for planning
        dists = [graphs[i].pos - a_pos for i, a_pos in enumerate(start_raw)]
        start_node_idxs = [torch.argmin(torch.norm(d, dim=1)).item() for d in dists]
        if verbose: print("Start nodes:", start_node_idxs)

        # Plan trajectory (here we use A* search) f = g + h
        all_traj = []
        for i, graph in enumerate(graphs):
            start_idx = start_node_idxs[i]
            if verbose: print("start:", start_idx)
            num_nodes = len(graph.pos)

            # Init priority queue & score tracking
            open_set = [(0.0, start_idx)] # (f, id)
            parents = {}
            g_score = {node: float('inf') for node in range(num_nodes)}
            f_score = {node: float('inf') for node in range(num_nodes)}
            
            # TODO Heuristic evaluation
            g_score[start_idx] = 0.0
            f_score[start_idx] = g_score[start_idx]# + heuristic_weights[i] * torch.norm(graph.pos[start_idx] - graph.pos[goal_idx])

            count = 0
            while open_set:
                # Get node with lowest f_score, remove from queue
                if verbose: print("Open set:", open_set)
                _, current = min(open_set, key=lambda x: f_score[x[1]])
                open_set = [node for node in open_set if node[1] != current]
                if verbose: print("Expanding", current)

                if count == horizon:
                    # Reconstruct path
                    path = []
                    while current in parents:
                        path.append(current)
                        current = parents[current]
                    path.append(start_idx)
                    all_traj.append(path[::-1])
                    break

                neighbors = graph.edge_index[1][graph.edge_index[0] == current]
                # print(f"Neighbors of {current}: {neighbors}. \n Edge Index: \n{graph.edge_index}")

                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    # TODO update to use edge_attr to add to g_score
                    tentative_g_score = g_score[current] + torch.norm(graph.pos[current] - graph.pos[neighbor])
                    if tentative_g_score < g_score[neighbor]: # faster path found
                        parents[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        # TODO heuristic evaluation
                        f_score[neighbor] = g_score[neighbor] #+ heuristic_weights[i] * torch.norm(graph.pos[neighbor] - graph.pos[goal_idx])
                    if neighbor not in [n[1] for n in open_set]:
                        open_set.append((f_score[neighbor], neighbor))
                
                count += 1

        if verbose: print("Computed trajs:", all_traj)

        return all_traj
        
    def get_control_action(self, graph_batch, heuristic_weights, horizon):
        """
        Get control action that makes best progress towards traj

        May need to consider agent drive type
        """

        # Make plan
        traj = self._compute_trajectory(graph_batch, heuristic_weights, horizon)

        # TODO Find best control action (given current pos/vel and traj)
        cur_pos = self.state.pos