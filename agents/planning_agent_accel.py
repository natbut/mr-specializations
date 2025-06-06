# Nathan Butler

import typing
from typing import Dict, List, Callable, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World, Shape, Sensor, Dynamics, Entity
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
    # from agents.planning_agent import PlanningAgent # This was causing circular import, it's the class itself.

from vmas.simulator.utils import (ANGULAR_FRICTION, DRAG, LINEAR_FRICTION,
                                   Color, ScenarioUtils)

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class PlanningAgent(Agent):
    
    def __init__(
        self,
        name: str,
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
        
        self.obs = {} # Changed to dict to store the obs from Scenario
        self.trajs = [] # List of trajectories, one for each batch
        self.traj_idx = torch.tensor([], dtype=torch.long) # Tensor to store current waypoint index for each batch
        self.sim_velocity = sim_velocity
        
        self.control_action_dict = {str([0.0,0.0]) : 0,
                                str([0.0,-1.0]): 1,
                                str([0.0, 1.0]): 2,
                                str([-1.0,0.0]): 3,
                                str([-1.0,-1.0]): 4,
                                str([-1.0,1.0]): 5,
                                str([1.0,0.0]): 6,
                                str([1.0,-1.0]): 7,
                                str([1.0,1.0]): 8,             
                                }
        
        super().__init__(name, shape, movable, rotatable, collide, density, mass, f_range, max_f, t_range, max_t, v_range, max_speed, color, alpha, obs_range, obs_noise, u_noise, u_range, u_multiplier, action_script, sensors, c_noise, silent, adversary, drag, linear_friction, angular_friction, gravity, collision_filter, render_action, dynamics, action_size, discrete_action_nvec)
    

    def _compute_trajectory_cont(self,
                                 world_idx: int,
                                 current_pos: Tensor, 
                                 heuristic_weights: Tensor, 
                                 heuristic_eval_fns: List[Callable], 
                                 horizon: float = 0.25, 
                                 max_pts: int = 25, 
                                 verbose: bool = False) -> List[Tensor]:
        """
        Computes a continuous trajectory using an RRT-like algorithm for a single environment batch.
        This function remains largely iterative due to the sequential nature of RRT,
        but internal helper functions are vectorized where possible.
        """

        # Establish sampling range based on current position and horizon
        samp_rng_x = (max(-1.0, current_pos[0].item() - horizon), 
                      min(1.0, current_pos[0].item() + horizon))
        samp_rng_y = (max(-1.0, current_pos[1].item() - horizon), 
                      min(1.0, current_pos[1].item() + horizon))
        
        if verbose: print(f"Sampling range from current pos {current_pos}: \nx rng: {samp_rng_x} \ny rng: {samp_rng_y}")

        # Initialize tree with current position as root
        # V will be a list of tensors for dynamic growth, then converted to a single tensor for vectorized operations
        V = [current_pos.clone()] 
        E = [] # Edges
        parents = {0: None} # Parent mapping for path reconstruction
        costs = {0: 0.0} # Cost from root to node

        rad = 0.2 * horizon # Neighborhood radius for rewiring

        num_samps = 0
        
        while num_samps < max_pts:
            # Sample a single point within the trajectory horizon
            samp_pos = self._random_pos(samp_rng_x, samp_rng_y, current_pos.dtype, current_pos.device)
            if verbose: print(f"Sampled pt: {samp_pos}")
            num_samps += 1

            # Check if sampled point is in an obstacle (vectorized check)
            if not self._obstacle_free_check(world_idx, samp_pos.unsqueeze(0)).item(): # Unsqueeze for _obstacle_free_check
                if verbose: print("Pt in obstacle")
                continue

            # Convert V to a single tensor for vectorized distance calculations
            V_stacked = torch.stack(V) if len(V) > 0 else torch.empty(0, 2, dtype=current_pos.dtype, device=current_pos.device)

            # Find nearest point in search tree
            idx_nearest = self._find_nearest_node(V_stacked, samp_pos).item()
            
            # Find neighbors and best neighbor for rewiring
            idx_best, neighbors = self._find_neighbors(V_stacked, samp_pos, costs, rad)

            # Only link if path to best parent is obstacle-free (vectorized check)
            if self._is_path_obstacle_free(V[idx_best], samp_pos, world_idx):
                if verbose: print("Path is obstacle free")
                
                # Add sampled position to vertices and update parent/cost
                new_node_idx = len(V)
                V.append(samp_pos)
                parents[new_node_idx] = idx_best
                costs[new_node_idx] = costs[idx_best.item()] + torch.norm(samp_pos - V[idx_best]).item()
                E.append((idx_best, new_node_idx))

                # Rewire neighbors if cheaper to use new point and path is obstacle-free
                # Neighbors are now a tensor of indices
                if neighbors.numel() > 0: # Check if there are any neighbors
                    # Filter out the new_node_idx itself if it happens to be in neighbors (shouldn't if logic is correct)
                    neighbors_filtered = neighbors[neighbors != new_node_idx] 
                    
                    # Compute new costs for neighbors through the new node
                    # Costs from new node to neighbors: [num_neighbors]
                    dists_to_neighbors = torch.norm(samp_pos - V_stacked[neighbors_filtered], dim=-1)
                    new_costs_through_new_node = costs[new_node_idx] + dists_to_neighbors
                    
                    # Compare new costs with existing costs for neighbors
                    current_costs_of_neighbors = torch.tensor([costs[idx.item()] for idx in neighbors_filtered], device=V_stacked.device)
                    should_rewire = new_costs_through_new_node < current_costs_of_neighbors

                    for i, idx_n in enumerate(neighbors_filtered[should_rewire]):
                        # Check path obstacle-free for rewiring (iterative as it's one path at a time)
                        if self._is_path_obstacle_free(samp_pos, V[idx_n.item()], world_idx):
                            parents[idx_n.item()] = new_node_idx
                            costs[idx_n.item()] = new_costs_through_new_node[i].item()
                            E.append((new_node_idx, idx_n.item()))

        # If no samples are added to V (e.g., all samples are in obstacles), return empty trajectory or initial point
        if len(V) == 1: # Only contains the initial current_pos
            return [current_pos] # Return just the current position if no path can be found

        # Extract path: find node with best heuristic value (vectorized)
        V_final_stacked = torch.stack(V) # Stack all vertices for final evaluation
        
        vals = torch.zeros(len(V_final_stacked), device=current_pos.device, dtype=current_pos.dtype)
        for i, fn in enumerate(heuristic_eval_fns):
            # Assumes heuristic_eval_fns can handle batch evaluation across multiple points.
            # If not, this loop would still be necessary.
            fn_out = heuristic_weights[i] * fn(self.obs, world_idx, V_final_stacked) # fn should return [N_V]
            vals += fn_out
            if verbose: print(f"Heuristic eval for {fn.__name__} for points in V with w {heuristic_weights[i]}: {fn_out}")
        
        goal_idx = int(torch.argmin(vals))
        if verbose: print(f"Goal idx:{goal_idx}")

        # Backtrack to root to get trajectory
        traj = []
        idx = goal_idx
        while idx is not None:
            traj.append(V[idx])
            idx = parents[idx]
        traj = traj[::-1]
        if verbose: print(f"Backtracked traj: {traj}")
        
        return traj
    
    def _is_path_obstacle_free(self, start: Tensor, end: Tensor, world_idx: int, num_checks: int = 10) -> bool:
        """
        Checks if the path between start and end is obstacle-free.
        Interpolates points along the path and uses vectorized obstacle checking.
        """
        # Interpolate points along the path in a vectorized manner
        alphas = torch.linspace(0, 1, steps=num_checks, device=start.device, dtype=start.dtype).unsqueeze(-1) # [num_checks, 1]
        # interp: [num_checks, 2]
        interp_points = start * (1 - alphas) + end * alphas

        # Perform obstacle check for all interpolated points simultaneously
        # _obstacle_free_check expects [N_points, 2]
        # If any point is not obstacle-free, the path is not obstacle-free
        if not self._obstacle_free_check(world_idx, interp_points).all().item():
            # if verbose: print("OBSTACLE IN PATH")
            return False
        return True

    def _obstacle_free_check(self, world_idx: int, positions: Tensor, buffer: float = 0.1, verbose: bool = False) -> Tensor:
        """
        Returns a boolean tensor indicating if each position in `positions` is obstacle-free.
        `positions` expected shape: [N_points, 2]
        """
        # Get obstacles positions for the current world_idx
        # self.obs["obs_obstacles"] is [B, N_obstacles, 2]
        obstacles_pos = self.obs["obs_obstacles"][world_idx] # [N_obstacles, 2]
        
        if obstacles_pos.numel() == 0: # No obstacles present
            return torch.full((positions.shape[0],), True, device=positions.device, dtype=torch.bool)

        # Calculate squared Euclidean distance from each position to each obstacle
        # positions: [N_points, 1, 2]
        # obstacles_pos: [1, N_obstacles, 2]
        # diff: [N_points, N_obstacles, 2]
        diff = positions.unsqueeze(1) - obstacles_pos.unsqueeze(0)
        
        # sq_dists: [N_points, N_obstacles]
        sq_dists = torch.sum(diff ** 2, dim=-1)
        
        # If any distance is less than buffer, it's NOT clear.
        # clearances: [N_points, N_obstacles]
        clearances = sq_dists >= (buffer ** 2) 
        
        # A position is obstacle-free if it's clear of ALL obstacles
        # all_clear_for_pos: [N_points]
        all_clear_for_pos = clearances.all(dim=-1) 

        if verbose: 
            print(f"world {world_idx} obstacle locs:\n", obstacles_pos)
            print("Clearances:", all_clear_for_pos)

        return all_clear_for_pos
        
    def _random_pos(self, samp_rng_x: tuple, samp_rng_y: tuple, dtype: torch.dtype, device: torch.device) -> Tensor:
        """
        Sample a single random position within specified ranges.
        This remains non-vectorized for compatibility with the RRT loop's iterative sampling.
        """
        x = torch.empty(1, dtype=dtype, device=device).uniform_(samp_rng_x[0], samp_rng_x[1])
        y = torch.empty(1, dtype=dtype, device=device).uniform_(samp_rng_y[0], samp_rng_y[1])
        return torch.tensor([x.item(), y.item()], dtype=dtype, device=device)

    def _find_nearest_node(self, vertices: Tensor, pos: Tensor) -> Tensor:
        """
        Returns the index of the nearest node from `vertices` to `pos`.
        `vertices` expected shape: [N_vertices, 2]
        `pos` expected shape: [2]
        """
        if vertices.numel() == 0:
            return torch.tensor(-1, device=pos.device, dtype=torch.long) # Or handle error appropriately
        
        # Calculate distances from `pos` to all `vertices` in a vectorized manner
        # `vertices` is [N_vertices, 2], `pos` is [2]
        # `dists` will be [N_vertices]
        dists = torch.norm(vertices - pos, dim=-1)
        return torch.argmin(dists)

    def _find_neighbors(self, vertices: Tensor, samp_pos: Tensor, costs: Dict[int, float], radius: float) -> tuple[Tensor, Tensor]:
        """
        Get neighbors within search radius and the best neighbor (nearest) within radius.
        `vertices` expected shape: [N_vertices, 2]
        `samp_pos` expected shape: [2]
        `costs` is a dict {node_idx: cost_value}
        """
        if vertices.numel() == 0:
            return torch.tensor(-1, device=samp_pos.device, dtype=torch.long), torch.tensor([], dtype=torch.long, device=samp_pos.device)

        # Calculate distances from `samp_pos` to all `vertices`
        dists = torch.norm(vertices - samp_pos, dim=-1) # [N_vertices]
        
        # Identify neighbors within the search radius
        # `neighbors` will be a tensor of indices [num_neighbors]
        neighbors_indices = (dists < radius).nonzero(as_tuple=True)[0]
        
        if neighbors_indices.numel() > 0:
            # Calculate costs from samp_pos to each neighbor
            # This is `costs[i] + dist(samp_pos, vertices[i])`
            # For each neighbor_idx, we need its cost from `costs` dict and distance from `dists` tensor
            
            # Extract costs for identified neighbors
            current_costs_of_neighbors = torch.tensor([costs[idx.item()] for idx in neighbors_indices], device=dists.device, dtype=dists.dtype)
            
            # Combine current costs with distances from samp_pos to get total costs
            # Costs to neighbors from root through samp_pos if it were a parent
            costs_through_samp_pos = current_costs_of_neighbors + dists[neighbors_indices]
            
            # Find the best neighbor among these (the one that minimizes cost through samp_pos)
            best_idx_in_neighbors_list = torch.argmin(costs_through_samp_pos)
            best_idx = neighbors_indices[best_idx_in_neighbors_list]
        else:
            # If no neighbors within radius, the "best" neighbor is simply the nearest node overall
            best_idx = self._find_nearest_node(vertices, samp_pos)

        return best_idx, neighbors_indices
    

    def get_control_action_cont(self, heuristic_weights: Tensor, heuristic_eval_fns: List[Callable], horizon: float, verbose: bool = False) -> Tensor:
        """
        Determines the next control action for the agent based on RRT-like planning.
        Optimized to handle multiple environments (batches) in a vectorized manner where possible.
        """
        current_pos = self.state.pos # [B, 2]
        batch_dim = current_pos.shape[0]

        if verbose: print(f"Agent {self.name} heuristic weights:\n {heuristic_weights}")

        # Initialize trajectories if not already done. This part still involves a loop for _compute_trajectory_cont.
        # This loop runs once at the beginning or when all trajectories are 'empty'.
        if not self.trajs: # Check if the list is empty
            if verbose: print("Initializing trajectories...")
            self.trajs = [[] for _ in range(batch_dim)] # Initialize an empty list for each batch
            self.traj_idx = torch.zeros(batch_dim, dtype=torch.long, device=current_pos.device)

            for b in range(batch_dim):
                # _compute_trajectory_cont operates on a single environment (batch_idx)
                self.trajs[b] = self._compute_trajectory_cont(
                    b, current_pos[b], heuristic_weights[b], heuristic_eval_fns, horizon, verbose=False
                )
                # traj_idx for each batch is already 0 at initialization

        # Stack current target waypoints for all batches
        # Ensure that self.trajs[b] has at least one waypoint
        target_waypt = torch.stack([
            self.trajs[b][self.traj_idx[b].item()] if self.trajs[b] and self.traj_idx[b].item() < len(self.trajs[b]) else current_pos[b] # Fallback to current pos if traj is empty/index out of bounds
            for b in range(batch_dim)
        ]) # [B, 2]

        if verbose: print("Target waypoints:", target_waypt)

        # Vectorized check for arriving at target waypoints
        # dists_to_waypoint: [B]
        dists_to_waypoint = torch.norm(current_pos - target_waypt, dim=-1)
        
        # arrived_at_waypoint: [B] boolean tensor
        arrived_at_waypoint = dists_to_waypoint < 0.05

        # Update traj_idx for batches that have arrived at their waypoint
        # If arrived, increment, otherwise keep current index
        next_traj_idx_candidates = self.traj_idx + 1
        
        # Ensure next_traj_idx doesn't exceed the length of the current trajectory for each batch
        # This requires iterating, as each trajectory length is different
        new_traj_idx = self.traj_idx.clone()
        batches_to_replan = []

        for b in range(batch_dim):
            if arrived_at_waypoint[b]:
                old_idx = self.traj_idx[b].item()
                len_traj = len(self.trajs[b]) if self.trajs[b] else 0 # Handle empty trajectory case
                
                # Check if we are at the end of the trajectory
                if len_traj == 0 or old_idx >= len_traj - 1: # Reached end or empty traj
                    batches_to_replan.append(b)
                    new_traj_idx[b] = 0 # Reset to 0 for the new trajectory
                else:
                    new_traj_idx[b] = old_idx + 1
        
        self.traj_idx = new_traj_idx

        # Recompute trajectories only for batches that completed their current trajectory
        if batches_to_replan:
            for b_idx in batches_to_replan:
                if verbose: print(f"AGENT {self.name} BATCH {b_idx} COMPLETED TRAJ. Located at {current_pos[b_idx]}. Recomputing...")
                new_traj = self._compute_trajectory_cont(
                    b_idx, current_pos[b_idx], heuristic_weights[b_idx], heuristic_eval_fns, horizon, verbose=False
                )
                if verbose: print("New traj:", new_traj)
                self.trajs[b_idx] = new_traj
                # traj_idx[b_idx] is already reset to 0 above

        # Get the new target waypoints after updates
        final_target_waypt = torch.stack([
            self.trajs[b][self.traj_idx[b].item()] if self.trajs[b] and self.traj_idx[b].item() < len(self.trajs[b]) else current_pos[b]
            for b in range(batch_dim)
        ])

        # Get control action to reach next waypt (fully vectorized)
        pos_diff = final_target_waypt - current_pos # [B, 2]
        if verbose: print("Pos diff:", pos_diff)
        
        # Clip the action to max_speed in a vectorized manner
        u_action = torch.clamp(pos_diff, -self.max_speed, self.max_speed) # [B, 2]

        return u_action

    


    def _compute_dist_heuristics_val(self, cur_node_pos, node_pos, node_features, heuristic_weights, verbose = False):
        """
        Computes euclidean distance heuristic from cur_node to each node in graph that contains non-zero features.
        """
        total_val = 0
        for i, node_vec in enumerate(node_features): #x
            # Compute dist from cur_pos to node_pos; fill for each feature present at node
            h_vec = torch.where(node_vec == 1, torch.norm(node_pos[i]-cur_node_pos), 0)
            # if verbose: print(f"Heuristic eval for {node_vec}: {h_vec}")
            # if verbose: print("Heuristic weights:", heuristic_weights, " shape:", heuristic_weights.shape)

            # Sum weighted heuristics
            # print("Heuristic weights:", heuristic_weights)
            h_val = torch.dot(heuristic_weights, h_vec)
            # if verbose: print("!!! VAL:", h_val)
            total_val += h_val

        if verbose: print("=== H VAL:", total_val, " ===")
        return total_val

    def _compute_trajectory_graph(self, start_node_idxs, graphs, heuristic_weights, horizon, verbose=False):
        """
        Use a planner to compute min cost path through each graph in batch
        """
        if verbose: print("Start nodes:", start_node_idxs)

        # print(f"Agent {self.name} heuristic weights:", heuristic_weights)

        # Plan trajectory (here we use A* search) f = g + h
        all_traj = []
        for i, graph in enumerate(graphs): # iterate through batches
            start_idx = start_node_idxs[i]
            if verbose: print("start:", start_idx)
            num_nodes = len(graph.pos)
            parents = {}
            g_score = {node: float('inf') for node in range(num_nodes)}
            h_score = {node: float('inf') for node in range(num_nodes)}
            f_score = {node: float('inf') for node in range(num_nodes)}
            
            # TODO Heuristic evaluation
            g_score[start_idx] = 0.0
            h_score[start_idx] = self._compute_dist_heuristics_val(graph.pos[start_idx],
                                                                    graph.pos,
                                                                    graph.x,
                                                                    heuristic_weights[i]
                                                                    )
            f_score[start_idx] = g_score[start_idx] + h_score[start_idx]
            
            # Init priority queue & score tracking
            open_set = [(f_score[start_idx], start_idx)] # (f, id)

            count = 0
            path = []
            min_h = float('inf')
            min_node = None
            while open_set:
                # Get node with lowest f_score, remove from queue
                if verbose: print("Open set:", open_set)
                _, current = min(open_set, key=lambda x: h_score[x[1]])
                # open_set = [node for node in open_set if node[1] != current]
                if verbose: print("Expanding", current)

                if h_score[current] < min_h: # NOTE we are appending lowest-cost nodes to path
                    path.append(current) 
                    min_h = h_score[current]
                    min_node = current
                else:
                    path.append(min_node)

                if count == horizon:
                    break

                neighbors = graph.edge_index[1][graph.edge_index[0] == current]
                # print(f"Neighbors of {current}: {neighbors}. \n Edge Index: \n{graph.edge_index}")
                open_set = [(f_score[current], current)]
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    tentative_g_score = g_score[current] + torch.norm(graph.pos[current] - graph.pos[neighbor])
                    if tentative_g_score < g_score[neighbor]: # faster path found
                        parents[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        h_score[neighbor] = self._compute_dist_heuristics_val(graph.pos[neighbor],
                                                                                graph.pos,
                                                                                graph.x,
                                                                                heuristic_weights[i]
                                                                                )
                        f_score[neighbor] = g_score[neighbor] + h_score[neighbor]

                        if neighbor not in [n[1] for n in open_set]: #h_score[neighbor] < h_score[current]: #
                            open_set.append((f_score[neighbor], neighbor))

                count += 1
            
            all_traj.append(path[:])

        if verbose: print("Computed trajs:", all_traj)

        return all_traj
        
    def get_control_action_graph(self, graph_batch, heuristic_weights: torch.Tensor, horizon, verbose=False):
        """
        Get control action that makes best progress towards traj

        May need to consider agent drive type
        """
        graphs_list = graph_batch.to_data_list()
        # print("X:", graphs_list[0]["x"])
        # print("Graphs list:", graphs_list)
        
        # Get nearest starting node for planning
        start_raw = self.state.pos
        # print("Start raw:", start_raw)
        if self.batch_dim != 1:
            dists = [graphs_list[i].pos - a_pos for i, a_pos in enumerate(start_raw)]
        else:
            # print("Graph pos:", graphs_list[0].pos, "A pos:", start_raw)
            dists = [graphs_list[0].pos - a_pos for i, a_pos in enumerate(start_raw)]
        # print("Before calc:", dists)
        # print("Intermediate calc:", [torch.linalg.norm(d, dim=1) for d in dists])
        norm_dists = torch.stack([torch.norm(d, dim=1) for d in dists])#.squeeze(0)
        # print("Norm dists:", norm_dists)
        start_node_idxs = [torch.argmin(norm_d).item() for norm_d in norm_dists]
        # print("Start node idxs:", start_node_idxs)

        
        # if self.batch_dim == 1:
        #     # heuristic_weights = [heuristic_weights[i][idx] for i, idx, in enumerate(start_node_idxs)]
        #     heuristic_weights = [heuristic_weights]
        # else:
            # heuristic_weights = [heuristic_weights[i] for i, idx, in enumerate(start_node_idxs)]

            
        # if verbose: print("\nWEIGHTS:", heuristic_weights)

        # Make plan
        if self.trajs == []:
            self.trajs = self._compute_trajectory_graph(start_node_idxs, graphs_list, heuristic_weights, horizon)
            self.traj_idx = [0 for _ in range(heuristic_weights.shape[0])]
        else:
            # print("Plan exists, following:", self.trajs)
            for i, traj in enumerate(self.trajs):
                # if traj == []:
                #     self.trajs = self._compute_trajectory(start_node_idxs, graphs_list, heuristic_weights, horizon)
                #     self.traj_idx[i] = 0

                if start_node_idxs[i] == traj[self.traj_idx[i]] and norm_dists[i][start_node_idxs[i]] < 0.1: #traj[1]
                    # print(f"NODE {start_node_idxs[i]} REACHED")
                    # self.trajs[i] = traj[1:]
                    self.traj_idx[i] = min(self.traj_idx[i] + 1, len(traj)-1)
                

        # Find best control action (given current pos/vel and traj)
        # Take action towards reaching next node in traj
        # print("TRajs:", self.trajs)
        # print("TRajs Idx:", self.traj_idx)
        next_node_idx = [traj[self.traj_idx[i]] for i, traj in enumerate(self.trajs)] # test commanding to node 0 [0 for traj in trajs]
        # print("Next node idxs:", next_node_idx)
        cur_pos = self.state.pos
        if self.batch_dim != 1:
            next_pos = torch.stack([graph.pos[next_node_idx[i]] for i, graph in enumerate(graphs_list)])
        else:
            next_pos = graphs_list[0].pos[next_node_idx[0]]

        # print("Next pos:", next_pos)
        # print("Cur pos:", cur_pos)
        
        pos_diff = next_pos-cur_pos

        u_action = torch.where(pos_diff > 1.0, 1.0, pos_diff)
        u_action = torch.where(u_action < -1.0, -1.0, u_action)

        return u_action