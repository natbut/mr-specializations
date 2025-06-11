from typing import Union

import torch
from vmas.simulator.core import *
import time

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
        
        self.obs = []
        self.trajs = []
        self.traj_idx = []
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
                                 world_idx,
                                 current_pos, 
                                 heuristic_weights, 
                                 heuristic_eval_fns, 
                                 rewire=False,
                                 horizon=0.25, 
                                 max_pts=10,
                                 random_sampling=True,
                                 verbose=False):

        # Establish sampling range
        samp_rng_x = (max(-1.0, min(1.0, current_pos[world_idx][0] - horizon)), 
                  min(1.0, max(-1.0, current_pos[world_idx][0] + horizon)))
        samp_rng_y = (max(-1.0, min(1.0, current_pos[world_idx][1] - horizon)), 
                  min(1.0, max(-1.0, current_pos[world_idx][1] + horizon)))
        if verbose: print(f"Sampling range from current pos {current_pos[world_idx]}: \nx rng: {samp_rng_x} \ny rng: {samp_rng_y}")

        # Initialize tree with current position as root
        V = [current_pos[world_idx].clone()]
        parents = {0: None}
        costs = {0: 0}
        rad = 0.2 * horizon  # Neighborhood radius for rewiring

        num_samps = 0

        # Sample up to max_pts points
        # TODO: Might be able to vectorize this
        # t_start = time.time()
        alpha = torch.linspace(0, 1, steps=4)
        # Uniformly sample points in a circle around the current position
        angles = torch.linspace(0.0, 2 * torch.pi, steps=max_pts)
        samp_dist_x = horizon * torch.cos(angles)
        samp_dist_y = horizon * torch.sin(angles)
        while num_samps < max_pts:
            # st_start = time.time()
            # Sample point within traj horizon
            if random_sampling:
                samp_pos = self._random_pos(samp_rng_x, samp_rng_y, dtype=current_pos.dtype, device=current_pos.device)
            else:
                # Uniformly sample points around current position
                pos_x = current_pos[world_idx][0] + samp_dist_x[num_samps]
                pos_y = current_pos[world_idx][1] + samp_dist_y[num_samps]
                samp_pos = torch.tensor([pos_x, pos_y], dtype = current_pos.dtype, device=current_pos.device)

            if verbose: print(f"Sampled pt: {samp_pos}")
            num_samps += 1
            # print(f"\t\t Positon sample time: {time.time() - st_start} s")
            # st_start = time.time()

            # If point is in obstacle, continue
            if not self._obstacle_free_check(world_idx, samp_pos):
                if verbose: print("Pt in obstacle")
                continue
            
            # print(f"\t\t Obstacle check time: {time.time() - st_start} s")
            # st_start = time.time()

            # Find nearest point in search tree & compute cost
            idx_nearest = self._find_nearest_node(V, samp_pos)
            candidate_pos = V[idx_nearest] * (1 - alpha[1]) + samp_pos * alpha[1]
            # Probably need to have an additional obstacle check here
            if not self._is_path_obstacle_free(V[idx_nearest], candidate_pos, world_idx, num_checks=4):
                continue
            cost_new = costs[idx_nearest] + torch.norm(candidate_pos - V[idx_nearest])
            parents[len(V)] = idx_nearest
            costs[len(V)] = cost_new

            # print(f"\t\t Nearest node time: {time.time() - st_start} s")
            # st_start = time.time()

            if rewire: # TODO Update rewire step if going to use this
                # Find neighbors & best neighbor
                idx_best, neighbors = self._find_neighbors(V, samp_pos, costs, rad)

                # Only link if path to best parent is obstacle-free
                if self._is_path_obstacle_free(V[idx_best], samp_pos, world_idx):
                    if verbose: print("Path is obstacle free")
                    parents[len(V)] = idx_best  # Assign best neighbor as samp_pos parent
                    costs[len(V)] = costs[idx_best] + torch.norm(samp_pos - V[idx_best])
                    # Rewire neighbors if cheaper to use new point and path is obstacle-free
                    for idx_n in neighbors:
                        # Use the cost from the root to the new node (costs[len(V)]) plus the cost from new node to neighbor
                        new_cost = costs[len(V)] + torch.norm(samp_pos - V[idx_n])
                        if new_cost < costs[idx_n]:
                            if self._is_path_obstacle_free(samp_pos, V[idx_n], world_idx):
                                parents[idx_n] = len(V)
                                costs[idx_n] = new_cost

            # Add sampled position to vertices
            V.append(candidate_pos)

        # print(f"\tSampling pts took {time.time() - t_start} s")
        # t_start = time.time()

        # Extract path: find leaf node with best heuristic value
        vals = []
        for i, v in enumerate(V):
            # Skip if vertex is a parent of another vertex (only eval leaf nodes)
            if i in parents.values():
                continue
            val = 0
            for i, fn in enumerate(heuristic_eval_fns):
                fn_out = heuristic_weights[world_idx][i] * fn(self.obs, world_idx, v)
                val += fn_out
                if verbose: print(f"Heuristic eval for {fn} at pt {v} with w {heuristic_weights[world_idx]}: {fn_out}")
            vals.append(val)
        # print("Vals:", vals)
        goal_idx = int(torch.argmin(torch.stack(vals)))
        if verbose: print(f"Goal idx:{goal_idx}")

        # Backtrack to root to get trajectory
        traj = []
        idx = goal_idx
        while idx is not None:
            traj.append(V[idx])
            idx = parents[idx]
        traj = traj[::-1]
        if verbose: print(f"Backtracked traj: {traj}")

        # print(f"\tHeuristic eval & backtrack path took {time.time() - t_start} s")
        
        return traj
    
    # Check for obstacles along the path to best parent
    def _is_path_obstacle_free(self, start, end, world_idx, num_checks=10):
        for alpha in torch.linspace(0, 1, steps=num_checks):
            interp = start * (1 - alpha) + end * alpha
            if not self._obstacle_free_check(world_idx, interp):
                # print("OBSTACLE IN PATH")
                return False
        return True

    def _obstacle_free_check(self, world_idx, pos, buffer=0.05, verbose=False):
        """
        Returns true if pos is not within buffer radius of the centerpoint
        of any obstacles in agent's observations
        """
        # print("Agent obstacles:\n", self.obs["obs_obstacles"])
        obstacles_pos = self.obs["obs_base"][world_idx] + self.obs["obs_obstacles"][world_idx]
        if verbose: print("world", world_idx, "obstacle locs:\n", obstacles_pos)

        clearances = torch.norm(obstacles_pos - pos, dim=1) >= buffer
        if verbose:  print("Clearances:", clearances)

        return clearances.all()
        
    
    def _random_pos(self, samp_rng_x, samp_rng_y, dtype, device):
        """
        Sample random position within specified ranges
        """
        x = torch.empty(1, dtype=dtype, device=device).uniform_(samp_rng_x[0], samp_rng_x[1]).item()
        y = torch.empty(1, dtype=dtype, device=device).uniform_(samp_rng_y[0], samp_rng_y[1]).item()
        return torch.tensor([x, y], dtype=dtype, device=device)

    
    def _find_nearest_node(self, vertices, pos):
        """
        Returns nearest node from vertices to pos
        """
        dists = [torch.norm(x - pos) for x in vertices]
        return int(torch.argmin(torch.stack(dists)))

    
    def _find_neighbors(self, vertices, samp_pos, costs, radius):
        """
        Get neighbors within search radius & best neighbor (nearest) within radius
        """
        dists = [torch.norm(x - samp_pos) for x in vertices]
        neighbors = [i for i, d in enumerate(dists) if d < radius]
        if neighbors:
            costs_neighbors = [costs[i] + torch.norm(samp_pos - vertices[i]) for i in neighbors]
            best_idx = neighbors[int(torch.argmin(torch.stack(costs_neighbors)))]
        else:
            best_idx = self._find_nearest_node(vertices, samp_pos)
        return best_idx, neighbors
    

    def get_control_action_cont(self, heuristic_weights, heuristic_eval_fns, horizon, verbose=False):
        """
        Sampling-based search within radius defined by horizon. Observations from environment should
        allow us to evaluate value of each sampled point towards each heuristic.
        """

        current_pos = self.state.pos
        if verbose: print(f"Agent {self.name} heuristic weights:\n {heuristic_weights}")

        # Make plans on init
        target_waypt = []
        if self.trajs == []:
            if verbose: print("Initializing trajectories...")
            self.traj_idx = []
            for i in range(heuristic_weights.shape[0]):
                self.trajs.append(self._compute_trajectory_cont(i, current_pos, heuristic_weights, heuristic_eval_fns, horizon, verbose=False))
                self.traj_idx.append(0) # start at waypoint 0 for each traj in batch
                target_waypt.append(self.trajs[i][0])

        # If arrived at target waypoint, update target to next waypoint
        target_waypt = torch.stack([self.trajs[i][idx] for i, idx in enumerate(self.traj_idx)])
        if verbose: print("Target waypoints:", target_waypt)

        # TODO: Vectorize me cap'n!
        for i, traj in enumerate(self.trajs):
            # target_waypt = traj[self.traj_idx[i]]
            # print("Current pos:", current_pos[i], "Target waypoint:", target_waypt[i])
            if torch.norm(current_pos[i] - target_waypt[i]) < 0.05:
                # Arrived at next waypt in traj
                # if i == 0: print(f"AGENT {self.name} NODE {target_waypt[i]} REACHED")
                old_idx = self.traj_idx[i]
                next_idx = min(self.traj_idx[i] + 1, len(traj)-1)
                self.traj_idx[i] = next_idx

                # If we are at end of trajectory, compute new trajectory
                if old_idx == next_idx:
                    if verbose and i == 0: print(f"AGENT {self.name} COMPLETED TRAJ. Located at {current_pos[i]}. Recomputing...")
                    traj = self._compute_trajectory_cont(i, current_pos, heuristic_weights, heuristic_eval_fns, horizon, verbose=False)
                    if verbose and i == 0: print("New traj:", traj)
                    self.trajs[i] = traj
                    self.traj_idx[i] = 0

                target_waypt[i] = traj[self.traj_idx[i]]

        # Get control action to reach next waypt
        pos_diff = target_waypt - current_pos
        if verbose: print("Pos diff:", pos_diff)
        # print("Pos diff:", pos_diff)
        
        u_action = torch.where(pos_diff > self.max_speed, self.max_speed, pos_diff)
        u_action = torch.where(u_action < -self.max_speed, -self.max_speed, u_action)

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