import torch
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, Composite, Bounded
from torch_geometric.data import Data, Batch  # For Graph Representation
from tensordict.tensordict import TensorDict

from vmas.simulator.scenario import BaseScenario
from vmas import make_env
from typing import Tuple, Dict


class VMASPlanningEnv(EnvBase):
    def __init__(
            self, scenario: BaseScenario, 
            device: str = "cpu",
            env_kwargs: Dict = None,
            scenario_kwargs: Dict = None,
            ):
        
        self.scenario = scenario

        self.num_envs = env_kwargs.pop("num_envs", 1)
        self.horizon = env_kwargs.pop("horizon", 10)  # Number of steps to execute per trajectory 
        self.node_dim = env_kwargs.pop("node_dim", 4)
        self.connectivity = env_kwargs.pop("connectivity", 4)
        self.render = env_kwargs.pop("render", False)

        self.graph_batch = None
        self.sim_obs = None
        self.render_fp = None
        self.count = 0

        # TODO Check kwargs passing
        if self.num_envs == 1:
            super().__init__(batch_size=[], device=device)
            # VMAS Environment Configuration
            self.sim_env = make_env(
                                scenario=self.scenario,
                                num_envs=1,
                                max_steps=env_kwargs.pop("max_steps", 100),
                                device=self.device,
                                **scenario_kwargs,
                                )

            n_features = self.scenario.num_feats #self.scenario.n_agents + self.scenario.n_tasks + self.scenario.n_obstacles
            
            # Define Observation & Action Specs
            self.observation_spec = Composite(
                obs=Composite(
                    cell_feats=Unbounded(
                        shape=(self.node_dim**2, n_features),
                        dtype=torch.float64,
                        device=device
                        ),
                    cell_pos=Unbounded(
                        shape=(self.node_dim**2, 2),
                        dtype=torch.float64,
                        device=device
                    ),
                    rob_pos=Unbounded(
                        shape=(self.sim_env.n_agents, 2),
                        dtype=torch.float64,
                        device=device
                    ),
                    device=device
                ),
                shape=(1,),
                device=device
            )

            self.action_spec = Bounded(
                low=-torch.ones((
                                self.sim_env.n_agents,
                                n_features
                                ),
                                device=device
                                ),
                high=torch.ones((
                                self.sim_env.n_agents,
                                n_features
                                ),
                                device=device
                                ),
                shape=(self.sim_env.n_agents, n_features), # self.node_dim**2, 
                device=device
            )

            self.reward_spec = Unbounded(
                shape=(1,), #(self.sim_env.n_agents),
                device=device
                )#, self.scenario.n_agents))

        else:
            super().__init__(batch_size=[self.num_envs], device=device)
            # VMAS Environment Configuration
            self.sim_env = make_env(
                                scenario=self.scenario,
                                num_envs=self.batch_size[0],
                                max_steps=env_kwargs.pop("max_steps", 100),
                                device=self.device,
                                **scenario_kwargs,
                                )
        
            n_features = self.scenario.num_feats #self.scenario.n_agents + self.scenario.n_tasks + self.scenario.n_obstacles

            # Define Observation & Action Specs
            self.observation_spec = Composite(
                # obs=Composite(
                    cell_feats=Unbounded(
                        shape=(self.num_envs, self.node_dim**2, n_features),
                        dtype=torch.float64,
                        device=device
                        ),
                    cell_pos=Unbounded(
                        shape=(self.num_envs, self.node_dim**2, 2),
                        dtype=torch.float64,
                        device=device
                    ),
                    rob_pos=Unbounded(
                        shape=(self.num_envs, self.sim_env.n_agents, 2),
                        dtype=torch.float64,
                        device=device
                    ),
                #     device=device
                # ),
                shape=(self.num_envs,),
                device=device
            )

            self.action_spec = Bounded(
                low=-torch.ones((
                                self.num_envs,
                                self.sim_env.n_agents,
                                n_features
                                ),
                                device=device
                                ),
                high=torch.ones((
                                self.num_envs,
                                self.sim_env.n_agents,
                                n_features
                                ),
                                device=device
                                ),
                shape=(self.num_envs, self.sim_env.n_agents, n_features),
                device=device
            )

            self.reward_spec = Unbounded(
                shape=(self.num_envs, 1),
                device=device
                )

    def _reset(self, obs_tensordict=None) -> TensorDict:
        """Reset all VMAS worlds and return initial state."""
        self.sim_obs = self.sim_env.reset()
        self._build_obs_graph(node_dim=self.node_dim)
        obs = TensorDict(self.graph_obs,
                         batch_size=self.batch_size,
                         device=self.device)
        # out.set("step_count", torch.full(self.batch_size, 1))
        # print("Reset TDict:", obs)
        # print("Expanded:", [(x_i, edges_i) for x_i, edges_i in zip(self.graph_obs["x"], self.graph_obs["edge_index"])])
        return obs

    def _step(self, actions: TensorDict) -> TensorDict:
        """
        Steps through the environment (IN PARALLEL...sort of):
        1. Use heuristic weights to plan agent trajectories.
        2. Execute the trajectories for the given horizon.
        3. Return next state, rewards, and termination status.
        """

        # Process actions (heuristic weights)
        # heuristic_weights = actions.view(self.batch_size, self.scenario.n_agents, self.scenario.n_tasks)
        heuristic_weights = actions["action"] # NOTE THESE ARE WEIGHTS FOR EVERY NODE, BUT WE ONLY USE AGENT LOCS
        if self.render:
            print(f"\nHeuristic Weights:\n {heuristic_weights} Shape: {heuristic_weights.shape}") # [B, N_AGENTS, N_FEATS]
        # print("WEIGHTS:", heuristic_weights, "shape:", heuristic_weights.shape)

        # Execute and aggregate rewards
        rewards = torch.zeros((self.num_envs, 1), device=self.device)
        # dones = torch.full((self.num_envs, 1), False, device=self.device)
        # null_rews = torch.zeros_like(rewards, device=self.device)
        frame_list = []
        # Update planning graph (assuming env is dynamic)
        self._build_obs_graph(node_dim=self.node_dim) # TODO for more dynamic envs, update this more frequently (in below loop)
        for agent in self.sim_env.agents: # Reset agent trajectories
            agent.trajs = []
        # print("\n= Pre-rollout step! =")
        for t in range(self.horizon):
            verbose = False
            # if t == 0: verbose = True
            # Compute agent trajectories & get actions
            u_action = []
            for i, agent in enumerate(self.sim_env.agents):
                u_action.append(agent.get_control_action(self.graph_batch, heuristic_weights[:,i,:], self.horizon-t, verbose)) # get next actions from agent controllers
            # print("U-ACTION:", u_action)
            self.sim_obs, rews, dones, info = self.sim_env.step(u_action)
            # print("Rewards:", rewards, "stacked rews:", rews, "sum:", rews.sum(dim=0))

            # NOTE Ignores additional rewards from envs that reset during rollout (from completing early)
                # Hypothesis here is that envs reset at done indices and continue to accumulate negative rewards
                # Ideally, we won't even step completed envs, but this is a quick fix for now
            # done = done.unsqueeze(1)
            # dones = torch.where(done == True, done, dones)
            team_rews = torch.stack(rews).sum(dim=0)
            rewards += team_rews #torch.where(dones == False, team_rews, null_rews)

            if self.render:
                frame = self.sim_env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                )
                frame_list.append(frame)

            if dones.all():
                break 
            
        if self.render:
            from moviepy import ImageSequenceClip
            fps = 20
            clip = ImageSequenceClip(frame_list, fps=fps)
            clip.write_gif(f"{self.render_fp}_{self.count}.gif", fps=fps)
            self.count += 1


        # print("Rewards:", rewards, "\nDones:", dones.unsqueeze(1))
        # rewards = rewards / (self.horizon*(self.sim_env.n_agents*0.9)) # NOTE ADDED NORMALIZATION FACTOR TO MAX PER-STEP REWARD

        self._build_obs_graph(node_dim=self.node_dim)
        # Construct next state representation   
        next_state = TensorDict(
            self.graph_obs,
            device=self.device,
        )
        rew_tdict = TensorDict(
            {"reward": rewards},
            device=self.device,
        )
        done_tdict = TensorDict(
            {"done": dones.unsqueeze(1)},
            device=self.device,
        )

        # next_state.set("step_count", 1)

        # Should return next_state, rewards, done as tensordict
        next_state.update(rew_tdict).update(done_tdict) 
        # print("Next TDict:\n", next_state)
        
        # print("= Post-rollout step! = ")

        return next_state #, rewards, done, {}

    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)


    def _build_obs_graph(self,
                         node_dim=2,
                         connectivity=4,
                         verbose=False
                         ):
        """
        Discretize type & pos based observations into a planning graph.
        Obs should be dict with "env_dims": (x, y), "NAME": (x,y), ...
        """
        global_obs_dict = self.sim_obs[0] # each entry is num_envs x entry_size
        if verbose: print(f"BuildGraph Obs:\n {global_obs_dict}") # NOTE only need 1 obs if global

        all_graphs = []
        for i in range(self.num_envs):
            # Dynamically compute node_rad to allow fixed graph topology for varying problem sizes
            element_positions = []
            for key in global_obs_dict:
                if "obs" in key:
                    for entity in global_obs_dict[key]:
                        element_positions.append(entity[i])
            # print([global_obs_dict[key] for key in global_obs_dict if "obs" in key])
            # element_positions = torch.cat([el[i] for el in [global_obs_dict[key] for key in global_obs_dict if "obs" in key]])
            # print("el pos", torch.stack(element_positions))
            element_positions = torch.stack(element_positions)
            min_dims = torch.min(element_positions, dim=0).values
            max_dims = torch.max(element_positions, dim=0).values

            cell_dim = 1.001*((max_dims - min_dims)/node_dim) #torch.max
            
            if verbose: print("Element Positions", element_positions, "Max/Min:", max_dims, min_dims, "Cell Dim:", cell_dim)

            # Create a grid of node centers
            node_positions = []
            node_indices = {}
            index = 0
            for x_idx in range(node_dim):
                for y_idx in range(node_dim):
                    node_pos = min_dims + cell_dim*(torch.tensor((x_idx+0.5, y_idx+0.5), device=self.device))
                    node_positions.append(node_pos)
                    node_indices[(x_idx, y_idx)] = index
                    index += 1
            
            num_nodes = len(node_positions)
            node_positions = torch.stack(node_positions)
            
            if verbose: print(f"Env Node positions: \n{node_positions} \nEnv Node Indices: {node_indices}")

            # Compute binary feature vectors using square-shaped cells
            features = torch.full((num_nodes, len(element_positions)),
                                  0.0001,
                                  dtype=torch.float32,
                                  device=self.device
                                  )
            
            for j, feature_pos in enumerate(element_positions):
                for k, node_pos in enumerate(node_positions):
                    # Check if the feature is within the square cell centered at the node
                    within_x = torch.abs(feature_pos[0] - node_pos[0]) <= cell_dim[0] / 2
                    within_y = torch.abs(feature_pos[1] - node_pos[1]) <= cell_dim[1] / 2
                    if within_x and within_y:
                        if verbose:
                            print(f"Feature {feature_pos} within square cell of node {k}: {node_pos}")
                        features[k, j] = 1

            if verbose: print("Node features:\n", features)

            # Compute edge adjacency list & attributes (4-way or 8-way connectivity)
            edge_list = []
            edge_attrs = []  # To store distances between connected nodes
            for (x_idx, y_idx), node_id in node_indices.items():
                neighbors = []
                if connectivity == 4:
                    neighbors = [(x_idx-1, y_idx), (x_idx+1, y_idx), (x_idx, y_idx-1), (x_idx, y_idx+1)]
                elif connectivity == 8:
                    neighbors = [(x_idx-1, y_idx-1), (x_idx-1, y_idx), (x_idx-1, y_idx+1),
                                (x_idx, y_idx-1), (x_idx, y_idx+1),
                                (x_idx+1, y_idx-1), (x_idx+1, y_idx), (x_idx+1, y_idx+1)]
                
                for neighbor in neighbors:
                    if neighbor in node_indices:
                        neighbor_id = node_indices[neighbor]
                        edge_list.append((node_id, neighbor_id))
                        # Compute distance between nodes
                        dist = torch.norm(node_positions[node_id] - node_positions[neighbor_id])
                        edge_attrs.append(dist)

            edge_index = torch.tensor(edge_list, dtype=torch.int64, device=self.device).T  # Shape [2, num_edges] (COO)
            edge_attrs = torch.tensor(edge_attrs, device=self.device)  # Shape [num_edges]
            if verbose: print("Edge index:\n", edge_index)
            if verbose: print("Edge attributes (distances):\n", edge_attrs)

            # Create a graph for this environment
            graph = Data(x=features, edge_index=edge_index, edge_attr=edge_attrs, pos=node_positions).to(self.device)
            # x_vec.append(features)
            # edge_index_vec.append(edge_index)
            # edge_attr_vec.append(edge_attrs)
            # pos_vec.append(node_pos)

            # print("\n!!! ROB POS:", self.sim_obs[0]["obs_agents"].squeeze(1), "Shape:", self.sim_obs[0]["obs_agents"].squeeze(1).shape)
            all_graphs.append(graph)

        self.graph_batch = Batch.from_data_list(all_graphs)
        reshaped_rob_pos = self.sim_obs[0]["obs_agents"].squeeze(1).reshape(self.num_envs, self.sim_env.n_agents, 2)

        self.graph_obs = {
            "cell_feats": torch.stack([g["x"] for g in all_graphs]),
            "cell_pos": torch.stack([g["pos"] for g in all_graphs]),
            "rob_pos": reshaped_rob_pos, # TODO Sequeeze needed for vectorized env?
            # "edge_attr": torch.stack([g["edge_attr"] for g in all_graphs]),
            # "pos": torch.stack([g["pos"] for g in all_graphs]),
            # "batch": batched_graph.batch,
        }

        if verbose: print("Graph Obs:", self.graph_obs)