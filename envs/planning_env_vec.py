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

        num_envs = env_kwargs.pop("num_envs", 1)
        self.horizon = env_kwargs.pop("horizon", 10)  # Number of steps to execute per trajectory 
        self.node_dim = env_kwargs.pop("node_dim", 4)
        self.connectivity = env_kwargs.pop("connectivity", 4)
        self.render = env_kwargs.pop("render", False)

        # TODO Check kwargs passing
        if num_envs == 1:
            super().__init__(batch_size=[], device=device)
            # VMAS Environment Configuration
            self.sim_env = make_env(
                                scenario=self.scenario,
                                num_envs=1,
                                max_steps=env_kwargs.pop("max_steps", 100),
                                device=self.device,
                                **scenario_kwargs,
                                )
        else:
            super().__init__(batch_size=[num_envs], device=device)
            # VMAS Environment Configuration
            self.sim_env = make_env(
                                scenario=self.scenario,
                                num_envs=self.batch_size[0],
                                max_steps=env_kwargs.pop("max_steps", 100),
                                device=self.device,
                                **scenario_kwargs,
                                )

        self.graph_batch = None
        self.sim_obs = None
        self.render_fp = None
        self.count = 0

        n_features = self.scenario.n_agents + self.scenario.n_tasks + self.scenario.n_obstacles

        # Define Observation & Action Specs
        self.observation_spec = Composite(
            obs=Composite(
                cell_feats=Unbounded(
                    shape=(num_envs, self.node_dim**2, n_features),
                    dtype=torch.float64,
                    device=device
                    ),
                cell_pos=Unbounded(
                    shape=(num_envs, self.node_dim**2, 2),
                    dtype=torch.float64,
                    device=device
                ),
                rob_pos=Unbounded(
                    shape=(num_envs, self.sim_env.n_agents, 2),
                    dtype=torch.float64,
                    device=device
                ),
                device=device
            ),
            shape=(num_envs,),
            device=device
        )
        # print(f"\nObservation Spec:\n{self.observation_spec}")

        self.action_spec = Bounded(
            low=-torch.ones((
                            num_envs,
                            self.sim_env.n_agents,
                            n_features
                            ),
                            device=device
                            ),
            high=torch.ones((
                            num_envs,
                            self.sim_env.n_agents,
                            n_features
                            ),
                            device=device
                            ),
            shape=(num_envs, self.sim_env.n_agents, n_features), # self.node_dim**2, 
            device=device
        )
        # print(f"\nAction (Weights) Spec:\n{self.action_spec}")

        self.reward_spec = Unbounded(
            shape=(num_envs, 1), #(self.sim_env.n_agents),
            device=device
            )#, self.scenario.n_agents))
        # print(f"\nReward Spec:\n{self.reward_spec}")

    def _reset(self, obs_tensordict=None) -> TensorDict:
        """Reset all VMAS worlds and return initial state."""
        self.sim_obs = self.sim_env.reset()
        self._build_obs_graph(node_dim=self.node_dim)
        obs = TensorDict({"obs": self.graph_obs},
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
            print(f"\nHeuristic Weights:\n {heuristic_weights} Shape: {heuristic_weights.shape}")

        # Execute and aggregate rewards
        rewards = torch.zeros((1,), device=self.device)
        frame_list = []
        # Update planning graph (assuming env is dynamic)
        self._build_obs_graph(node_dim=self.node_dim) # TODO for more dynamic envs, update this more frequently (in below loop)
        for agent in self.sim_env.agents: # Reset agent trajectories
            agent.trajs = []
        for t in range(self.horizon):
            verbose = False
            # if t == 0: verbose = True
            # Compute agent trajectories & get actions
            u_action = []
            for i, agent in enumerate(self.sim_env.agents):
                u_action.append(agent.get_control_action(self.graph_batch, heuristic_weights[i], self.horizon-t, verbose)) # get next actions from agent controllers
            # print("U-ACTION:", u_action)
            self.sim_obs, rews, dones, info = self.sim_env.step(u_action)
            # print("Rewards:", rewards, "rews:", rews, "sum:", torch.sum(torch.stack(rews)))
            rewards += torch.sum(torch.stack(rews))

            if self.render:
                frame = self.sim_env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                )
                frame_list.append(frame)
            
        if self.render:
            from moviepy import ImageSequenceClip
            fps = 20
            clip = ImageSequenceClip(frame_list, fps=fps)
            clip.write_gif(f"{self.render_fp}_{self.count}.gif", fps=fps)
            self.count += 1

        # Construct next state representation   
        next_state = TensorDict(
            {"obs": self.graph_obs},
            device=self.device,
        )
        rew_tdict = TensorDict(
            {"reward": rewards},
            device=self.device,
        )
        done_tdict = TensorDict(
            {"done": self.scenario.done()}, # TODO check done compute
            device=self.device,
        )

        # next_state.set("step_count", 1)

        # Should return next_state, rewards, done as tensordict
        next_state.update(rew_tdict).update(done_tdict) 
        # print("Next TDict:\n", next_state)
        return next_state #, rewards, done, {}

    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)


    def _build_obs_graph(self,
                         node_dim=2,
                         connectivity=4,
                         verbose=False
                         ) -> Batch:
        """
        Discretize type & pos based observations into a planning graph.
        Obs should be dict with "env_dims": (x, y), "NAME": (x,y), ...
        """
        global_obs_dict = self.sim_obs[0] # each entry is num_envs x entry_size
        if verbose: print(f"BuildGraph Obs:\n {global_obs_dict}") # NOTE only need 1 obs if global

        # Dynamically compute node_rad to allow fixed graph topology for varying problem sizes
        element_positions = []
        for key in global_obs_dict:
            if "obs" in key:
                for entity in global_obs_dict[key]:
                    element_positions.append(entity.squeeze(0))
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
                node_positions.append(node_pos.squeeze(0))
                node_indices[(x_idx, y_idx)] = index
                index += 1
        
        num_nodes = len(node_positions)
        node_positions = torch.stack(node_positions)
        
        if verbose: print(f"Env Node positions: \n{node_positions} \nEnv Node Indices: {node_indices}")

        # Compute binary feature vectors using square-shaped cells
        features = torch.zeros((num_nodes, len(element_positions)), dtype=torch.float32, device=self.device)  # [agent_presence, task_presence, obstacle_presence]
        
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

        self.graph_batch = Batch.from_data_list([graph])
        self.graph_obs = {
            "cell_feats": graph["x"],
            "cell_pos": graph["pos"],
            "rob_pos": self.sim_obs[0]["obs_agents"].squeeze(1),
            # "edge_attr": torch.stack([g["edge_attr"] for g in all_graphs]),
            # "pos": torch.stack([g["pos"] for g in all_graphs]),
            # "batch": batched_graph.batch,
        }

        return graph