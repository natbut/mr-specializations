import torch
import torchrl
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, Composite, Bounded
from torch_geometric.data import Data, Batch  # For Graph Representation
from tensordict.tensordict import TensorDict

from vmas.simulator.scenario import BaseScenario
from vmas import make_env
from typing import Tuple, Dict

# def make_composite_from_td(td):
#     # custom function to convert a ``tensordict`` in a similar spec structure
#     # of unbounded values.
#     composite = Composite(
#         {
#             key: make_composite_from_td(tensor)
#             if isinstance(tensor, TensorDictBase)
#             else Unbounded(dtype=tensor.dtype, device=tensor.device, shape=tensor.shape)
#             for key, tensor in td.items()
#         },
#         shape=td.shape,
#     )
#     return composite

# def _make_spec(self, td_params):
#     # Under the hood, this will populate self.output_spec["observation"]
#     self.observation_spec = Composite(
#         x=Unbounded(
#             shape=(),
#             ),
#         edge_index=Unbounded(
#             shape=(),
#         ),
#         params=make_composite_from_td(td_params["params"]),
#         shape=(),
#     )

#     # since the environment is stateless, we expect the previous output as input.
#     # For this, ``EnvBase`` expects some state_spec to be available
#     self.state_spec = self.observation_spec.clone()
#     # action-spec will be automatically wrapped in input_spec when
#     # `self.action_spec = spec` will be called supported
#     self.action_spec = Bounded(
#         low=-torch.ones((
#                             self.node_dim**2,
#                             n_features
#                             ),
#                         device=device
#                         ),
#         high=torch.ones((
#                             self.node_dim**2,
#                             n_features
#                             ),
#                         device=device
#                         ),
#         shape=(n_agents, n_features),
#     )

#     self.reward_spec = Unbounded(shape=(*td_params.shape, 1))


# def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
#     """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
#     if batch_size is None:
#         batch_size = []
#     td = TensorDict(
#         {
#             "params": TensorDict(
#                 {
#                     "max_speed": 8,
#                     "max_torque": 2.0,
#                     "dt": 0.05,
#                     "g": g,
#                     "m": 1.0,
#                     "l": 1.0,
#                 },
#                 [],
#             )
#         },
#         [],
#     )
#     if batch_size:
#         td = td.expand(batch_size).contiguous()
#     return td



class VMASPlanningEnv(EnvBase):
    def __init__(
            self, scenario: BaseScenario, 
            num_envs: int, 
            device: str = "cpu",
            node_dim = 5,
            **kwargs
            ):
        
        self.scenario = scenario
        # TODO Check kwargs passing
        if num_envs is None:
            super().__init__(batch_size=[], device=device)
            # VMAS Environment Configuration
            self.sim_env = make_env(scenario=self.scenario,
                                num_envs=1,
                                device=self.device,
                                )
        else:
            super().__init__(batch_size=[num_envs], device=device)
            # VMAS Environment Configuration
            self.sim_env = make_env(scenario=self.scenario,
                                num_envs=self.batch_size[0],
                                device=self.device,
                                )

        
        self.horizon = 20  # Number of steps to execute per trajectory 
        self.node_dim = node_dim
        self.render = False
        self.count = 0
        self.graph_batch = None
        self.sim_obs = None
        connectivity=4

        n_features = self.scenario.n_agents + self.scenario.n_tasks + self.scenario.n_obstacles

        # Define Observation & Action Specs
        self.observation_spec = Composite(
            x=Unbounded(
                shape=(node_dim**2 * n_features),
                dtype=torch.float32,
                device=device
                ),
            edge_index=Unbounded(
                shape=(2, (node_dim**2)*connectivity-node_dim**2),
                dtype=torch.int64,
                device=device
            ),
            shape=(),
            device=device
        )
        # print(f"\nObservation Spec:\n{self.observation_spec}")

        self.action_spec = Bounded(
            low=-torch.ones((
                            #  self.node_dim**2,
                             n_features
                             ),
                            device=device
                            ),
            high=torch.ones((
                            #  self.node_dim**2,
                             n_features
                             ),
                            device=device
                            ),
            shape=(n_features), # self.node_dim**2, 
            device=device
        )
        # print(f"\nAction (Weights) Spec:\n{self.action_spec}")

        self.reward_spec = Unbounded(
            shape=(self.sim_env.n_agents),
            device=device
            )#, self.scenario.n_agents))
        # print(f"\nReward Spec:\n{self.reward_spec}")

        self.heuristic_softmax = torch.nn.Softmax(dim=-1)

    def _reset(self, obs_tensordict=None) -> TensorDict:
        """Reset all VMAS worlds and return initial state."""
        self.sim_obs = self.sim_env.reset()
        self._build_obs_graph(node_dim=self.node_dim)
        out = TensorDict(
            self.graph_obs,
            device=self.device,
            batch_size=self.batch_size
            )
        out.set("step_count", torch.full(self.batch_size, 1))
        # print("Reset TDict:", out)
        # print("Expanded:", [(x_i, edges_i) for x_i, edges_i in zip(self.graph_obs["x"], self.graph_obs["edge_index"])])
        return out

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
            print(f"\nHeuristic Weights: {heuristic_weights}")
        # heuristic_weights = self.heuristic_softmax(heuristic_weights) # Normalize weights to sum to 1 

        # Execute and aggregate rewards
        rewards = torch.zeros((self.sim_env.n_agents), device=self.device)
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
                u_action.append(agent.get_control_action(self.graph_batch, heuristic_weights, self.horizon-t, verbose)) # get next actions from agent controllers
            # print("U-ACTION:", u_action)
            self.sim_obs, rews, dones, info = self.sim_env.step(u_action)
            # print("Rewards:", rewards, "rews:", rews)
            rewards += rews[0]

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
            clip.write_gif(f"img/rollout_{self.count}.gif", fps=fps)
            self.count += 1

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
            {"done": self.scenario.done()}, # TODO check done compute
            device=self.device,
        )

        next_state.set("step_count", 1)

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
                    element_positions.append(entity)
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
        features = torch.zeros((num_nodes, len(element_positions)), dtype=torch.float32, device=self.device)  # [agent_presence, task_presence, obstacle_presence]
        
        for j, feature_pos in enumerate(element_positions):
            for k, node_pos in enumerate(node_positions):
                # Check if the feature is within the square cell centered at the node
                within_x = torch.abs(feature_pos[0][0] - node_pos[0][0]) <= cell_dim[0][0] / 2
                within_y = torch.abs(feature_pos[0][1] - node_pos[0][1]) <= cell_dim[0][1] / 2
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

        self.graph_batch = Batch.from_data_list([graph])
        self.graph_obs = {
            "x": graph["x"].reshape(-1),
            "edge_index": graph["edge_index"],
            # "edge_attr": torch.stack([g["edge_attr"] for g in all_graphs]),
            # "pos": torch.stack([g["pos"] for g in all_graphs]),
            # "batch": batched_graph.batch,
        }

        return graph