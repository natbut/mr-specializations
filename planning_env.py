import torch
import torchrl
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
            batch_size: int, 
            device: str = "cpu",
            node_dim = 5,
            **kwargs
            ):
        # TODO Check kwargs passing
        super().__init__(batch_size=[batch_size], device=device)

        # VMAS Environment Configuration
        # TODO need batch_size worlds?
        self.scenario = scenario
        # self.device = device
        # self.world = scenario.make_world(batch_dim=1, device=device)
        self.sim_env = make_env(self.scenario,
                            self.batch_size[0],
                            device=self.device,
                            )
        self.horizon = 20  # Number of steps to execute per trajectory
        self.node_dim = node_dim
        self.render = False
        self.count = 0
        self.graph_batch = None
        self.sim_obs = None
        print(f"Initialized environment...")

        # Define Observation & Action Specs
        self.observation_spec = Composite(
            x=Unbounded(
                shape=self.batch_size,
                device=device
                ),
            edge_index=Unbounded(
                shape=self.batch_size,
                device=device
            ),
            shape=self.batch_size,
            device=device
        )
        # print(f"\nObservation Spec:\n{self.observation_spec}")

        n_features = self.scenario.n_agents + self.scenario.n_tasks + self.scenario.n_obstacles
        self.action_spec = Bounded(
            low=torch.zeros((self.batch_size[0],
                             self.node_dim**2,
                             n_features
                             ),
                            device=device
                            ),
            high=torch.ones((self.batch_size[0],
                             self.node_dim**2,
                             n_features
                             ),
                            device=device
                            ),
            shape=(self.batch_size[0], self.node_dim**2, n_features),
            device=device
        )
        # print(f"\nAction (Weights) Spec:\n{self.action_spec}")

        self.reward_spec = Unbounded(
            shape=(self.batch_size[0], self.sim_env.n_agents),
            device=device
            )#, self.scenario.n_agents))
        # print(f"\nReward Spec:\n{self.reward_spec}")

    def _reset(self, obs_tensordict=None) -> TensorDict:
        """Reset all VMAS worlds and return initial state."""
        self.sim_obs = self.sim_env.reset()
        self._build_obs_graph(node_dim=self.node_dim)
        out = TensorDict(
            self.graph_obs,
            device=self.device,
            batch_size=self.batch_size
            )
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
        # print(f"Heuristic Weights: \n{heuristic_weights}")

        # Execute and aggregate rewards
        rewards = torch.zeros((self.batch_size[0], self.sim_env.n_agents), device=self.device)
        frame_list = []
        for t in range(self.horizon):
            # Update planning graph (env is dynamic)
            self._build_obs_graph(node_dim=self.node_dim)
            # Compute agent trajectories & get actions
            u_action = []
            for i, agent in enumerate(self.sim_env.agents):
                u_action.append(agent.get_control_action(self.graph_batch, heuristic_weights[i], self.horizon-t)) # get next actions from agent controllers
            # print("U-ACTION:", u_action)
            self.sim_obs, rews, dones, info = self.sim_env.step(u_action)
            # print("Rewards:", rewards, "rews:", torch.stack(rews).T)
            rewards += torch.stack(rews).T

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
        # next_state = TensorDict(
        #     {"observation": self._build_obs_graph()},
        #     device=self.device
        # )
        next_state = TensorDict(
            self.graph_obs,
            device=self.device,
            batch_size=self.batch_size
        )

        rew_tdict = TensorDict(
            {"reward": rewards},
            device=self.device,
            batch_size=self.batch_size
        )
        done_tdict = TensorDict(
            {"done": self.scenario.done()}, # TODO check done compute
            device=self.device,
            batch_size=self.batch_size
        )

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

        all_graphs = []
        x_vec = []
        edge_index_vec = []
        edge_attr_vec = []
        pos_vec = []
        for i in range(self.batch_size[0]):
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

            node_rad = torch.max((max_dims - min_dims)/node_dim)
            
            if verbose: print("Element Positions", element_positions, "Max/Min:", max_dims, min_dims, "Rad:", node_rad)

            # Create a grid of node centers
            node_positions = []
            node_indices = {}
            index = 0
            for x_idx in range(node_dim):
                for y_idx in range(node_dim):
                    node_pos = min_dims + node_rad*(torch.tensor((x_idx+0.5, y_idx+0.5), device=self.device))
                    node_positions.append(node_pos)
                    node_indices[(x_idx, y_idx)] = index
                    index += 1
            
            num_nodes = len(node_positions)
            node_positions = torch.stack(node_positions)
            
            if verbose: print(f"Env {i} Node positions: \n{node_positions} \nEnv {i} Node Indices: {node_indices}")

            # Compute binary feature vectors
            features = torch.zeros((num_nodes, len(element_positions)), device=self.device)  # [agent_presence, task_presence, obstacle_presence]
            
            for j, feature_pos in enumerate(element_positions):
                for k, node_pos in enumerate(node_positions):
                    # print("Feat pos:", feature_pos, " Node pos:", node_pos)
                    dists = torch.norm(feature_pos - node_pos, dim=0)
                    if torch.any(dists < node_rad):
                        # print(f"Feature {feature_pos} near node {k}: {node_pos} with dist: {dists}")
                        features[k, j] = 1 #torch.sum(dists < node_rad) # Normalize by num features? /len(feature_positions)

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

            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).T  # Shape [2, num_edges] (COO)
            edge_attrs = torch.tensor(edge_attrs, dtype=torch.float, device=self.device)  # Shape [num_edges]
            if verbose: print("Edge index:\n", edge_index)
            if verbose: print("Edge attributes (distances):\n", edge_attrs)

            # Create a graph for this environment
            graph = Data(x=features, edge_index=edge_index, edge_attr=edge_attrs, pos=node_positions).to(self.device)
            # x_vec.append(features)
            # edge_index_vec.append(edge_index)
            # edge_attr_vec.append(edge_attrs)
            # pos_vec.append(node_pos)

            all_graphs.append(graph)
        
        # Batch graphs together
        batched_graph = Batch.from_data_list(all_graphs)
        if verbose: print("Batched graph:", batched_graph)
        self.graph_batch = batched_graph # TODO maybe change where this is defined
        # print("X", batched_graph["x"])

        # print("BATCH X", batched_graph["x"])
        # batch_x = torch.stack(x_vec)
        # batch_edge_index = torch.stack(edge_index_vec)
        # batch_edge_attr = torch.stack(edge_attr_vec)
        # batch_pos = torch.stack(pos_vec)

        self.graph_obs = {
            "x": torch.stack([g["x"] for g in all_graphs]),
            "edge_index": torch.stack([g["edge_index"] for g in all_graphs]),
            # "edge_attr": torch.stack([g["edge_attr"] for g in all_graphs]),
            # "pos": torch.stack([g["pos"] for g in all_graphs]),
            # "batch": batched_graph.batch,
        }

        return batched_graph