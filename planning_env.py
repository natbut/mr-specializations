import torch
import torchrl
from torchrl.envs import EnvBase
from torch_geometric.data import Data  # For Graph Representation

from vmas.simulator import World
from vmas.simulator.scenario import BaseScenario
from typing import Tuple, Dict


class VMASPlanningEnv(EnvBase):
    def __init__(self, scenario: BaseScenario, batch_size: int, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)

        # VMAS Environment Configuration
        self.scenario = scenario
        self.device = torch.device(device)
        self.world = scenario.make_world(batch_dim=batch_size, device=self.device)
        self.batch_size = batch_size
        self.horizon = 20  # Number of steps to execute per trajectory

        # Define Observation & Action Specs
        self.observation_spec = torchrl.envs.CompositeSpec(
            {"graph": torchrl.envs.UnboundedContinuousTensorSpec((self.batch_size,))}
        )
        self.action_spec = torchrl.envs.UnboundedContinuousTensorSpec(
            (self.batch_size, self.scenario.n_agents, self.scenario.n_tasks)
        )
        self.reward_spec = torchrl.envs.UnboundedContinuousTensorSpec((self.batch_size,))

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the VMAS world and return initial state."""
        self.scenario.reset_world_at()
        return {"graph": self._build_graph_representation()}

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict]:
        """
        Steps through the environment:
        1. Use heuristic weights to plan agent trajectories.
        2. Execute the trajectories for the given horizon.
        3. Return next state, rewards, and termination status.
        """
        # Process actions (heuristic weights)
        heuristic_weights = actions.view(self.batch_size, self.scenario.n_agents, self.scenario.n_tasks)

        # Compute trajectories based on heuristic weights
        self._compute_trajectories(heuristic_weights)

        # Execute and aggregate rewards
        rewards = torch.zeros(self.batch_size, device=self.device)
        for _ in range(self.horizon):
            self.world.step()
            rewards += self._compute_rewards()

        # Construct next state representation
        next_state = {"graph": self._build_graph_representation()}
        done = self.scenario.done()

        return next_state, rewards, done, {}

    def _build_graph_representation(self) -> Data:
        """Convert VMAS state into a graph representation for policy input."""
        agents = self.world.agents
        tasks = self.scenario.tasks
        obstacles = self.scenario.obstacles

        # Collect node features
        agent_features = torch.stack([a.state.pos for a in agents], dim=0)  # Agent positions
        task_features = torch.stack([t.state.pos for t in tasks], dim=0)  # Task positions
        obstacle_features = torch.stack([o.state.pos for o in obstacles], dim=0)  # Obstacle positions

        # Create graph data structure
        nodes = torch.cat([agent_features, task_features, obstacle_features], dim=0)
        edges = self._compute_graph_edges(nodes)
        return Data(x=nodes, edge_index=edges)

    def _compute_graph_edges(self, nodes: torch.Tensor) -> torch.Tensor:
        """Compute edges based on spatial proximity."""
        dist_matrix = torch.cdist(nodes, nodes)  # Compute pairwise distances
        edge_index = (dist_matrix < 0.5).nonzero(as_tuple=True)  # Threshold-based connectivity
        return torch.stack(edge_index, dim=0)

    def _compute_trajectories(self, heuristic_weights: torch.Tensor):
        """Plan agent trajectories based on heuristic weights."""
        for agent, weights in zip(self.world.agents, heuristic_weights):
            agent.policy_weights = weights  # Assign heuristic weights
            agent.plan_trajectory()  # Custom trajectory planner

    def _compute_rewards(self) -> torch.Tensor:
        """Compute reward based on task completion."""
        return torch.stack([self.scenario.reward(agent) for agent in self.world.agents], dim=0).sum(dim=0)


# Example Usage:
if __name__ == "__main__":
    scenario = Scenario()
    env = VMASPlanningEnv(scenario, batch_size=4, device="cuda")
    
    state = env.reset()
    actions = torch.randn(env.batch_size, scenario.n_agents, scenario.n_tasks)  # Random actions
    next_state, rewards, done, _ = env.step(actions)
    
    print("Next State:", next_state)
    print("Rewards:", rewards)
    print("Done:", done)
