import torch
from torch import Tensor
from typing import Dict

# File for storing heuristic evaluation functions
def min_dist(points_a: Tensor, points_b: Tensor) -> Tensor:
    """
    Calculates the minimum distance from each point in `points_b` to any point in `points_a`.

    Args:
        points_a (Tensor): Reference points, shape `[N_ref, 2]`.
        points_b (Tensor): Sampled points, shape `[N_sampled_points, 2]`.

    Returns:
        Tensor: Minimum distances for each sampled point, shape `[N_sampled_points]`.
                Returns `inf` if `points_a` is empty.
    """
    if points_a.numel() == 0:
        # If no reference points, return a very large distance for all sampled points
        # to discourage paths towards non-existent targets.
        return torch.full((points_b.shape[0],), float('inf'), device=points_b.device, dtype=points_b.dtype)
    
    # Calculate pairwise distances: [N_sampled_points, N_ref]
    distances = torch.cdist(points_b, points_a)
    
    # Take the minimum distance for each sampled point to any reference point
    min_distances = torch.min(distances, dim=-1).values # Result shape: [N_sampled_points]
    
    return min_distances

def nearest_task(agent_obs: Dict[str, Tensor], world_idx: int, sampled_pos: Tensor) -> Tensor:
    """
    Given set of candidate tasks positions and sampled points, returns the distance
    to the nearest task for each sampled point.

    Args:
        agent_obs (Dict[str, Tensor]): Agent's local observations from the scenario.
                                       Must include "obs_tasks".
        world_idx (int): The index of the current environment (batch).
        sampled_pos (Tensor): Points from the RRT tree, shape `[N_sampled_points, 2]`.

    Returns:
        Tensor: Distances to the nearest task for each sampled point, shape `[N_sampled_points]`.
    """
    if "obs_tasks" not in agent_obs:
        print("'obs_tasks' not included in agent local observations")
        return torch.full((sampled_pos.shape[0],), float('inf'), device=sampled_pos.device, dtype=sampled_pos.dtype)

    tasks = agent_obs["obs_tasks"][world_idx]  # Gets tasks for corresponding world, shape [N_tasks, 2]
    
    # Filter out tasks that are outside typical world bounds (e.g., stored tasks)
    # Assuming tasks are within [-1, 1] range normally.
    mask = torch.any(tasks <= 1.0, dim=-1) & torch.any(tasks >= -1.0, dim=-1) # Ensure within -1 to 1 range
    tasks = tasks[mask]
    
    return min_dist(tasks, sampled_pos)

def nearest_agent(agent_obs: Dict[str, Tensor], world_idx: int, sampled_pos: Tensor) -> Tensor:
    """
    Given set of other agents' positions and sampled points, returns the distance
    to the nearest other agent for each sampled point.

    Args:
        agent_obs (Dict[str, Tensor]): Agent's local observations from the scenario.
                                       Must include "obs_agents".
        world_idx (int): The index of the current environment (batch).
        sampled_pos (Tensor): Points from the RRT tree, shape `[N_sampled_points, 2]`.

    Returns:
        Tensor: Distances to the nearest other agent for each sampled point, shape `[N_sampled_points]`.
    """
    if "obs_agents" not in agent_obs:
        print("'obs_agents' not included in agent local observations")
        return torch.full((sampled_pos.shape[0],), float('inf'), device=sampled_pos.device, dtype=sampled_pos.dtype)

    agents = agent_obs["obs_agents"][world_idx] # Gets other agents for corresponding world, shape [N_other_agents, 2]
    
    return min_dist(agents, sampled_pos)

def nearest_frontier(agent_obs: Dict[str, Tensor], world_idx: int, sampled_pos: Tensor) -> Tensor:
    """
    Given set of candidate frontiers positions and sampled points, returns the distance
    to the nearest frontier for each sampled point.

    Args:
        agent_obs (Dict[str, Tensor]): Agent's local observations from the scenario.
                                       Must include "obs_frontiers".
        world_idx (int): The index of the current environment (batch).
        sampled_pos (Tensor): Points from the RRT tree, shape `[N_sampled_points, 2]`.

    Returns:
        Tensor: Distances to the nearest frontier for each sampled point, shape `[N_sampled_points]`.
    """
    if "obs_frontiers" not in agent_obs:
        # print("'obs_frontiers' not included in agent local observations")
        return torch.full((sampled_pos.shape[0],), float('inf'), device=sampled_pos.device, dtype=sampled_pos.dtype)
    
    # obs_frontiers is [B, MaxNumFrontiers, 4, 2]
    # world_idx selects the specific batch: [MaxNumFrontiers, 4, 2]
    frontiers_for_world = agent_obs["obs_frontiers"][world_idx] 
    
    # Reshape to [MaxNumFrontiers * 4, 2]
    frontiers_reshaped = frontiers_for_world.view(-1, 2)

    # Filter out padding values (100.0, 100.0) that indicate non-existent or already-explored neighbors
    is_padding = torch.isclose(frontiers_reshaped, torch.tensor([100.0, 100.0], device=frontiers_reshaped.device, dtype=frontiers_reshaped.dtype), atol=1e-6).all(dim=-1)
    
    actual_frontiers = frontiers_reshaped[~is_padding] # [N_actual_frontiers, 2]

    # If no actual frontiers, return a large distance to penalize this heuristic
    if actual_frontiers.numel() == 0:
        return torch.full((sampled_pos.shape[0],), float('inf'), device=sampled_pos.device, dtype=sampled_pos.dtype)

    return min_dist(actual_frontiers, sampled_pos)

def nearest_comms_midpt(agent_obs: Dict[str, Tensor], world_idx: int, sampled_pos: Tensor) -> Tensor:
    """
    Given positions of all agents and the base, computes communication midpoints
    and returns the distance to the nearest one for each sampled point.

    Args:
        agent_obs (Dict[str, Tensor]): Agent's local observations from the scenario.
                                       Must include "obs_agents", "obs_base", and "pos".
        world_idx (int): The index of the current environment (batch).
        sampled_pos (Tensor): Points from the RRT tree, shape `[N_sampled_points, 2]`.

    Returns:
        Tensor: Distances to the nearest communication midpoint for each sampled point,
                shape `[N_sampled_points]`.
    """
    if "obs_agents" not in agent_obs or "obs_base" not in agent_obs or "pos" not in agent_obs:
        print("'obs_agents', 'obs_base', or 'pos' not included in agent local observations")
        return torch.full((sampled_pos.shape[0],), float('inf'), device=sampled_pos.device, dtype=sampled_pos.dtype)
    
    current_agent_pos = agent_obs["pos"][world_idx].unsqueeze(0) # [1, 2] for the current agent in this batch
    other_agents = agent_obs["obs_agents"][world_idx] # [N_other_agents, 2] for other agents in this batch
    base = agent_obs["obs_base"][world_idx] # [2] for the base in this batch

    # Combine current agent's position with other agents' positions
    all_agents = torch.cat((current_agent_pos, other_agents), dim=0) # [N_total_agents, 2]

    comms_midpoints = []

    # Midpoints between each agent and the base
    # (all_agents: [N_total_agents, 2] + base: [2]) broadcasts base correctly
    agent_base_midpoints = (all_agents + base) / 2.0 # [N_total_agents, 2]
    comms_midpoints.append(agent_base_midpoints)

    # Midpoints between all unique pairs of agents
    num_agents = all_agents.shape[0]
    if num_agents > 1:
        # Generate indices for unique pairs using triu_indices
        row_indices, col_indices = torch.triu_indices(num_agents, num_agents, offset=1)
        
        # Select agents based on these indices and compute midpoints
        agent_agent_midpoints = (all_agents[row_indices] + all_agents[col_indices]) / 2.0 # [Num_unique_pairs, 2]
        comms_midpoints.append(agent_agent_midpoints)
    
    if not comms_midpoints: # If no midpoints were computed (e.g., only 1 agent overall)
        return torch.full((sampled_pos.shape[0],), float('inf'), device=sampled_pos.device, dtype=sampled_pos.dtype)

    # Concatenate all calculated midpoints into a single tensor
    all_comms_midpoints = torch.cat(comms_midpoints, dim=0) # [Total_Midpoints, 2]

    # Calculate min distance from each sampled_pos to the generated midpoints
    return min_dist(all_comms_midpoints, sampled_pos)

def goto_base(agent_obs: Dict[str, Tensor], world_idx: int, sampled_pos: Tensor) -> Tensor:
    """
    Given base location and sampled points, returns the distance to the base
    for each sampled point.

    Args:
        agent_obs (Dict[str, Tensor]): Agent's local observations from the scenario.
                                       Must include "obs_base".
        world_idx (int): The index of the current environment (batch).
        sampled_pos (Tensor): Points from the RRT tree, shape `[N_sampled_points, 2]`.

    Returns:
        Tensor: Distances to the base for each sampled point, shape `[N_sampled_points]`.
    """
    if "obs_base" not in agent_obs:
        print("'obs_base' not included in agent local observations")
        return torch.full((sampled_pos.shape[0],), float('inf'), device=sampled_pos.device, dtype=sampled_pos.dtype)

    base = agent_obs["obs_base"][world_idx] # Gets base pos for corresponding world, shape [2]
    
    # Calculate Euclidean distance from each sampled_pos to the base
    # (base: [2] - sampled_pos: [N_sampled_points, 2]) broadcasts base correctly
    return torch.norm(base - sampled_pos, dim=-1) # Result shape: [N_sampled_points]