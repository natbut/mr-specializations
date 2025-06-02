import torch

# File for storing heuristic evaluation functions

def nearest_task(agent_obs, world_idx, current_pos):
    """
    Given set of candidate tasks pos & current pos, returns (normalized?)
    distance to nearest frontier pos

    Requires scenario.observation to assign agents with local obs that include
    task positions, labeled "obs_tasks"
    """
    if "obs_tasks" not in agent_obs:
        print("'obs_tasks' not included in agent local observations")
        return 0

    tasks = agent_obs["obs_tasks"][:, world_idx]  # Gets tasks for corresponding world
    return min_dist(tasks, current_pos)

def nearest_agent(agent_obs, world_idx, current_pos):
    """
    Given set of candidate agents pos & current pos, returns (normalized?)
    distance to nearest frontier pos

    Requires scenario.observation to assign agents with local obs that include
    agent positions, labeled "obs_agents"
    """
    if "obs_agents" not in agent_obs:
        print("'obs_agents' not included in agent local observations")
        return 0

    agents = agent_obs["obs_agents"][:, world_idx] # Gets agents for corresponding world
    # Remove own location from agents tensor
    mask = ~torch.all(agents == current_pos, dim=1)
    agents = agents[mask]
    return min_dist(agents, current_pos)

def nearest_frontier(agent_obs, world_idx, current_pos):
    """
    Given set of candidate frontiers pos & current pos, returns (normalized?)
    distance to nearest frontier pos

    Requires scenario.observation to assign agents with local obs that include
    frontier positions, labeled "obs_frontiers"
    """
    if "obs_frontiers" not in agent_obs:
        # print("'obs_frontiers' not included in agent local observations")
        return 0

    frontiers = agent_obs["obs_frontiers"][:, world_idx] # Gets frontiers for corresponding world
    return min_dist(frontiers, current_pos)

def nearest_comms_midpt(agent_obs, world_idx, current_pos):
    """
    Given set of candidate comms pos & current pos, returns (normalized?)
    distance to nearest frontier pos

    Requires scenario.observation to assign agents with local obs that include
    comms positions, labeled "obs_comms_midpt"
    """
    if "obs_comms_midpt" not in agent_obs:
        print("'obs_comms_midpt' not included in agent local observations")
        return 0

    agents = agent_obs["obs_comms_midpt"][:, world_idx] # Gets agents for corresponding world
    return min_dist(agents, current_pos)

def goto_base(agent_obs, world_idx, current_pos):
    """
    Given base location, returns (normalized?) distance to base

    Requires scenario.observation to assign agents with local obs that include
    base position, labeled "obs_base"
    """
    if "obs_base" not in agent_obs:
        print("'obs_base' not included in agent local observations")
        return 0

    base = agent_obs["obs_base"][:, world_idx] # Gets base pos for corresponding world
    return min_dist(base, current_pos)


def min_dist(points_a, points_b):
    return torch.min(torch.norm(points_a - points_b, dim=1))