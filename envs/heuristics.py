import torch

# File for storing heuristic evaluation functions

def nearest_task(agent_obs, world_idx, sampled_pos):
    """
    Given set of candidate tasks pos & current pos, returns (normalized?)
    distance to nearest frontier pos

    Requires scenario.observation to assign agents with local obs that include
    task positions, labeled "obs_tasks"
    """
    if "obs_tasks" not in agent_obs:
        print("'obs_tasks' not included in agent local observations")
        return 0

    tasks = agent_obs["obs_tasks"][world_idx]  # Gets tasks for corresponding world
    # print("TASKS:", tasks)
    mask = torch.any(tasks <= 1.0, dim=-1)
    # print("MASK:", mask)
    tasks = tasks[mask]
    # print("NEW TASKS:", tasks)
    return min_dist(tasks, sampled_pos)

def nearest_agent(agent_obs, world_idx, sampled_pos):
    """
    Given set of candidate agents pos & current pos, returns (normalized?)
    distance to nearest frontier pos

    Requires scenario.observation to assign agents with local obs that include
    agent positions, labeled "obs_agents"
    """
    if "obs_agents" not in agent_obs:
        print("'obs_agents' not included in agent local observations")
        return 0

    agents = agent_obs["obs_agents"][world_idx] # Gets agents for corresponding world
    # Remove own location from agents tensor
    # mask = ~torch.all(agents == sampled_pos, dim=1)
    # print("AGENTS:", agents, "\n CURRENT POS:", sampled_pos, "\nMASK:", mask)
    # agents = agents[mask]
    # print("AGENTS AFTER MASK:", agents)
    return min_dist(agents, sampled_pos)

def nearest_frontier(agent_obs, world_idx, sampled_pos):
    """
    Given set of candidate frontiers pos & current pos, returns (normalized?)
    distance to nearest frontier pos

    Requires scenario.observation to assign agents with local obs that include
    frontier positions, labeled "obs_frontiers"
    """
    if "obs_frontiers" not in agent_obs:
        # print("'obs_frontiers' not included in agent local observations")
        return 0
    if len(agent_obs["obs_frontiers"]) == 0:
        # print(f"No frontiers initialized yet")
        return 0

    frontiers = agent_obs["obs_frontiers"][world_idx] # Gets frontiers for corresponding world
    # print(f"World {world_idx} frontiers: {frontiers}, \n Current Pos: {sampled_pos}, \nMin dist: {min_dist(frontiers, sampled_pos)}")
    # print("Frontiers min dist:", min_dist(frontiers, sampled_pos))
    return min_dist(frontiers, sampled_pos)

def nearest_comms_midpt(agent_obs, world_idx, sampled_pos):
    """
    Given set of candidate comms pos & current pos, returns (normalized?)
    distance to nearest frontier pos

    Requires scenario.observation to assign agents with local obs that include
    comms positions, labeled "obs_comms_midpt"
    """
    if "obs_agents" not in agent_obs or "obs_base" not in agent_obs:
        print("'obs_agents' or 'obs_base' not included in agent local observations")
        return 0
    
    # Compute midpoints between agents-agents and agents-base
    agents = agent_obs["obs_agents"][world_idx]
    base = agent_obs["obs_base"][world_idx]

    # print("Agents pos:", agents)
    # print("Base pos:", base)

    comms_midpt = (agents + base) / 2.0

    if len(agents) > 1:
        agents_midpt = (agents + agents) / 2.0
        comms_midpt = torch.cat((comms_midpt, agents_midpt), dim=0)

    # print("Comms midpt:", comms_midpt)
    # print("Min dist:", min_dist(comms_midpt, sampled_pos))

    return min_dist(comms_midpt, sampled_pos)

def goto_base(agent_obs, world_idx, sampled_pos):
    """
    Given base location, returns (normalized?) distance to base

    Requires scenario.observation to assign agents with local obs that include
    base position, labeled "obs_base"
    """
    if "obs_base" not in agent_obs:
        print("'obs_base' not included in agent local observations")
        return 0

    base = agent_obs["obs_base"][world_idx] # Gets base pos for corresponding world
    # print("Base pos:", base)
    # print("Sampled pos:", sampled_pos)
    # print("Dist to base:", torch.norm(base - sampled_pos, dim=0))
    return torch.norm(base - sampled_pos, dim=0)


def min_dist(points_a, points_b):
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    return torch.min(torch.norm(points_a - points_b, dim=1))