
import torch
from torchrl.data import CompositeSpec, BoundedTensorSpec, Unbounded

from scenarios.SR_tasks import Scenario
from planning_env import VMASPlanningEnv
from tensordict.tensordict import TensorDict

if __name__ == "__main__":
    node_dim = 4

    env = VMASPlanningEnv(
        scenario=Scenario(),
        batch_size=2,
        device="cuda",
        node_dim=node_dim
        )
    
    env.render = True
    
    # actions = torch.zeros((env.batch_size[0],
    #                       env.scenario.n_agents,
    #                       env.scenario.n_tasks + env.scenario.n_agents + env.scenario.n_obstacles),
    #                       device="cuda"
    #                       )  # Random actions

    actions = torch.tensor([[1., 1., 1., 1., 1.] for _ in range(node_dim**2)], device="cuda")
    actions = torch.stack([actions for _ in range(2)])\
    
    print("Actions:", actions)
    
    # Convert actions into a TensorDict
    actions_tdict = TensorDict({"action": actions}, batch_size=env.batch_size)
    print("Actions:", actions_tdict)

    obs_graph = env.reset()
    print("OBS", obs_graph)
    # print("\nReset Obs Graph:\n", obs_graph["graph"], "Num graphs:", obs_graph["graph"].num_graphs) #, "\n Graph 0:\n", obs_graph[0])
    # print("\nGraphs to Data list:\n", obs_graph["graph"].to_data_list())

    for _ in range(10):
        next_tdict= env.step(actions_tdict)

    print("ENV STEP RETURN:", next_tdict)
    