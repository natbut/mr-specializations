
import torch
from torchrl.data import CompositeSpec, BoundedTensorSpec, Unbounded

from scenarios.SR_tasks import Scenario
from planning_env import VMASPlanningEnv
from tensordict.tensordict import TensorDict

if __name__ == "__main__":


    env = VMASPlanningEnv(
        scenario=Scenario(),
        batch_size=2,
        device="cuda",
        )
    
    env.render = True
    
    # actions = torch.zeros((env.batch_size[0],
    #                       env.scenario.n_agents,
    #                       env.scenario.n_tasks + env.scenario.n_agents + env.scenario.n_obstacles),
    #                       device="cuda"
    #                       )  # Random actions

    actions = torch.tensor([[[1., 0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0., 0.]],

                            [[0., 5., 5., 0., 0., 0.],
                            [0., 5., 5., 0., 0., 0.]]], device='cuda:0')
    
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
    