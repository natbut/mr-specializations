
import torch
from torchrl.data import CompositeSpec, BoundedTensorSpec, Unbounded

from scenarios.SR_tasks import Scenario
from planning_env import VMASPlanningEnv
from tensordict.tensordict import TensorDict

if __name__ == "__main__":


    env = VMASPlanningEnv(
        scenario=Scenario(),
        batch_size=(2,),
        device="cuda",
        )
    
    actions = torch.zeros(env.batch_size[0],
                          env.scenario.n_agents,
                          env.scenario.n_tasks
                          )  # Random actions

    for i in range(env.scenario.n_agents):
        actions[:, i, i] = 1
    # Convert actions into a TensorDict
    actions_tdict = TensorDict({"actions": actions}, batch_size=env.batch_size)

    obs_graph = env.reset()
    print("\nReset Obs Graph:\n", obs_graph["graph"], "Num graphs:", obs_graph["graph"].num_graphs) #, "\n Graph 0:\n", obs_graph[0])
    print("\nGraphs to Data list:\n", obs_graph["graph"].to_data_list())

    next_state, rewards, done, _ = env.step(actions_tdict)
    
    print("Next State:", next_state)
    print("Rewards:", rewards)
    print("Done:", done)