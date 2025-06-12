
import torch
from tensordict.tensordict import TensorDict

from envs.planning_env_vec import VMASPlanningEnv
# from envs.scenarios.SR_tasks import Scenario
from envs.scenarios.explore_comms_tasks import Scenario
from experiment_vec import load_yaml_to_kwargs

if __name__ == "__main__":
    
    scenario_configs = [
        "conf/scenarios/exploring_0.yaml",
    ]
    env_configs = [
        "conf/envs/planning_env_explore.yaml",
    ]

    env = VMASPlanningEnv(Scenario(),
                            device="cuda",
                            env_kwargs=load_yaml_to_kwargs(env_configs[0]),
                            scenario_kwargs=load_yaml_to_kwargs(scenario_configs[0])
                            )

    env.render = True

    # act = [torch.stack([torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device="cuda")]) for _ in range(8)]
    actions = torch.tensor([0.0001, 0.0001, 0.0001, 0.0001, 0.0001], device="cuda")
    actions = torch.stack([actions for _ in range(3)]) # stack for robots
    actions[0][0] = 1.0
    actions[1][1] = 1.0
    actions[2][2] = 1.0
    actions = torch.stack([actions for _ in range(8)]) # stack for envs
    
    print("Actions:", actions)
    
    # Convert actions into a TensorDict
    actions_tdict = TensorDict({"action": actions}, batch_size=env.batch_size)
    print("Actions:", actions_tdict)

    obs = env.reset()
    print("OBS", obs)
    # print("\nReset Obs Graph:\n", obs_graph["graph"], "Num graphs:", obs_graph["graph"].num_graphs) #, "\n Graph 0:\n", obs_graph[0])
    # print("\nGraphs to Data list:\n", obs_graph["graph"].to_data_list())

    for i in range(3):
        print("STEP", i)
        next_tdict= env.step(actions_tdict)

    print("ENV STEP RETURN:", next_tdict)
    