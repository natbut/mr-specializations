
import torch
from torchrl.data import CompositeSpec, BoundedTensorSpec, Unbounded

from envs.scenarios.SR_tasks import Scenario
from envs.planning_env_vec import VMASPlanningEnv
from tensordict.tensordict import TensorDict
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    EnvBase
)
from experiment_vec import load_yaml_to_kwargs

if __name__ == "__main__":
    
    scenario_configs = [
        "conf/scenarios/SR_tasks.yaml",
    ]
    env_configs = [
        "conf/envs/planning_env_vec_2.yaml",
    ]

    env = VMASPlanningEnv(Scenario(),
                            device="cuda",
                            env_kwargs=load_yaml_to_kwargs(env_configs[0]),
                            scenario_kwargs=load_yaml_to_kwargs(scenario_configs[0])
                            )

    env.render = True

    act = [torch.stack([torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], device="cuda")]) for _ in range(8)]

    actions = torch.stack(act)
    # actions = torch.stack([actions for _ in range(2)])\
    
    print("Actions:", actions)
    
    # Convert actions into a TensorDict
    actions_tdict = TensorDict({"action": actions}, batch_size=env.batch_size)
    print("Actions:", actions_tdict)

    obs_graph = env.reset()
    print("OBS", obs_graph)
    # print("\nReset Obs Graph:\n", obs_graph["graph"], "Num graphs:", obs_graph["graph"].num_graphs) #, "\n Graph 0:\n", obs_graph[0])
    # print("\nGraphs to Data list:\n", obs_graph["graph"].to_data_list())

    for _ in range(4):
        next_tdict= env.step(actions_tdict)

    print("ENV STEP RETURN:", next_tdict)
    