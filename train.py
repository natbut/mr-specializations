from experiment import train_PPO
from envs.scenarios.SR_tasks import Scenario



if __name__ == "__main__":

    ### List test config files here ###

    # Env, Scenario & params
    scenario = Scenario()  
    scenario_configs = [
        "conf/scenarios/SR_tasks.yaml",
    ]
    env_configs = [
        "conf/envs/planning_env_2.yaml",
    ]

    # RL Hyperparams
    rl_configs = [
        "conf/algos/ppo_2_2.yaml",
    ]

    # Model Params
    model_configs = [
        "conf/models/mlp_1.yaml",
    ]

    
    train_PPO(scenario,
              scenario_configs,
              env_configs,
              rl_configs,
              model_configs,
              )


    # == Problem/Project Notes ==

    # 1. We have a set of "candidate behaviors/specializations" for passenger
    # 2. Mothership/Agent learns to prioritize these behaviors/specializations given
    # state of environment to maximize reward G.

    # Why not have Mothership/Agent use a sampling-based planner to find best behavior?
    # 1. Can be expensive/slow (especially for multiagent)
    # 2. Communication overhead (especially for multiagent)
    # 3 Information mismatch: Mothership/Agent priorization observation space is different
    # from the operating agent's observation space.