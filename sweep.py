# from experiment import train_PPO
# from envs.scenarios.SR_tasks import Scenario
from envs.scenarios.explore_comms_tasks import Scenario
from experiment_vec import sweep_PPO


if __name__ == "__main__":

    ### List test config files here ###

    # Env, Scenario & params
    scenario = Scenario()  
    scenario_configs = "conf/scenarios/exploring_0.yaml", #SR_tasks_5.yaml",
    env_configs = "conf/envs/planning_env_explore.yaml", #planning_env_vec_4.yaml",
    
    # Model Params
    model_configs = "conf/models/mat_8.yaml",

    # Sweep Config (RL hyperparams)
    sweep_configs = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "train/mean_reward"},
        "parameters": {
            "lr": {"min": 0.0001, "max": 0.01},
            "max_grad_norm": {"values": [1.0]},  # Use "value" to keep constant
            "frames_per_batch": {"min": 512, "max": 4096},
            "total_frames": {"values": [1000000]},
            "sub_batch_size": {"min": 64, "max": 512},
            "num_epochs": {"min": 4, "max": 32},
            "clip_epsilon": {"values": [0.2]},
            "gamma": {"min": 0.9, "max": 0.99},
            "lmda": {"min": 0.8, "max": 0.95},
            "entropy_eps": {"min": 0.001, "max": 0.1},
        }
    }

    
    sweep_PPO(scenario, 
            scenario_configs,
            env_configs,
            model_configs,
            sweep_configs,
            project_name="mothership-complex",
            entity="nlbutler18-oregon-state-university"
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
    