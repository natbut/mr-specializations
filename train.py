# import cProfile
# import pstats
import sys

# from experiment import train_PPO
# from envs.scenarios.SR_tasks import Scenario
from envs.scenarios.explore_comms_tasks import Scenario
from experiment_vec import train

# with cProfile.Profile() as pr:

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print("Usage: python train.py <scenario_fp> <env_fp> <algo_fp> <model_fp> <checkpt_fp> <wandb_mode> <project_name>")
        sys.exit(1)

    scenario_fp = sys.argv[1]
    env_fp = sys.argv[2]
    algo_fp = sys.argv[3]
    model_fp = sys.argv[4]
    checkpt_fp = sys.argv[5]
    wandb_mode = sys.argv[6]
    project_name = sys.argv[7]

    # Env, Scenario & params
    scenario = Scenario()  
    scenario_configs = [scenario_fp]
        # "conf/scenarios/exploring_0.yaml", #SR_tasks_5.yaml",
    # ]
    env_configs = [env_fp]
        # "conf/envs/planning_env_explore_1.yaml", #planning_env_vec_4.yaml",
    # ]

    # RL Hyperparams
    rl_configs = [algo_fp]
        # "conf/algos/ppo_4_0.yaml",
    # ]

    # Model Params
    model_configs = [model_fp]
        # "conf/models/mat_9.yaml",
    # ]

    # Checkpoint
    checkpoint = [checkpt_fp]
    if checkpt_fp == "None":
        checkpt_fp = None

    train(scenario, 
            scenario_configs,
            env_configs,
            rl_configs,
            model_configs,
            checkpt_fp,
            wandb_mode=wandb_mode,
            project_name=project_name
            )
    
    # python train.py "conf/scenarios/comms_0.yaml" "conf/envs/planning_env_explore_1.yaml" "conf/algos/ppo_4_0.yaml" "conf/models/mat_2_0.yaml" "TRAIN" "mothership-complex"
    
    # python train.py "conf/scenarios/exploring_0.yaml" "conf/envs/planning_env_explore_1.yaml" "conf/algos/ppo_4_0.yaml" "conf/models/mat_9.yaml" "runs\exploring_0_planning_env_explore_1_ppo_4_0_mat_9\checkpoints\best.pt" "TRAIN" "mothership-complex"

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # # Now you have two options, either print the data or save it as a file
    # # stats.print_stats() # Print The Stats
    # stats.dump_stats("eval_explore.prof") # Saves the data in a file, can me used to see the data visually
