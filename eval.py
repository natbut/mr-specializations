# import cProfile
# import pstats
import sys

# from experiment import train_PPO
# from envs.scenarios.SR_tasks import Scenario
from envs.scenarios.explore_comms_tasks import Scenario
from experiment_vec import eval

# with cProfile.Profile() as pr:

if __name__ == "__main__":

    if len(sys.argv) < 6:
        print("Usage: python train.py <scenario_fp> <env_fp> <model_fp> <checkpt_fp> <save_fp> <num_evals> <steps_per_rollout>")
        sys.exit(1)

    scenario_fp = sys.argv[1]
    env_fp = sys.argv[2]
    model_fp = sys.argv[3]
    checkpt_fp = sys.argv[4]
    save_fp = sys.argv[5]
    num_evals = int(sys.argv[6])
    steps_per_rollout = int(sys.argv[7])

    # Env, Scenario & params
    scenario = Scenario()  
    scenario_configs = [scenario_fp]
        # "conf/scenarios/exploring_0.yaml", #SR_tasks_5.yaml",
    # ]
    env_configs = [env_fp]
        # "conf/envs/planning_env_explore_1.yaml", #planning_env_vec_4.yaml",
    # ]

    # Model Params
    model_configs = [model_fp]
        # "conf/models/mat_9.yaml",
    # ]

    # Checkpoint
    checkpoint = [checkpt_fp]
    if checkpt_fp == "None":
        checkpt_fp = None

    for i in range(num_evals):
        eval(scenario, 
            scenario_configs,
            env_configs,
            model_configs,
            checkpt_fp,
            save_fp,
            eval_id=i,
            rollout_steps=steps_per_rollout,
            )
        
#  python eval.py "conf/scenarios/comms_0.yaml" "conf/envs/planning_env_explore_2.yaml" "conf/models/mat_2_0.yaml" "runs\comms_0_planning_env_explore_2_ppo_4_2_flatAct_mat_2_0\checkpoints\best.pt" "eval_test" 1 8