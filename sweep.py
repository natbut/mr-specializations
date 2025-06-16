import sys
import yaml
# from experiment import train_PPO
# from envs.scenarios.SR_tasks import Scenario
from envs.scenarios.explore_comms_tasks import Scenario
from experiment_vec import sweep

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python sweep.py <sweep_fp>")
        sys.exit(1)

    sweep_fp = sys.argv[1]

    # Env, Scenario & params
    scenario = Scenario()  
    scenario_configs = "conf/scenarios/exploring_0.yaml", #SR_tasks_5.yaml",
    env_configs = "conf/envs/planning_env_explore.yaml", #planning_env_vec_4.yaml",
    
    # Model Params
    model_configs = "conf/models/mat_8.yaml",

    with open(sweep_fp, 'r') as file:
        sweep_params = yaml.safe_load(file)

        # Sweep Config (RL hyperparams
        sweep_configs = {
            "method": "grid",
            "metric": {"goal": "maximize", "name": "train/mean_reward"},
            "parameters": sweep_params
        }

        sweep(
            scenario, 
            scenario_configs,
            env_configs,
            model_configs,
            sweep_configs,
            project_name="mothership-complex",
            entity="nlbutler18-oregon-state-university",
            conf_name=sweep_fp.split('/')[-1].split('.')[0]
        )
