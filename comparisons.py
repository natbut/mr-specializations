import sys, os
import yaml
import torch
from torchrl.modules import ProbabilisticActor
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import (ExplorationType,
                                set_exploration_type)
from tensordict.tensordict import TensorDict
from collections import defaultdict

from envs.scenarios.explore_comms_tasks import Scenario
from envs import planning_env_vec
from experiment_vec import load_yaml_to_kwargs, init_device, create_env, create_actor

from pathlib import Path

# Get the parent directory of the current file
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent

# Append "mcts-smop" directory to sys.path
mcts_smop_path = parent_dir / "mcts-smop"
sys.path.append(str(mcts_smop_path))

from planner import HybDecPlanner
from control.task import Task
import csv

def test_setup(test,
               scenario_configs,
               env_configs,
               model_configs,
               checkpoint,
               comp_configs
               ) -> tuple[TransformedEnv, ProbabilisticActor, HybDecPlanner, defaultdict, dict]:

    # LOAD CONFIGS #
    print("Loading configs & checkpoint...")
    scenario_config = load_yaml_to_kwargs(scenario_configs[test])
    env_config = load_yaml_to_kwargs(env_configs[test])
    model_config = load_yaml_to_kwargs(model_configs[test])
    comp_config = load_yaml_to_kwargs(comp_configs[test])
    # LOAD MODEL WEIGHTS #
    torch.serialization.add_safe_globals([defaultdict])
    torch.serialization.add_safe_globals([list])
    checkpt_data = torch.load(checkpoint, weights_only=True)

    device = init_device()

    # CREATE SIM ENVIRONMENT #
    print("Creating sim environment...")
    env = create_env(scenario, device, env_config, scenario_config)

    # ACTOR POLICY MODULES, LOAD WEIGHTS #
    print("Loading model weights...")
    num_features = model_config["num_features"]
    num_heuristics = model_config["num_heuristics"]
    d_feedforward = model_config["d_feedforward"]
    d_model = model_config["d_model"]
    agent_attn = model_config["agent_attn"]
    cell_pos_as_features=model_config["cell_pos_as_features"]
    agent_id_enc = model_config.get("agent_id_enc", True)
    tf_act, policy_module = create_actor(env, num_features, num_heuristics, d_feedforward, d_model, agent_attn, cell_pos_as_features, agent_id_enc, device)
    tf_act.load_state_dict(checkpt_data['actor_state_dict'])
    tf_act.eval()

    logs = defaultdict(list)

    # CREATE PLANNER #
    print("Creating planner...")
    sim_data, merger_data, dec_mcts_data, sim_brvns_data = generate_planner_data(scenario_config,
                                                                                comp_config,
                                                                                )
    planner = HybDecPlanner(sim_data, merger_data, dec_mcts_data, sim_brvns_data)

    return env, policy_module, planner, logs, sim_data


def run_tests(scenario_configs,
              env_configs,
              test_configs,
              model_configs,
              checkpoint,
              comp_configs,
              folder_path="test",
              ):
    """Run tests to evaluate model vs comparison method."""

    TASKS_ONLY = [1, 0, 0, 0, 0, 0]
    COMMS_ONLY = [0, 0, 0, 0, 1, 0]

    ### CYCLE THROUGH TEST CONFIGS ###
    for test in range(max(len(scenario_configs), len(env_configs), len(model_configs))):
        print(f"Test {test} setup...")
        test_config = load_yaml_to_kwargs(test_configs[test])
                                              
        # Set up environments, policy, planner, and logs details
        env, policy, planner, logs, sim_data = test_setup(test,
                                                scenario_configs, 
                                                env_configs, 
                                                model_configs, 
                                                checkpoint, 
                                                comp_configs,
                                                )
        
        env.base_env.render = True
        
        # Load in test parameters
        num_runs = test_config["num_runs"]
        rollout_steps = test_config["rollout_steps"]
        agent_assignments = test_config["agent_assignments"]
        planning_iters = test_config["planning_iters"]
        
        plan_heuristic_fns = ["goto_goal", "neediest_comms_midpt", "nearest_agent"]

        data_dir = os.path.join(folder_path, "data")
        os.makedirs(data_dir, exist_ok=True)
        csv_fp = os.path.join(data_dir, "results.csv")
        csv_header = [
            "test", "run",
            "policy_reward_mean", "policy_reward_sum",
            "h_tasks_reward_mean", "h_tasks_reward_sum",
            "h_split_reward_mean", "h_split_reward_sum",
            "planner_reward_mean", "planner_reward_sum"
        ]

        for run in range(num_runs):

            seed = TensorDict({"seed": torch.tensor([run,])}, batch_size=[1])

            # ---- Run test with policy ----
            print(f"Preparing to run test {test} with policy...")

            # Configure for eval with policy
            render_name = f"render_policy"
            render_fp = os.path.join(f"{folder_path}/gif/policy/", render_name)
            env.base_env.render_fp = render_fp
            env.base_env.count = run*rollout_steps
            os.makedirs(os.path.dirname(render_fp), exist_ok=True)

            # Set heuristics for policy
            env.base_env.reset_heuristic_eval_fns()
            
            # Run test & log   
            policy_reward_mean, policy_reward_sum = run_policy(env, logs, policy, rollout_steps, seed=seed)
            print("Mean reward policy:", logs["policy reward (mean)"])


            # ---- Run test with h_tasks only ----
            
            # Fix agent actions
            agents = env.base_env.sim_env.scenario.active_agents
            actions = []
            for _ in agents:
                actions += TASKS_ONLY
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            h_tasks_reward_mean, h_tasks_reward_sum = run_fixed_heuristics(env, logs, actions, rollout_steps, name="all_tasks", seed=seed, folder_path=folder_path)            
            print("Mean all_tasks heuristics:", logs["all_tasks reward (mean)"])


             # ---- Run test with split h_tasks, h_comms ----
            
            # Fix agent actions
            agents = env.base_env.sim_env.scenario.active_agents
            actions = []
            for i, _ in enumerate(agents):
                if i % 2 == 0:
                    actions += TASKS_ONLY
                else:
                    actions += COMMS_ONLY
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            h_split_reward_mean, h_split_reward_sum = run_fixed_heuristics(env, logs, actions, rollout_steps, name="split", seed=seed, folder_path=folder_path)      
            print("Mean split heuristics:", logs["split h reward (mean)"])


            # ---- Run test with planner ----
            print("\nPreparing to run test with planner...")
            render_name = f"render_planner"
            render_fp = os.path.join(f"{folder_path}/gif/planner/", render_name)
            env.base_env.render_fp = render_fp
            env.base_env.count = run*rollout_steps
            os.makedirs(os.path.dirname(render_fp), exist_ok=True)

            # Configure for eval with planner
            # Update heuristics for planning
            env.base_env.set_heuristic_eval_fns(plan_heuristic_fns)

            # Update actions according to assignments
            actions = []
            agents = env.base_env.sim_env.scenario.active_agents
            for i, assignment in enumerate(agent_assignments):
                if assignment == "WORKER":
                    actions += [1, 0, 0]
                    agents[i].set_mode("WORKER")
                elif assignment == "SUPPORT":
                    actions += [1, 0, 0]
                    agents[i].set_mode(f"SUPPORT_{i}")
                else:
                    actions += [0, 0, 0]  # fallback/default
            actions = TensorDict({"action": torch.tensor(actions, dtype=torch.float32).unsqueeze(0)}, batch_size=env.batch_size)

            # Run test & log
            print("Running test...")
            env.reset(seed) # TODO seed
            # env.base_env._set_seed(run)
            planner_rollout_rewards = rollout_with_planner(env,
                                        actions,
                                        planner,
                                        sim_data,
                                        rollout_steps,
                                        planning_iters,
                                        )
            planner_reward_sum = sum(planner_rollout_rewards[1:])
            planner_reward_mean = planner_reward_sum / len(planner_rollout_rewards[1:])
            logs["planner reward (mean)"].append(planner_reward_mean)
            logs["planner reward (sum)"].append(planner_reward_sum)
            
            print("Mean reward planner:", logs["planner reward (mean)"])

            for a in agents:
                a.set_mode("NoMode")


            # ---- Save results to CSV ----
            write_header = not os.path.exists(csv_fp)
            with open(csv_fp, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(csv_header)
                writer.writerow([
                    test, run,
                    policy_reward_mean, policy_reward_sum,
                    h_tasks_reward_mean, h_tasks_reward_sum,
                    h_split_reward_mean, h_split_reward_sum,
                    planner_reward_mean, planner_reward_sum
                ])
        
        print("Done!")

def run_fixed_heuristics(env: TransformedEnv, logs: defaultdict, actions, rollout_steps: int, name: str, seed, folder_path="test"):
    print("Running fixed heuristics...")
    render_name = f"render_{name}"
    render_fp = os.path.join(f"{folder_path}/gif/{name}/", render_name)
    env.base_env.render_fp = render_fp
    env.base_env.count = seed["seed"][0]*rollout_steps
    os.makedirs(os.path.dirname(render_fp), exist_ok=True)

    env.reset(seed) # TODO seed
    # env.base_env._set_seed(seed)
    fixed_rollout_rewards = rollout_fixed_actions(env, actions, rollout_steps)
    reward_sum = sum(fixed_rollout_rewards[1:])
    reward_mean = reward_sum / len(fixed_rollout_rewards[1:])
    logs[name + " reward (mean)"].append(reward_mean)
    logs[name + " reward (sum)"].append(reward_sum)

    return reward_mean, reward_sum

def run_policy(env: TransformedEnv, logs: defaultdict, policy, rollout_steps: int, seed):
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # execute a rollout with the trained policy
        print("Running policy...")
        env.reset(seed) # TODO seed
        # env.base_env._set_seed(seed)
        policy_rollout = env.rollout(rollout_steps, policy, return_contiguous=False)
        print("Policy reward shape:", policy_rollout["next", "reward"].shape)
        policy_reward_mean = policy_rollout["next", "reward"][0][1:].mean().item()  # Skip first step (startup)
        policy_reward_sum = policy_rollout["next", "reward"][0][1:].sum().item()  # Skip first step (startup)
        logs["policy reward (mean)"].append(policy_reward_mean)
        logs["policy reward (sum)"].append(policy_reward_sum)
        logs["action"] = policy_rollout["action"]

    return policy_reward_mean, policy_reward_sum


def rollout_fixed_actions(env: TransformedEnv,
                            actions,
                            rollout_steps: int,
                            ):
    logs = {}
    logs["reward"] = 0.0
    rewards = []
    # For step in rollout_steps:
    for i in range(rollout_steps):

        # Roll out plan
        tdict = env.base_env._step(actions)
        rewards.append(tdict["reward"].item())

    
    # logs["reward"] += tdict["reward"].item()

    return rewards



def rollout_with_planner(env: TransformedEnv,
                         actions,
                         planner,
                         sim_data,
                         rollout_steps: int,
                         planning_iters: int
                         ):
    scen = env.base_env.sim_env.scenario
    agents = scen.active_agents

    logs = {}
    logs["reward"] = 0.0
    rewards = []

    # Update agent planning observations
    tasks_pos, workers_pos = get_updated_tasks_workers(agents)
    base_pos = scen.base.state.pos.squeeze().tolist() + [0.0] # 3D pos add Z
    
    # Create task dict
    task_dict = {}
    task_dict[sim_data["start"]] = Task(sim_data["start"], base_pos, 0, 1)
    task_dict[sim_data["end"]] = Task(sim_data["end"], base_pos, 0, 1)

    for i, pos in enumerate(tasks_pos):
        pos += [0.0] # 3D pos add Z
        task_dict["v"+str(i)] = Task("v"+str(i), pos, 1, 1) # work 1, reward 1

    # Create initial plan
    planner.prepare_init_plans(task_dict, workers_pos, base_pos)
    
    # For step in rollout_steps:
    for step in range(rollout_steps):

        # Update goal assignments
        i = 0
        for agent in agents:
            if agent.mode == "WORKER":
                # Manually set schedule goals according to planner
                agent.set_mission_plan(planner.get_agent_plan(i))
                i += 1

        # Roll out plan
        tdict = env.base_env._step(actions)
        logs["reward"] += tdict["reward"].item()
        rewards.append(tdict["reward"].item())

        # Create next plan (if steps remain)
        task_dict = {}
        if step < rollout_steps-1:
            # Update agent planning observations
            tasks_pos, workers_pos = get_updated_tasks_workers(agents)
            for j, pos in enumerate(tasks_pos):
                pos += [0.0] # 3D pos add Z
                task_dict["v"+str(j)] = Task("v"+str(j), pos, 1, 1) # work 1, reward 1
            task_dict[sim_data["rob_task"]] = Task(sim_data["rob_task"][:], base_pos[:], 0, 1) # Will be overwritten in planner
            task_dict[sim_data["end"]] = Task(sim_data["end"][:], base_pos[:], 0, 1)

            # Create plans
            planner.solve_worker_schedules(workers_pos, task_dict, planning_iters)

    return rewards

def get_updated_tasks_workers(agents):
    """Process new observations from environment"""

    return_task_pos = []
    tasks_pos = agents[0].obs["obs_tasks"][0].squeeze().tolist()[:]
    print("TASKS POS:", tasks_pos)
    for t in tasks_pos:
        if t[0] < 2.0: return_task_pos.append(t)
    print("RETURN TASKS POS:", return_task_pos)
    workers_pos = [a.state.pos.squeeze().tolist()+[0.0] for a in agents if a.mode == "WORKER"]

    return return_task_pos, workers_pos


def generate_planner_data(scenario_config, solver_config):
    """Load data into correct configurations for solver"""
    
    # with open(scenario_config, "r") as p_fp:
    #     scenario_config = yaml.safe_load(p_fp)
    #     with open(solver_config, "r") as s_fp:
    #         solver_config = yaml.safe_load(s_fp)

    dims = (tuple(solver_config["xCoordRange"]),
            tuple(solver_config["yCoordRange"]),
            tuple(solver_config["zCoordRange"]),
            )

    sim_data = {  # "graph": deepcopy(planning_graph),
        "start": solver_config["start"],
        "end": solver_config["end"],
        "c": solver_config["c"],
        "budget": solver_config["budget"],
        "velocity": solver_config["velocity"],
        "energy_burn_rate": solver_config["energy_burn_rate"],
        "basic": solver_config["basic"],
        "m_id": solver_config["m_id"],
        "env_dims": dims,
        "rob_task": solver_config["rob_task"],
        "base_loc": solver_config["base_loc"], # Dummy initial base location
    }

    merger_data = {"rel_mod": solver_config["rel_mod"],
                    "rel_thresh": solver_config["rel_thresh"],
                    "mcs_iters": solver_config["mcs_iters"]
                    }

    dec_mcts_data = {"num_robots": solver_config["num_workers"],
                        "fail_prob": solver_config["failure_probability"],
                        "comm_n": solver_config["comm_n"],
                        "plan_iters": solver_config["planning_iters"], # for DecMCTS
                        "t_max": solver_config["t_max_decMCTS"],
                        "sim_iters": solver_config["sim_iters"] # for MCS
                        }

    sim_brvns_data = {"num_robots": solver_config["num_workers"],
                        "alpha": solver_config["alpha"],
                        "beta": solver_config["beta"],
                        "k_initial": solver_config["k_initial"],
                        "k_max": solver_config["k_max"],
                        "t_max": solver_config["t_max_simBRVNS"],
                        "t_max_init": solver_config["t_max_init"],
                        "explore_iters": solver_config["exploratory_mcs_iters"],
                        "intense_iters": solver_config["intensive_mcs_iters"],
                        "act_samples": solver_config["act_samples"],
                        }

    return sim_data, merger_data, dec_mcts_data, sim_brvns_data



if __name__ == "__main__":

    if len(sys.argv) < 6:
        print("Usage: python run_tests.py <scenario_fp> <env_fp> <test_fp> <model_fp> <checkpt_fp> <comp_fp> <folder_name>")
        sys.exit(1)

    scenario_fp = sys.argv[1]
    env_fp = sys.argv[2]
    test_fp = sys.argv[3]
    model_fp = sys.argv[4]
    checkpt_fp = sys.argv[5]
    comp_fp = sys.argv[6]
    test_folder_name = sys.argv[7]

    # Env, Scenario & params
    scenario = Scenario()  
    scenario_configs = [scenario_fp]
        # "conf/scenarios/exploring_0.yaml", #SR_tasks_5.yaml",
    # ]
    env_configs = [env_fp]
        # "conf/envs/planning_env_explore_1.yaml", #planning_env_vec_4.yaml",
    # ]

    test_configs = [test_fp]
        # "conf/envs/planning_env_explore_1.yaml", #planning_env_vec_4.yaml",
    # ]

    # Model Params
    model_configs = [model_fp]
        # "conf/models/mat_9.yaml",
    # ]

    # Comparison Method Params
    comp_configs = [comp_fp]
        # "conf/models/mat_9.yaml",
    # ]

    # Checkpoint
    checkpoint = checkpt_fp


    run_tests(scenario_configs,
              env_configs,
              test_configs,
              model_configs,
              checkpoint,
              comp_configs,
              folder_path=test_folder_name
              )
    


# python comparisons.py "conf/scenarios/comms_5_eval.yaml" "conf/envs/planning_env_explore_5_1env.yaml" "conf/tests/trial1.yaml" "conf/models/mat_2_3.yaml" "runs\comms_5_planning_env_explore_5_ppo_4_7_mat_2_3\checkpoints\best.pt" "conf/hybdec/solver1.yaml"