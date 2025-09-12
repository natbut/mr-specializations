import time
from typing import Dict, Tuple

import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from vmas import make_env
from vmas.simulator.scenario import BaseScenario

import envs.heuristics

from moviepy import ImageSequenceClip

def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    func = dotpath #dotpath.rsplit(".", maxsplit=1)
    m = envs.heuristics #(module_)
    # print("Func:", func, "m:", m)
    return getattr(m, func)

class VMASPlanningEnv(EnvBase):
    def __init__(
            self, scenario: BaseScenario, 
            device: str = "cpu",
            env_kwargs: Dict = None,
            scenario_kwargs: Dict = None,
            ):
        
        self.scenario = scenario

        self.heuristic_eval_fns_names = env_kwargs.pop("heuristic_fns", None)
        self.set_heuristic_eval_fns(self.heuristic_eval_fns_names)
        self.num_envs = env_kwargs.pop("num_envs", 1)
        self.horizon = env_kwargs.pop("horizon", 0.25)
        self.planning_pts = env_kwargs.pop("planning_pts", 30)
        self.macro_step = env_kwargs.pop("macro_step", 10) # Number of sub-steps to execute per step 
        self.render = env_kwargs.pop("render", False)

        self.use_softmax = False
        self.use_max = False

        self.render_fp = None
        self.count = 0

        super().__init__(batch_size=[self.num_envs], device=device)
        # VMAS Environment Configuration
        self.sim_env = make_env(
                            scenario=self.scenario,
                            num_envs=self.batch_size[0],
                            max_steps=env_kwargs.pop("max_steps", 100),
                            device=self.device,
                            **scenario_kwargs,
                            )
    
        n_features = self.scenario.num_feats #self.scenario.n_agents + self.scenario.n_tasks + self.scenario.n_obstacles

        # MAX_CELLS = 100

        # Define Observation & Action Specs
        self.observation_spec = Composite(
            # obs=Composite(
                cell_feats=Unbounded(
                    shape=(self.num_envs, -1, n_features),
                    dtype=torch.float64,
                    device=device
                    ),
                cell_pos=Unbounded(
                    shape=(self.num_envs, -1, 2),
                    dtype=torch.float64,
                    device=device
                ),
                num_cells=Unbounded(
                    shape=(self.num_envs,),
                    dtype=torch.float32,
                    device=device
                ),
                rob_data=Unbounded(
                    shape=(self.num_envs, -1, 3), # (self.num_envs, self.sim_env.n_agents, 3)
                    dtype=torch.float64,
                    device=device
                ),
                num_robs=Unbounded(
                    shape=(self.num_envs,),
                    dtype=torch.float32,
                    device=device
                ),
            #     device=device
            # ),
            shape=(self.num_envs,),
            device=device
        )

        self.action_spec = Bounded(
            low=-torch.ones((
                            self.num_envs,
                            self.sim_env.n_agents * self.n_heuristics
                            ),
                            device=device
                            ),
            high=torch.ones((
                            self.num_envs,
                            self.sim_env.n_agents * self.n_heuristics
                            ),
                            device=device
                            ),
            shape=(self.num_envs, self.sim_env.n_agents * self.n_heuristics),
            device=device
        ) # NOTE ADDED TO HAVE ALL ROBOTS AS ONE TRANSFORMER ACTION

        self.reward_spec = Unbounded(
            shape=(self.num_envs, 1),
            device=device
            )
            
        self.steps = 0

    def set_heuristic_eval_fns(self, fns: list):
        """Temporarily set eval fns to those provided"""
        self.heuristic_eval_fns = [load_func(name) for name in fns]
        self.n_heuristics = len(self.heuristic_eval_fns)

    def reset_heuristic_eval_fns(self):
        """Reset eval fns to stored fns"""
        self.heuristic_eval_fns = [load_func(name) for name in self.heuristic_eval_fns_names]
        self.n_heuristics = len(self.heuristic_eval_fns)

    def _reset(self, obs_tensordict=None) -> TensorDict:
        """Reset all VMAS worlds and return initial state."""
        sim_obs = self.sim_env.reset() # Gets global obs from agent 0

        # print("STEP SIM OBS:", sim_obs[0])
        obs = TensorDict(sim_obs[0], batch_size=self.batch_size, device=self.device)
        
        # out.set("step_count", torch.full(self.batch_size, 1))
        # print("Reset TDict:", obs)
        # print("Expanded:", [(x_i, edges_i) for x_i, edges_i in zip(self.graph_obs["x"], self.graph_obs["edge_index"])])
        return obs

    def _step(self, actions: TensorDict) -> TensorDict:
        """
        Steps through the environment (IN PARALLEL...sort of):
        1. Use heuristic weights to plan agent trajectories.
        2. Execute the trajectories for the given horizon.
        3. Return next state, rewards, and termination status.
        """

        # Process actions (heuristic weights)
        heuristic_weights = actions["action"] 
        heuristic_weights = heuristic_weights.view(
            heuristic_weights.shape[0],
            self.sim_env.n_agents,
            len(self.heuristic_eval_fns)
        ) # Breaks weights apart per-robot
        if self.use_softmax:
            heuristic_weights = torch.softmax(heuristic_weights, dim=-1)
        if self.use_max:
            # Set the max value in each H vector to 1, others to 0
            max_indices = torch.argmax(heuristic_weights, dim=-1, keepdim=True)
            heuristic_weights = torch.zeros_like(heuristic_weights)
            heuristic_weights.scatter_(-1, max_indices, 1.0)
            # print("Using max action selection; weights are:", heuristic_weights)
        if self.render:
            print(f"\nHeuristic Weights:\n {heuristic_weights} Shape: {heuristic_weights.shape}") # [B, N_AGENTS, N_FEATS]
        # print("WEIGHTS SAMPLE:", heuristic_weights[:5], "shape:", heuristic_weights.shape)
        # print("AGENTS:", self.sim_env.scenario.agents, "shape:", len(self.sim_env.agents))

        # Execute and aggregate rewards
        rewards = torch.zeros((self.num_envs, 1), device=self.device)
        # dones = torch.full((self.num_envs, 1), False, device=self.device)
        # null_rews = torch.zeros_like(rewards, device=self.device)
        frame_list = []
        for agent in self.sim_env.agents: # Reset agent trajectories
            agent.trajs = []
        # print("\n= Pre-rollout step! =")
        self.steps += 1
        # print("Active agents:", [a.is_active for a in self.sim_env.agents])
        for t in range(self.macro_step):
            verbose = False
            # Compute agent trajectories & get actions
            u_action = []
            # t_start = time.time()
            i = 0
            for agent in self.sim_env.agents:
                if agent.is_active:
                    u_action.append(agent.get_control_action_cont(heuristic_weights[:,i,:], self.heuristic_eval_fns, self.horizon, self.planning_pts, verbose
                                                                )) # get next actions from agent controllers
                    i += 1
                else:
                    u_action.append(agent.null_action)

            # BURST TASK SPAWNING (ONLY ON LAST STEP)
            if t == self.macro_step-1 and self.sim_env.scenario.spawn_tasks_burst:
                self.sim_env.scenario.spawn_tasks(attempts=self.macro_step)
            sim_obs, rews, dones, _ = self.sim_env.step(u_action)

            team_rews = torch.stack(rews).sum(dim=0)
            rewards += team_rews #torch.where(dones == False, team_rews, null_rews)

            if self.render:
                frame = self.sim_env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                )
                frame_list.append(frame)

            if dones.any():
                # print("Step", self.steps, "\nRETURNED DONES:", dones.unsqueeze(-1), "\nAccumulated rewards:", rewards)
                break 
            
        if self.render:
            fps = 20
            clip = ImageSequenceClip(frame_list, fps=fps)
            clip.write_gif(f"{self.render_fp}_{self.count}.gif", fps=fps)
            self.count += 1


        # print("Rewards:", rewards, "\nDones:", dones.unsqueeze(1))
        
        # print("STEP SIM OBS:", sim_obs[0])

        # Construct next state representation   
        next_state = TensorDict(
            sim_obs[0], # Global obs taken from first agent (any will do)
            device=self.device,
        )
        rew_tdict = TensorDict(
            {"reward": rewards},
            device=self.device,
        )
        done_tdict = TensorDict(
            {"done": dones.unsqueeze(1)},
            device=self.device,
        )

        # next_state.set("step_count", 1)

        # Should return next_state, rewards, done as tensordict
        next_state.update(rew_tdict).update(done_tdict) 
        # print("Next TDict:\n", next_state)
        
        # print("= Post-rollout step! = ")

        return next_state #, rewards, done, {}

    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        self.sim_env.seed(seed)
