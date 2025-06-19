import time
from typing import Dict, Tuple

import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from vmas import make_env
from vmas.simulator.scenario import BaseScenario

from envs.heuristics import *


class VMASPlanningEnv(EnvBase):
    def __init__(
            self, scenario: BaseScenario, 
            device: str = "cpu",
            env_kwargs: Dict = None,
            scenario_kwargs: Dict = None,
            ):
        
        # TODO: Make this part of conf file
        self.heuristic_eval_fns = [nearest_task, 
                                   nearest_comms_midpt, 
                                   neediest_comms_midpt,#farthest_comms_midpt,
                                   nearest_frontier, 
                                   nearest_agent, 
                                   goto_base]
        
        self.scenario = scenario

        self.num_envs = env_kwargs.pop("num_envs", 1)
        self.horizon = env_kwargs.pop("horizon", 0.25)
        self.macro_step = env_kwargs.pop("macro_step", 10) # Number of sub-steps to execute per step 
        self.render = env_kwargs.pop("render", False)

        self.render_fp = None
        self.count = 0

        # TODO Check kwargs passing
        if self.num_envs == 1:
            super().__init__(batch_size=[], device=device)
            # VMAS Environment Configuration
            self.sim_env = make_env(
                                scenario=self.scenario,
                                num_envs=1,
                                max_steps=env_kwargs.pop("max_steps", 100),
                                device=self.device,
                                **scenario_kwargs,
                                )

            n_features = self.scenario.num_feats #self.scenario.n_agents + self.scenario.n_tasks + self.scenario.n_obstacles
            
            # Define Observation & Action Specs
            self.observation_spec = Composite(
                obs=Composite(
                    cell_feats=Unbounded(
                        shape=(-1, n_features),
                        dtype=torch.float64,
                        device=device
                        ),
                    cell_pos=Unbounded(
                        shape=(-1, 2),
                        dtype=torch.float64,
                        device=device
                    ),
                    rob_pos=Unbounded(
                        shape=(self.sim_env.n_agents, 2),
                        dtype=torch.float64,
                        device=device
                    ),
                    device=device
                ),
                shape=(1,),
                device=device
            )

            self.action_spec = Bounded(
                low=-torch.ones((
                                self.sim_env.n_agents,
                                len(self.heuristic_eval_fns)
                                ),
                                device=device
                                ),
                high=torch.ones((
                                self.sim_env.n_agents,
                                len(self.heuristic_eval_fns)
                                ),
                                device=device
                                ),
                shape=(self.sim_env.n_agents, 
                       len(self.heuristic_eval_fns)),
                device=device
            )

            self.reward_spec = Unbounded(
                shape=(1,), #(self.sim_env.n_agents),
                device=device
                )#, self.scenario.n_agents))

        else:
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
                    rob_pos=Unbounded(
                        shape=(self.num_envs, self.sim_env.n_agents, 2),
                        dtype=torch.float64,
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
                                self.sim_env.n_agents,
                                n_features
                                ),
                                device=device
                                ),
                high=torch.ones((
                                self.num_envs,
                                self.sim_env.n_agents,
                                n_features
                                ),
                                device=device
                                ),
                shape=(self.num_envs, self.sim_env.n_agents, n_features),
                device=device
            )

            self.reward_spec = Unbounded(
                shape=(self.num_envs, 1),
                device=device
                )
            
        self.steps = 0

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
        # heuristic_weights = actions.view(self.batch_size, self.scenario.n_agents, self.scenario.n_tasks)
        heuristic_weights = actions["action"] # NOTE THESE ARE WEIGHTS FOR EVERY NODE, BUT WE ONLY USE AGENT LOCS
        if self.render:
            print(f"\nHeuristic Weights:\n {heuristic_weights} Shape: {heuristic_weights.shape}") # [B, N_AGENTS, N_FEATS]
        # print("WEIGHTS:", heuristic_weights, "shape:", heuristic_weights.shape)

        # Execute and aggregate rewards
        rewards = torch.zeros((self.num_envs, 1), device=self.device)
        # dones = torch.full((self.num_envs, 1), False, device=self.device)
        # null_rews = torch.zeros_like(rewards, device=self.device)
        frame_list = []
        for agent in self.sim_env.agents: # Reset agent trajectories
            agent.trajs = []
        # print("\n= Pre-rollout step! =")
        self.steps += 1
        for t in range(self.macro_step):
            verbose = False
            # Compute agent trajectories & get actions
            u_action = []
            # t_start = time.time()
            for i, agent in enumerate(self.sim_env.agents):
                u_action.append(agent.get_control_action_cont(heuristic_weights[:,i,:],
                                                              self.heuristic_eval_fns,
                                                              self.horizon,
                                                              verbose
                                                              )) # get next actions from agent controllers
            # print(f"Planing took {time.time() - t_start} s")
            # t_start = time.time()
            # print("U-ACTION:", u_action)
            sim_obs, rews, dones, info = self.sim_env.step(u_action)
            # print(f"Step took {time.time() - t_start} s")
            # print("Rewards:", rewards, "stacked rews:", rews, "sum:", rews.sum(dim=0))

            # NOTE Ignores additional rewards from envs that reset during rollout (from completing early)
                # Hypothesis here is that envs reset at done indices and continue to accumulate negative rewards
                # Ideally, we won't even step completed envs, but this is a quick fix for now
            # done = done.unsqueeze(1)
            # dones = torch.where(done == True, done, dones)
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
            from moviepy import ImageSequenceClip
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
