from typing import Union

import numpy as np
import torch
from vmas.simulator.core import *


class Worker(Agent):
    
    def __init__(
        self,
        name: str,
        modality_funcs: List = None,
        sim_action_func = None,
        sim_velocity: float = 0.05,
        shape: Shape = None,
        movable: bool = True,
        rotatable: bool = True,
        collide: bool = True,
        density: float = 25.0,  # Unused for now
        mass: float = 1.0,
        f_range: float = None,
        max_f: float = None,
        t_range: float = None,
        max_t: float = None,
        v_range: float = None,
        max_speed: float = None,
        color=Color.BLUE,
        alpha: float = 0.5,
        obs_range: float = None,
        obs_noise: float = None,
        u_noise: Union[float, Sequence[float]] = 0.0,
        u_range: Union[float, Sequence[float]] = 1.0,
        u_multiplier: Union[float, Sequence[float]] = 1.0,
        action_script: Callable[[Agent, World], None] = None,
        sensors: List[Sensor] = None,
        c_noise: float = 0.0,
        silent: bool = True,
        adversary: bool = False,
        drag: float = None,
        linear_friction: float = None,
        angular_friction: float = None,
        gravity: float = None,
        collision_filter: Callable[[Entity], bool] = lambda _: True,
        render_action: bool = False,
        dynamics: Dynamics = None,  # Defaults to holonomic
        action_size: int = None,  # Defaults to what required by the dynamics
        discrete_action_nvec: List[
            int
        ] = None,
        
        ):
        
        self.agent_type = 'worker'
        self.obs = []
        self.modality_funcs = modality_funcs
        self.sim_action_func = sim_action_func
        self.sim_velocity = sim_velocity
        self.action_dict = {0: [0.0,0.0],
                        1: [0.0,-self.sim_velocity],
                        2: [0.0,self.sim_velocity],
                        3: [-self.sim_velocity,0.0],
                        4: [-self.sim_velocity,-self.sim_velocity],
                        5: [-self.sim_velocity,self.sim_velocity],
                        6: [self.sim_velocity,0.0],
                        7: [self.sim_velocity,-self.sim_velocity],
                        8: [self.sim_velocity,self.sim_velocity],                   
                        }
        self.specialization = None
        
        super().__init__(name, shape, movable, rotatable, collide, density, mass, f_range, max_f, t_range, max_t, v_range, max_speed, color, alpha, obs_range, obs_noise, u_noise, u_range, u_multiplier, action_script, sensors, c_noise, silent, adversary, drag, linear_friction, angular_friction, gravity, collision_filter, render_action, dynamics, action_size, discrete_action_nvec)
        
    def process_obs(self, obs):
        """
        Process agent observations, potentially record in internal map
        
        obs dim is obs_dim x batch_size
        """
        
        self.obs = obs
        
    # def _sim_action(self, action_id, env):
    #     # print('!! Sim action', action_id)
    #     # print("Sim init obs:\n", self.obs)
        
    #     # Translate action id tensor to action
    #     # print(torch.tensor(self.action_dict[action_id]))
    #     # print(torch.tensor(self.action_dict[action_id]).expand(env.batch_dim, 2))
    #     motion = (torch.tensor(self.action_dict[action_id], device=env.device)
    #                             .expand(env.batch_dim, 2)
    #                             )
    #     # print("Motion:", motion)
    #     sim_obs = self.sim_action_func(self.obs, motion)
    #     # print("Sim new obs:\n", sim_obs)
        
    #     # print("Recheck original obs:\n", self.obs)
    #     return sim_obs
        
    def get_action(self,
                   env,
                   id,
                   ):
        """
        Use coordinator-assigned agent specialization to select action
        
        Discrete action indices: 0-8, with:
        - 0 remain stationary
        - 1 down
        - 2 up
        - 3 left
        - 4 down left
        - 5 up left
        - 6 right
        - 7 down right
        - 8 up right
        """
        
        # if selection_strategy == 'one_step': # Use one-step lookahead selection
                
        # Evaluate action per-environment
        best_act_ids = (torch.tensor([0], device=env.device, dtype=torch.long)
                            # .unsqueeze(-1)
                            .expand(env.batch_dim, 1)
                            )
        best_act_vals = (torch.tensor([-1], device=env.device)
                            # .unsqueeze(-1)
                            .expand(env.batch_dim, 1)
                            )
        
        for act in self.action_dict.keys():
            # print("Eval act", act)
            # Evaluate each action impact given obs
            act_tensor = (torch.tensor([act], device=env.device, dtype=torch.long)
                            .unsqueeze(-1)
                            .expand(env.batch_dim, 1)
                            )
            
            # Simulate impacts of action
            # sim_obs = self._sim_action(act, env)
            motion = (torch.tensor(self.action_dict[act], device=env.device)
                                .expand(env.batch_dim, 2)
                                )
            modality_vals = []
            for mode_func in self.modality_funcs:
                modality_vals.append(mode_func(self.obs, motion))
            modality_vals = torch.stack(modality_vals, dim=1)

            # Process modality values, scale according to specialization
            # print("Act", act, "modality vals:", modality_vals, modality_vals.shape)
            # print("Worker specs:", self.specialization, self.specialization.shape)
            act_vals = (modality_vals * self.specialization).sum(1).unsqueeze(-1)
            # print("Act", act, " vals:", act_vals)
            
            # NOTE Finding action that minimizes vals
            best_act_ids = torch.where(act_vals > best_act_vals, act_tensor, best_act_ids)
            best_act_vals = torch.where(act_vals > best_act_vals, act_vals, best_act_vals)

        # print("Best act vals:\n", best_act_vals)
        # print("Best acts:\n", best_act_ids)

        action = best_act_ids
    
        return action.clone()