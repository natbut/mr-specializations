import torch
from vmas.simulator.core import *
from vmas.simulator.core import Agent


class Coordinator(Agent):
    
    def __init__(
        self,
        name: str,
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
        
        self.agent_type = 'coordinator'
        self.obs = []
        
        super().__init__(name, shape, movable, rotatable, collide, density, mass, f_range, max_f, t_range, max_t, v_range, max_speed, color, alpha, obs_range, obs_noise, u_noise, u_range, u_multiplier, action_script, sensors, c_noise, silent, adversary, drag, linear_friction, angular_friction, gravity, collision_filter, render_action, dynamics, action_size, discrete_action_nvec)
        
    def process_obs(self, obs):
        """Process communicated observations, record in internal map"""
        # TODO
        
        self.obs = obs
        
        
    def update_specializations(self, specs_policy):
        """
        Use specs_policy to updated agent specializations
        
        return torch.Tensor
        """
        combined_obs = []
        for key in self.obs.keys(): # convert all to floats
            if key == 'pos':
                continue
            # print(key, obs[0][key].float())
            combined_obs.append(self.obs[key].float())
        combined_obs = torch.cat(combined_obs, dim=1)
    
        return specs_policy.forward(combined_obs)
    
    def update_specializations_pop(self, policy_population):
        """
        Use specs_policy to updated agent specializations
        
        return torch.Tensor
        """
        combined_obs = []
        for key in self.obs.keys(): # convert all to floats
            if key == 'pos':
                continue
            # print(key, obs[0][key].float())
            combined_obs.append(self.obs[key].float())
        combined_obs = torch.cat(combined_obs, dim=1)
    
        specs = []
        for i, policy in enumerate(policy_population):
            out = policy.forward(combined_obs[i])
            specs.append(torch.stack(out, dim=0))
            
        specs = torch.cat(specs, dim=1)
        specs = torch.reshape(specs, (policy_population[0].num_agents,
                                      len(policy_population),
                                      policy_population[0].num_modes))
        return specs
    
        
    def get_action(self, env, id):
        """
        Coordinator does not move.
        """
        action = (torch.tensor([0], device=env.device, dtype=torch.long)
                    .unsqueeze(-1)
                    .expand(env.batch_dim, 1)
                )
        return action.clone()