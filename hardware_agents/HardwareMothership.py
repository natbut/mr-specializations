
import argparse
import copy
import os
import socket
import sys
import threading
import time
import os
import socket
import sys
import threading
import time
from collections import defaultdict

import torch
from HardwareAgent import *
from tensordict import TensorDict
from torch import float32, tensor
from torch.serialization import add_safe_globals, load
from tensordict import TensorDict
from torch import float32, tensor
from torch.serialization import add_safe_globals, load

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from envs.scenarios.explore_comms_tasks import Scenario
from experiment_vec import create_actor, create_env, init_device

from experiment_vec import create_actor, create_env, init_device


class Mothership(HardwareAgent):
    
    D_TYPE = float32

    def __init__(self, id):
    
    D_TYPE = float32

    def __init__(self, id):
        """Planning & comms coordination agent for Passenger robots."""

        super().__init__(id)
        super().__init__(id)

        self.recieved_obs = {} # Real observations, in lat/lon
        self.scaled_obs = {} # Observations scaled to [-1, 1]

        self.policy = None # Model for coordinating passenger robots
        self.device = "cpu"
        
        self.num_specializations = 0 # Number of possible passenger specializations
        self.device = "cpu"
        
        self.num_specializations = 0 # Number of possible passenger specializations

    
    def load_deployment_config(self, config_fp):
        super().load_deployment_config(config_fp)

        with open(config_fp, 'r') as file:
            params = yaml.safe_load(file)

            model_conf_fp = params["model_conf_fp"]
            model_weights_fp = params["model_weights_fp"]
            scenario_conf_fp = params["scenario_conf_fp"]
            env_conf_fp = params["env_conf_fp"]
            
            self.num_specializations = len(params["heuristic_fns"])
            
            self.use_max = params.get("action_max", False)

        # Initialize agents pos observations
        # Initially assume all at mothership
        agents = {}
        for i in range(self.num_passengers):
            agents[i+1] = copy.deepcopy(self.scaled_obs["mother_pos"])
        for i in range(self.num_passengers):
            agents[i+1] = copy.deepcopy(self.scaled_obs["mother_pos"])
        self.scaled_obs["agents_pos"] = agents
        print(f"Init scaled agents_pos:", self.scaled_obs["agents_pos"])

        self.my_location = self.scaled_obs["mother_pos"]
        
        

        # Load model
        model, policy = self.initialize_policy(model_conf_fp,
        model, policy = self.initialize_policy(model_conf_fp,
                                            model_weights_fp,
                                            scenario_conf_fp,
                                            env_conf_fp
                                            )
        self.policy = policy


    def send_spec_params_message(self):
        """
        Use model to find specialization parameters for agents.
        """
        joint_specs = self._query_policy()

        # Parse policy output to get per-passenger params
        for i, params in enumerate(joint_specs):
            a_id = i+1
            self.prepare_message("spec_params", a_id, (a_id, params.tolist()))
            
    def _update_dist_feature_value(self, loc, feature: torch.tensor):
        # Return 0.0 if feature tensor is empty
        if feature.numel() == 0:
            return 0.0

        loc_tensor = torch.tensor(loc).unsqueeze(0)  # Shape (1, 2)
        
        dists = torch.norm(loc_tensor - feature, dim=-1)

        if self.sparse:
            # Sum distances of features within discrete_resolution/2
            mask = dists < self.discrete_resolution / 2
            return dists[mask].sum().item() if mask.any() else 0.0

        min_dist, _ = dists.min(dim=-1)
        if (dists < self.discrete_resolution / 2).any(dim=-1):
            return 1.0
        else:
            return torch.clamp(1.0 - (min_dist / self.scaled_max_dist), min=0.0, max=self.scaled_max_dist).item()
        
        """ Compute distance feature value. Is 1.0 if feature is in same cell as agent. """
        
        # Return 0.0 if feature tensor is empty
        # if feature.numel() == 0:
        #     return 0.0

        # loc_tensor = torch.tensor(loc).unsqueeze(0)  # Shape (1, 2)
        
        # dists = torch.norm(loc_tensor - feature, dim=-1)
        
        # min_dist, _ = dists.min(dim=-1)

        # if (dists < self.discrete_resolution/2).any(dim=-1):
        #     return 1.0
        # else:
        #     return torch.clamp(1.0 - (min_dist/self.scaled_max_dist), min=0.0, max=self.scaled_max_dist).item()
            

    def _query_policy(self):
        """
        Perform forward pass of policy model to get agents specialization params.
        
        """
        
        # !! Update cell features for tasks and agents !!
        # 1) Prep obs to tensors
        tasks_tensor = torch.tensor(list(self.scaled_obs["tasks_pos"].values()))
        agents_tensor = torch.tensor(list(self.scaled_obs["agents_pos"].values()))
        for cell_pos in self.scaled_obs["cells"].keys():
        
            # Compute feature values
            task_obs = self._update_dist_feature_value(cell_pos, tasks_tensor)
            agent_obs = self._update_dist_feature_value(cell_pos, agents_tensor)
            
            # obs_vec = [task_obs,
            #        obst_obs,
            #        agent_obs,
            #        frontiers_obs,
            #        exploration_obs,
            #        mother_obs,
            #        ]
            
            print(f"Updating cell {cell_pos}: {self.scaled_obs["cells"][cell_pos]}")
            
            self.scaled_obs["cells"][cell_pos][0] = task_obs
            self.scaled_obs["cells"][cell_pos][2] = agent_obs
            # self.scaled_obs["cells"][cell_pos][3] = 1.0
            
            print(f"To {cell_pos}: {self.scaled_obs["cells"][cell_pos]}" )
            
        
        # Create observation tensors
        print("Creating observation tensors...")
        cell_feats = tensor(list(self.scaled_obs["cells"].values()),
                          dtype=self.D_TYPE,
                          device=self.device)
        cell_pos = tensor(list(self.scaled_obs["cells"].keys()),
                          dtype=self.D_TYPE,
                          device=self.device)
        num_cells = tensor(len(self.scaled_obs["cells"]),
                           dtype=self.D_TYPE,
                           device=self.device)
        rob_data = tensor(list(self.scaled_obs["agents_pos"].values()),
                          dtype=self.D_TYPE,
                          device=self.device)
        num_robs = tensor([self.num_passengers],
                          dtype=self.D_TYPE,
                          device=self.device)

        # Configure observation tensordict
        print("Creating tensordict...")
        tdict = TensorDict()
        tdict.set("cell_feats", cell_feats) # TODO may need to pad
        tdict.set("cell_pos", cell_pos) # TODO may need to pad
        tdict.set("num_cells", num_cells) # TODO may need to max
        tdict.set("rob_data", rob_data)
        tdict.set("num_robs", num_robs) # TODO may need to max
        
        print("Prepared tdict:", tdict)

        # Query model to get actions
        print("Running policy...")
        actions = self.policy.forward(tdict)        
        
        heuristic_weights = actions["action"] 
        heuristic_weights = heuristic_weights.view(
            self.num_passengers,
            self.num_specializations
        ) # Breaks weights apart per-robot
        
        if self.use_max:
            # Set the max value in each H vector to 1, others to 0
            max_indices = torch.argmax(heuristic_weights, dim=-1, keepdim=True)
            heuristic_weights = torch.zeros_like(heuristic_weights)
            heuristic_weights.scatter_(-1, max_indices, 1.0)
        
        print("Specializations: ", heuristic_weights)

        return heuristic_weights
    

    def initialize_policy(self, model_fp, weights_fp, scenario_fp, env_fp):
        """
        Initialize model for mothership coordination agent.
        """

        print("Initializing Mothership model...")
        with open(model_fp, 'r') as file:
            model_config = yaml.safe_load(file)

        with open(scenario_fp, 'r') as file:
            scenario_config = yaml.safe_load(file)

        with open(env_fp, 'r') as file:
            env_config = yaml.safe_load(file)
        
        add_safe_globals([defaultdict])
        add_safe_globals([list])
        checkpt_data = load(weights_fp, weights_only=True)

        print("Loaded checkpoint data. Creating dummy env...")

        num_features = model_config["num_features"]
        num_heuristics = model_config["num_heuristics"]
        d_feedforward = model_config["d_feedforward"]
        d_model = model_config["d_model"]
        agent_attn=model_config["agent_attn"]
        cell_pos_as_features=model_config["cell_pos_as_features"]
        agent_id_enc = model_config["agent_id_enc"]
        use_encoder = model_config.get("use_encoder", True)
        use_decoder = model_config.get("use_decoder", True)
        use_encoder = model_config.get("use_encoder", True)
        use_decoder = model_config.get("use_decoder", True)
        rob_pos_enc = model_config.get("rob_pos_enc", True)

        self.device = init_device()
        self.device = init_device()

        dummy_env = create_env(Scenario(), 
                               self.device, 
                               self.device, 
                               env_config, 
                               scenario_config, 
                               check_specs=False
                               )
        print("Dummy environment created. Creating policy...")

        tf_act, policy_module = create_actor(dummy_env,
                                            num_features,
                                            num_heuristics,
                                            d_feedforward,
                                            d_model,
                                            agent_attn, 
                                            cell_pos_as_features, 
                                            agent_id_enc, 
                                            use_encoder,
                                            use_decoder,
                                            use_encoder,
                                            use_decoder,
                                            rob_pos_enc,
                                            self.device
                                            self.device
                                            )
        print("Model and policy created, loading weights...")
        
        tf_act.load_state_dict(checkpt_data['actor_state_dict'])
        tf_act.eval()

        print("Model initialization complete.")

        return tf_act, policy_module

    def send_new_tasks_message(self):
        """Send message with new tasks"""

        # TODO

        self.prepare_message()

        pass



base_ports = {
        "plan": 10000,
        "update": 11000,
        "coordinate": 9999
    }

def listener():
    global coordinate_trigger
    s = socket.socket()
    s.bind(('localhost', base_ports["coordinate"])) 
    s.bind(('localhost', base_ports["coordinate"])) 
    s.listen(1)
    while True:
        conn, _ = s.accept()
        coordinate_trigger = True
        conn.close()

if __name__ == "__main__":
    coordinate_trigger = False

    parser = argparse.ArgumentParser(description="Run simulated hardware agents")
    parser.add_argument("--config_fp", type=str, required=True, help="Path to problem config file")    
    parser.add_argument("--robot_id", type=int, default=0, help="Mothership ID")
    parser.add_argument("--sim_comms", type=bool, default=False, help="Dummy comms bool. Defaults to False (no simulated comms)")
    parser.add_argument("--robot_id", type=int, default=0, help="Mothership ID")
    parser.add_argument("--sim_comms", type=bool, default=False, help="Dummy comms bool. Defaults to False (no simulated comms)")

    args = parser.parse_args()

    # Create agent
    mothership = Mothership(args.robot_id)
    mothership = Mothership(args.robot_id)
    mothership.load_deployment_config(args.config_fp) 

    # Trigger initialization
    # Trigger initialization
    threading.Thread(target=listener, daemon=True).start()

    # Action loop
    while True:
    
        # Process any recieved messages
        mothership.receive_messages()
    
        # Process any recieved messages
        mothership.receive_messages()

        # Process planning commands
        if coordinate_trigger:
            print("Coordinate triggered")
            coordinate_trigger = False
            mothership.send_spec_params_message() # create and share params
            print("Mothership socket waiting...")
            
        # Simulate message sending if enabled
        if args.sim_comms:
            mothership.dummy_send_messages() 

        time.sleep(1)

