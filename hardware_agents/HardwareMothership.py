
import argparse
import copy
import os
import socket
import sys
import threading
import time
from collections import defaultdict

from HardwareAgent import *
from tensordict import TensorDict
from torch import float32, tensor
from torch.serialization import add_safe_globals, load

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from envs.planning_env_vec import VMASPlanningEnv
from envs.scenarios.explore_comms_tasks import Scenario
from experiment_vec import create_actor, create_env, init_device
from models.transformer import EnvironmentTransformer


class Mothership(HardwareAgent):
    
    D_TYPE = float32

    def __init__(self, id):
        """Planning & comms coordination agent for Passenger robots."""

        super().__init__(id)

        self.recieved_obs = {} # Real observations, in lat/lon
        self.scaled_obs = {} # Observations scaled to [-1, 1]

        self.policy = None # Model for coordinating passenger robots
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

        # Initialize agents pos observations
        # Initially assume all at mothership
        agents = {}
        for i in range(self.num_passengers):
            agents[i] = copy.deepcopy(self.scaled_obs["mother_pos"])
        self.scaled_obs["agents_pos"] = agents
        print(f"Init scaled agents_pos:", self.scaled_obs["agents_pos"])

        self.my_location = self.scaled_obs["mother_pos"]
        

        # Load model
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
        for a_id, params in enumerate(joint_specs):
            self.prepare_message("spec_params", a_id, (a_id, params))
            

    def _query_policy(self):
        """
        Perform forward pass of policy model to get agents specialization params.
        
        """
        
        # Create observation tensors
        print("Creating observation tensors...")
        agents_pos = []
        for a_id in self.scaled_obs["agents_pos"]:
            agents_pos.append(self.scaled_obs["agents_pos"][a_id])

        cell_feats = tensor(self.scaled_obs["cells"].values(),
                          dtype=self.D_TYPE,
                          device=self.device)
        cell_pos = tensor(self.scaled_obs["cells"].keys(),
                          dtype=self.D_TYPE,
                          device=self.device)
        num_cells = tensor([len(self.scaled_obs["cells"])],
                           dtype=self.D_TYPE,
                           device=self.device)
        rob_data = tensor(agents_pos,
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
        

        # Query model to get actions
        print("Running policy...")
        actions = self.policy.forward(tdict)
        
        heuristic_weights = actions["action"] 
        heuristic_weights = heuristic_weights.view(
            heuristic_weights.shape[0],
            self.num_passengers,
            len(self.num_specializations)
        ) # Breaks weights apart per-robot
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
        no_transformer = model_config.get("no_transformer", False)
        rob_pos_enc = model_config.get("rob_pos_enc", True)

        self.device = init_device()

        dummy_env = create_env(Scenario(), 
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
                                            no_transformer,
                                            rob_pos_enc,
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




def listener():
    global coordinate_trigger
    s = socket.socket()
    s.bind(('localhost', 9997)) 
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

    args = parser.parse_args()

    # Create agent
    mothership = Mothership(args.robot_id)
    mothership.load_deployment_config(args.config_fp) 

    # Trigger initialization
    threading.Thread(target=listener, daemon=True).start()

    # Action loop
    while True:
    
        # Process any recieved messages
        mothership.receive_messages()

        # Process planning commands
        if coordinate_trigger:
            print("Planning triggered")
            coordinate_trigger = False
            mothership.send_spec_params_message() # create and share params
        else:
            print("Mothership socket waiting...")
            
        # Simulate message sending if enabled
        if args.sim_comms:
            mothership.dummy_send_messages() 

        time.sleep(1)

