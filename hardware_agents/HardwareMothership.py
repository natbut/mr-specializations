
import argparse
import sys, os
import socket, threading, time
import copy
from collections import defaultdict
from torch.serialization import load, add_safe_globals
from torch import ones
from torchrl.data import Bounded
from torchrl.envs import EnvBase

from HardwareAgent import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from models.transformer import EnvironmentTransformer
from experiment_vec import create_actor, init_device, create_env
from envs.planning_env_vec import VMASPlanningEnv
from envs.scenarios.explore_comms_tasks import Scenario

class Mothership(HardwareAgent):

    def __init__(self):
        """Planning & comms coordination agent for Passenger robots."""

        super().__init__()

        self.recieved_obs = {} # Real observations, in lat/lon
        self.scaled_obs = {} # Observations scaled to [-1, 1]

        self.policy = None # Model for coordinating passenger robots

    
    def load_deployment_config(self, config_fp):
        super().load_deployment_config(config_fp)

        with open(config_fp, 'r') as file:
            params = yaml.safe_load(file)
            num_passengers = params["num_passengers"]

            model_conf_fp = params["model_conf_fp"]
            model_weights_fp = params["model_weights_fp"]
            scenario_conf_fp = params["scenario_conf_fp"]
            env_conf_fp = params["env_conf_fp"]

        # Initialize agents pos observations
        # Initially assume all at mothership
        agents = {}
        for i in range(num_passengers):
            agents[i] = copy.deepcopy(self.scaled_obs["mother_pos"])
        self.scaled_obs["agents_pos"] = agents
        print(f"Init scaled agents_pos:", self.scaled_obs["agents_pos"])

        self.my_location = self.scaled_obs["mother_pos"]

        # Load model
        model, policy = self.initialize_model(model_conf_fp,
                                            model_weights_fp,
                                            scenario_conf_fp,
                                            env_conf_fp
                                            )
        self.policy = policy


    def send_spec_params_message(self):
        """
        Use model to find specialization parameters for agents.

        Store params in file format to be sent out by acoustic modems.
        """
        # TODO

        # NOTE: Will need to parse policy output to get per-passenger params

        self.prepare_message()

        pass

    def _query_model(self):

        # TODO
        tdict = {}

        self.policy.forward(tdict)

        pass

    def initialize_model(self, model_fp, weights_fp, scenario_fp, env_fp):
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

        device = init_device()

        dummy_env = create_env(Scenario(), 
                               device, 
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
                                            device
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


    def receive_message(self, message):
        """Handle Mothership-specific messages."""
        super().receive_message(message)

        for i, msg in enumerate(self.received_message_buffer):
            self.received_message_buffer.pop(i)

            if "obs_vec" == msg[0]:
                # msg = (msg_type, obs_vec)
                obs_vec = msg[1]
                self._update_env_cell(obs_vec)


    def _update_env_cell(self, obs_vec):
        """
        Updates environment cell with observation vector information.
        
        Selects environment cell nearest to feature location
        """

        # TODO

        pass


    def _update_cell_dynamic_feats(self):
        """
        Updates the task and agent distance features (dynamic features) of
        each cell prior to model inference.
        
        """

        # TODO

        pass



def listener():
    global coordinate_trigger
    s = socket.socket()
    s.bind(('localhost', 9999)) 
    s.listen(1)
    while True:
        conn, _ = s.accept()
        coordinate_trigger = True
        conn.close()

if __name__ == "__main__":
    coordinate_trigger = False

    parser = argparse.ArgumentParser(description="Run simulated hardware agents")
    parser.add_argument("--config_fp", type=str, required=True, help="Path to problem config file")    
    # parser.add_argument("--model_fp", type=str, required=True, help="Path to model config file")
    # parser.add_argument("--weights_fp", type=str, required=True, help="Path to model weights file")
    parser.add_argument("--logs_fp", type=str, required=True, help="Path to logs folder")
    parser.add_argument("--robot_id", type=int, default=0, help="Passenger ID")

    args = parser.parse_args()

    # Create agent
    mothership = Mothership()
    mothership.load_deployment_config(args.config_fp) 
    # mothership.initialize_model(args.model_fp, args.weights_fp)

    # Comms initialization (as neeeded)
    # TODO

    # Planning initialization
    threading.Thread(target=listener, daemon=True).start()

    # Action loop
    while True:

        # Process new comms messages
        # TODO: Check for new msg, update mothership properties

        # Process planning commands
        if coordinate_trigger:
            print("Planning triggered")
            coordinate_trigger = False
            mothership.send_spec_params_message() # create and share params
        else:
            print("Planning socket waiting...")

        time.sleep(1)

