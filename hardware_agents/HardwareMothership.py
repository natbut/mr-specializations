
import argparse
import socket, threading, time

from HardwareAgent import *

class Mothership(HardwareAgent):

    def __init__(self):
        """Planning & comms coordination agent for Passenger robots."""

        super().__init__()

        self.recieved_obs = {} # Real observations, in lat/lon
        self.scaled_obs = {} # Observations scaled to [-1, 1]

    
    def load_deployment_config(self, config_fp):
        super().load_deployment_config(config_fp)

        with open(config_fp, 'r') as file:
            params = yaml.safe_load(file)
            num_passengers = params["num_passengers"]

        # Initialize agents pos observations
        # Initially assume at mothership
        agents = {}
        for i in range(num_passengers):
            agents[i] = copy.deepcopy(self.scaled_obs["mother_pos"])
        self.scaled_obs["agents_pos"] = agents

        self.my_location = self.scaled_obs["mother_pos"]

        print(f"Init scaled agents_pos:", self.scaled_obs["agents_pos"])


    def send_spec_params_message(self):
        """
        Use model to find specialization parameters for agents.

        Store params in file format to be sent out by acoustic modems.
        """
        # TODO

        self.prepare_message()

        pass

    def _query_model(self):

        # TODO

        pass

    def initialize_model(self, model, weights):

        # TODO

        pass

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
    parser.add_argument("--model_fp", type=str, required=True, help="Path to model config file")
    parser.add_argument("--weights_fp", type=str, required=True, help="Path to model weights file")

    args = parser.parse_args()

    # Create agent
    mothership = Mothership()
    mothership.load_deployment_config(args.config_fp) 
    mothership.initialize_model(args.model_fp, args.weights_fp)

    # Comms initialization (as neeeded)
    # TODO

    # Planning initialization
    threading.Thread(target=listener, daemon=True).start()

    # Action loop
    while True:

        # Process new comms messages
        # TODO: Check for new yaml, update mothership properties

        # Process planning commands
        if coordinate_trigger:
            print("Planning triggered")
            coordinate_trigger = False
            mothership.send_spec_params_message() # create and share params
        else:
            print("Planning socket waiting...")

        time.sleep(1)

