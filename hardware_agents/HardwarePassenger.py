import torch
import argparse
import socket, threading, time
from HardwareAgent import *

class Passenger(HardwareAgent):

    def __init__(self, id):
        """Planning & comms coordination agent for Passenger robots."""

        super().__init__()
        
        
        self.my_id = id

        # Specialization parameters from Mothership
        self.my_specializations = []


    def send_cell_obs_message(self):
        """
        Send message containing:
        - dist to nearest task (1 if in cell)
        - dist to nearest obstacle (1 if in cell)
        - dist to nearest agent (1 if in cell)
        - percent of frontiers around cell
        - cell exploration status (1 or 0)
        - dist to base from cell
        - observation location

        Note: Values scaled in -1, 1 range

        Store in file format to be sent out by acoustic modems.
        """

        # Compute feature values
        task_obs = self._get_dist_feature_value(self.scaled_obs["tasks_pos"])
        obst_obs = self._get_dist_feature_value(self.scaled_obs["obsts_pos"])
        agent_obs = self._get_dist_feature_value(self.scaled_obs["agents_pos"])
        mother_obs = self._get_dist_feature_value(self.scaled_obs["mother_pos"])
        exploration_obs = 1.0 # assumes all cells are explored
        frontiers_obs = 0.0 # assumes all cells are explored

        obs_vec = [task_obs,
                   obst_obs,
                   agent_obs,
                   frontiers_obs,
                   exploration_obs,
                   mother_obs,
                   self.my_location
                   ]
        
        # TODO configure message
        msg = None # contain (msg_type, obs_vec)

        self.prepare_message(msg)


    def send_location_obs_message(self):
        """
        Send out my location (scaled)
        
        Store in file format to be sent out by acoustic modems.
        """

        locs = self.my_location

        # TODO configure message
        msg = None # contain (msg_type, agent_id, agent_pos)

        self.prepare_message(msg)


    def send_completed_task_message(self):
        """
        Send message when agent has completed a task.

        Store in file format to be sent out by acoustic modems.
        """

        # TODO: Process "task completion"

        msg = None # contaion (msg_type, task_id)

        self.prepare_message(msg)


    def receive_message(self, message):
        """Handle Passenger-specific messages."""
        super().receive_message(message)

        for i, msg in enumerate(self.received_message_buffer):
            self.received_message_buffer.pop(i)
            if "new_task" == msg[0]:
                # msg = (msg_type, task_id, task_pos) MESSAGE FORMAT
                # scaled_obs[tasks_pos] = {task_id: task_pos} STORED OBS FORMAT
                self.scaled_obs["tasks_pos"][msg[1]] = msg[2]
            if "spec_params" == msg[0]:
                # msg = (msg_type, agent_id, spec_params)
                if self.my_id == msg[1]:
                    self.my_specializations = msg[2]


    def create_plan(self):
        """
        Creates short-horizon plan using stored data.

        Stores plan in [FORMAT] file for upload to autopilot.       
        """

        pass


    def _get_dist_feature_value(self, feature):
        """ Compute distance feature value. Is 1.0 if feature is in same cell as agent. """
        dists = torch.norm(self.my_location - feature, dim=-1)
        min_dist, _ = dists.min(dim=-1)

        if (dists < self.discrete_resolution/2).any(dim=-1):
            return 1.0
        else:
            return torch.clamp(1.0 - (min_dist/self.scaled_max_dist), min=0.0, max=self.scaled_max_dist)


def listener():
    global planning_trigger
    s = socket.socket()
    s.bind(('localhost', 9999))
    s.listen(1)
    while True:
        conn, _ = s.accept()
        planning_trigger = True
        conn.close()

if __name__ == "__main__":
    planning_trigger = False

    parser = argparse.ArgumentParser(description="Run simulated hardware agents")
    parser.add_argument("--config_fp", type=str, required=True, help="Path to problem config file")
    parser.add_argument("--robot_id", type=int, default=0, help="Passenger ID")

    args = parser.parse_args()

    # Create agent
    passenger = Passenger(args.robot_id)
    passenger.load_deployment_config(args.config_fp) 

    # Comms initialization (as neeeded)
    # TODO

    # Planning initialization
    threading.Thread(target=listener, daemon=True).start()

    # Action loop
    while True:

        # Prepare new messages to send (periodically)
        # TODO my_location
        # TODO completed tasks
        # TODO obs_vec (maybe only when triggered)

        # Process any recieved messages
        # TODO: Check for new yaml, update passenger properties

        # Process planning commands
        if planning_trigger:
            print("Planning triggered")
            planning_trigger = False
            passenger.create_plan() # create and save new plan
        else:
            print("Planning socket waiting...")

        time.sleep(1)

