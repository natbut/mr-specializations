import argparse
import os
import socket
import sys
import threading
import time

import matplotlib.pyplot as plt
import torch
from HardwareAgent import *

# Add parent directory to sys.path to access heuristics module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import envs.heuristics
from agents.planning_agent import compute_traj_rrt_goals


def load_func(dotpath : str):
    """ load function in module.  function is right-most segment """
    func = dotpath #dotpath.rsplit(".", maxsplit=1)
    m = envs.heuristics #(module_)
    # print("Func:", func, "m:", m)
    return getattr(m, func)


class Passenger(HardwareAgent):

    def __init__(self, id):
        """Planning & comms coordination agent for Passenger robots."""

        super().__init__()
        
        
        self.my_id = id

        self.my_specializations = [] # Specialization parameters from Mothership
        self.heuristic_fns = [] # Functions used to evaluate plan heuristics

        self.plan_horizon: 0.0 # Planning horizon
        self.sampling_pts: 0 # Number of points to sample by planner
        self.plan_goal_samp_freq: 0.0 # Goal point sampling frequency
        self.plan_step_size: 0.0 # Plan step size
        
        self.num_plans = 0 # counter for number of plans created


    def load_deployment_config(self, config_fp):
        """
        Load in heuristic evaluation functions for planner, set initial specialization

        Function is called when class is created.
        """
        super().load_deployment_config(config_fp)

        with open(config_fp, 'r') as file:
            params = yaml.safe_load(file)

            self.heuristic_fns = [load_func(name) for name in params["heuristic_fns"]]
            self.my_specializations = [0.0 for _ in self.heuristic_fns]

            self.plan_horizon = params.get("plan_horizon", 0.25)
            self.sampling_pts = params.get("sampling_pts", 50)
            self.plan_goal_samp_freq = params.get("plan_goal_samp_freq", 3)
            self.plan_step_size = params.get("plan_step_size", 0.1)

            obstacles_latlon = params["obstacles_locs_latlon"]
            
            num_passengers = params["num_passengers"]

        # Initialize other agents and obstacles pos observations
        # Initially assume at mothership
        agents = {}
        for i in range(num_passengers-1): # only store positions for other agents
            agents[i] = copy.deepcopy(self.scaled_obs["mother_pos"])
        self.scaled_obs["agents_pos"] = agents

        osbtacles = {}
        for i, pos in enumerate(obstacles_latlon):
            osbtacles[i] = self.latlon_to_scaled(pos[0], pos[1])
        self.scaled_obs["obstacles_pos"] = osbtacles

        print(f"Init scaled agents_pos:", self.scaled_obs["agents_pos"])
        print(f"Init scaled obstacles_pos:", self.scaled_obs["obstacles_pos"])

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
        
        # TODO configure message (.txt file)
        msg = None # contain (msg_type, obs_vec)

        self.prepare_message(msg)


    def send_location_obs_message(self):
        """
        Send out my location (scaled)
        
        Store in file format to be sent out by acoustic modems.
        """

        locs = self.my_location

        # TODO configure message (.txt file)
        msg = None # contain (msg_type, agent_id, agent_pos)

        self.prepare_message(msg)


    def send_completed_task_message(self):
        """
        Send message when agent has completed a task.

        Store in file format to be sent out by acoustic modems.
        """

        # TODO: Process "task completion"
        
        # TODO configure message (.txt file)

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


    def create_plan(self, folder_fp: str):
        """
        Creates short-horizon plan using stored data.

        Stores plan in [FORMAT] file for upload to autopilot.       
        """
        # Set up planning observations
        obs = {}
        tasks = []
        for t in self.scaled_obs["tasks_pos"]:
            tasks.append(self.scaled_obs["tasks_pos"][t])

        agents = []
        for a in self.scaled_obs["agents_pos"]:
            agents.append(self.scaled_obs["agents_pos"][a])

        obstacles = []
        for o in self.scaled_obs["obstacles_pos"]:
            obstacles.append(self.scaled_obs["obstacles_pos"][o])

        
        obs["obs_tasks"] = torch.tensor([tasks])
        obs["obs_agents"] = torch.tensor([agents])
        obs["obs_obstacles"] = torch.tensor([obstacles])
        obs["obs_base"] = torch.tensor([self.scaled_obs["mother_pos"]])

        current_pos = torch.tensor([self.my_location])
        heuristic_weights = torch.tensor([self.my_specializations])
        heuristic_weights[0][0] = 1.0 # TODO remove this

        print("Nested obs:", obs)
        print("Heuristic weights:", heuristic_weights)
        print("Current pos: ", current_pos)

        # Create plan
        plan = compute_traj_rrt_goals(obs,
                                      0,
                                      current_pos,
                                      heuristic_weights,
                                      self.heuristic_fns,
                                      horizon=self.plan_horizon,
                                      max_pts=self.sampling_pts,
                                      goal_samp_freq=self.plan_goal_samp_freq,
                                      step_size=self.plan_step_size,
                                      verbose=True
                                      )
        
        plan = [p.tolist() for p in plan]
        latlon_plan = [self.scaled_to_latlon(p[0], p[1]) for p in plan]
        
        # Save to file & visualize
        print(f"Passenger {self.my_id} processed plan: {plan}")
        print("Latlon plan:", latlon_plan)
        plan_name =  "mission_plan_"+str(self.num_plans)+".txt"
        plan_path = os.path.join(folder_fp, plan_name)
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        save_waypoints_to_file(latlon_plan, plan_path)
        self.num_plans += 1
        
        self._visualize_plan(latlon_plan)

    
    def _visualize_plan(self, plan):
        """
        Plot environment tasks, obstacles, agents, and plan waypoints
        
        Plan is in scaled [lat (y), lon (x)]
        """

        plt.figure(figsize=(8, 6))

        # Plot tasks
        tasks = list(self.scaled_obs["tasks_pos"].values())
        tasks = [self.scaled_to_latlon(p[0], p[1]) for p in tasks]
        if tasks:
            tasks = torch.tensor(tasks).numpy()
            plt.scatter(tasks[:, 1], tasks[:, 0], c='green', label='Tasks', marker='o')

        # Plot obstacles
        obstacles = list(self.scaled_obs["obstacles_pos"].values())
        obstacles = [self.scaled_to_latlon(p[0], p[1]) for p in obstacles]
        if obstacles:
            obstacles = torch.tensor(obstacles).numpy()
            plt.scatter(obstacles[:, 1], obstacles[:, 0], c='red', label='Obstacles', marker='x')

        # Plot agents (others + my_location)
        agents = list(self.scaled_obs["agents_pos"].values())
        agents = [self.scaled_to_latlon(p[0], p[1]) for p in agents]
        agents.append(self.scaled_to_latlon(self.my_location[0], 
                                            self.my_location[1]
                                            )
                      )
        if agents:
            agents = torch.tensor(agents).numpy()
            plt.scatter(agents[:, 1], agents[:, 0], c='blue', label='Agents', marker='^')

        # Plot base/mothership
        mother = torch.tensor(self.scaled_to_latlon(self.scaled_obs["mother_pos"][0], self.scaled_obs["mother_pos"][1])).numpy()
        plt.scatter(mother[1], mother[0], c='purple', label='Base', marker='s')

        # Plot plan waypoints
        if plan is not None and len(plan) > 0:
            plan_np = torch.tensor(plan).numpy()
            plt.scatter(plan_np[:, 1], plan_np[:, 0], c='orange', label='Plan', marker='.')
            plt.plot(plan_np[:, 1], plan_np[:, 0], c='orange', linestyle='--', alpha=0.7)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Passenger {self.my_id} Plan Visualization')
        plt.legend()
        # plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()


    def _get_dist_feature_value(self, feature):
        """ Compute distance feature value. Is 1.0 if feature is in same cell as agent. """
        dists = torch.norm(self.my_location - feature, dim=-1)
        min_dist, _ = dists.min(dim=-1)

        if (dists < self.discrete_resolution/2).any(dim=-1):
            return 1.0
        else:
            return torch.clamp(1.0 - (min_dist/self.scaled_max_dist), min=0.0, max=self.scaled_max_dist)
        

    def _get_latlon(self):
        """Get position feature from MAVLink"""

        # TODO set up this feature

        pass

    def update_location(self, scaled_loc=None):
        """Update own current position"""

        # TODO set this up to use latlon from MAVLink
        if scaled_loc:
            self.my_location = scaled_loc
        else:
            self.my_location = copy.deepcopy(self.scaled_obs["mother_pos"])
            


def save_waypoints_to_file(waypoints, filename="mission_plan.txt"):
    """
    Saves the waypoints to a file in the format supported by QGroundControl and Mission Planner.
    
    Args:
    - waypoints (list): List of waypoints in the format [[lat1, lon1], [lat2, lon2], ...].
    - filename (str): The file name where the mission plan will be saved.
    """
    # Open file for writing
    with open(filename, "w") as file:
        # Write the header for the QGC WPL file
        file.write("QGC WPL 110\n")

        # Iterate over the waypoints to format and write them
        for idx, (lat, lon) in enumerate(waypoints):
            # The structure of each waypoint row
            # Format: <INDEX> <CURRENT WP> <COORD FRAME> <COMMAND> <PARAM1> <PARAM2> <PARAM3> <PARAM4> <PARAM5/X/LATITUDE> <PARAM6/Y/LONGITUDE> <PARAM7/Z/ALTITUDE> <AUTOCONTINUE>
            # We assume the following fixed values for most params:
            # - COMMAND: 16 (Navigation command)
            # - COORD FRAME: 0 (Global frame)
            # - PARAM1: 0.15 (This is a common parameter for waypoints)
            # - PARAM2, PARAM3, PARAM4: 0 (Typically 0 for these params in a standard waypoint)
            # - AUTOCONTINUE: 1 (Auto-continue flag)
            # For this example, we're using the latitude, longitude, and a fixed altitude of 550 meters
            file.write(f"{idx}\t1\t0\t16\t0.15\t0\t0\t0\t{lat:.10f}\t{lon:.10f}\t550\t1\n")

    print(f"Mission plan saved to {filename}")



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
    """
    Launches Passenger agent, which will:
    - Automatically read new message files
    - Create new plan when python trigger_passenger.py is ran
    """

    planning_trigger = False

    parser = argparse.ArgumentParser(description="Run simulated hardware agents")
    parser.add_argument("--config_fp", type=str, required=True, help="Path to problem config file")
    parser.add_argument("--logs_fp", type=str, required=True, help="Path to logs folder")
    parser.add_argument("--robot_id", type=int, default=0, help="Passenger ID")

    args = parser.parse_args()

    # Create agent
    passenger = Passenger(args.robot_id)
    passenger.load_deployment_config(args.config_fp)
    passenger.update_location()

    # Comms initialization (as needed)
    # TODO

    # Planning initialization
    threading.Thread(target=listener, daemon=True).start()

    # Action loop
    while True:

        # Prepare new messages to send (periodically)
        passenger.send_location_obs_message() # TODO get & send location
        passenger.send_completed_task_message() # TODO check completed tasks, send updates
        # passenger.send_cell_obs_message() # TODO obs_vec (maybe only when triggered)

        # Process any recieved messages
        # TODO: Check for new message files, update passenger properties (including specializations)
        location = [l + 0.002 for l in passenger.my_location]
        passenger.update_location(location)


        # Process planning commands
        if planning_trigger:
            print("Planning triggered")
            planning_trigger = False
            passenger.create_plan(folder_fp=args.logs_fp) # create and save new plan
        else:
            print("Planning socket waiting...")

        time.sleep(1)

