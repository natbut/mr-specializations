import argparse
import copy
import os
import socket
import struct
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
    
    TASK_COMP_RANGE = 0.075 # scaled dims

    def __init__(self, id):
        """Planning & comms coordination agent for Passenger robots."""

        super().__init__(id)

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

        # Initialize other agents' and obstacles pos observations
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
        
        # Convert obs to tensors
        tasks_tensor = torch.tensor(list(self.scaled_obs["tasks_pos"].values()))
        obsts_tensor = torch.tensor(list(self.scaled_obs["obstacles_pos"].values()))
        agents_tensor = torch.tensor(list(self.scaled_obs["agents_pos"].values()))

        # Compute feature values
        task_obs = self._get_dist_feature_value(tasks_tensor)
        obst_obs = self._get_dist_feature_value(obsts_tensor)
        agent_obs = self._get_dist_feature_value(agents_tensor)
        mother_obs = self._get_dist_feature_value(torch.tensor(self.scaled_obs["mother_pos"]))
        exploration_obs = 1.0 # assumes all cells are explored
        frontiers_obs = 0.0 # assumes all cells are explored

        obs_vec = [task_obs,
                   obst_obs,
                   agent_obs,
                   frontiers_obs,
                   exploration_obs,
                   mother_obs,
                   self.my_location[0],
                   self.my_location[1]
                   ]
        
        print("Cell obs vector:", obs_vec)
        
        self.prepare_message("obs_vec", 0, obs_vec)


    def send_location_obs_message(self):
        """
        Send out my location (scaled)
        """
        loc = (self.my_id, self.my_location)
        
        for a_id in range(1, self.num_passengers+1):
            if a_id != self.my_id:
                self.prepare_message("agent_pos", a_id, loc)
        self.prepare_message("agent_pos", 0, loc)


    def send_completed_task_message(self):
        """
        Send message when agent has completed a task.

        Store in file format to be sent out by acoustic modems.
        """

        # Process "task completion" & get task_id
        completed = []
        for t_id in self.scaled_obs["tasks_pos"]:
            t_pos = self.scaled_obs["tasks_pos"][t_id]
            # If agent has reached task, mark task complete & send messages
            diff = [[t_pos[0]-self.my_location[0]], 
            [t_pos[1]-self.my_location[1]]]
            print("Task pos:", t_pos, "Passenger pos:", self.my_location)
            print("!!! Task proximity:", torch.norm(torch.tensor(diff), dim=0))
            if torch.norm(torch.tensor(diff), dim=0) < self.TASK_COMP_RANGE:
                for a_id in range(1, self.num_passengers+1):
                    if a_id != self.my_id:
                        self.prepare_message("completed_task", a_id, t_id)
                self.prepare_message("completed_task", 0, t_id)
                completed.append(t_id)
                
        for t_id in completed:
            self.scaled_obs["tasks_pos"].pop(t_id) # remove from own obs


    def create_plan(self):
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
        plan_path = os.path.join(self.logs_fp, plan_name)
        os.makedirs(os.path.dirname(plan_path), exist_ok=True)
        save_waypoints_to_file(latlon_plan, plan_path)
        self.num_plans += 1
        
        self._visualize_plan(latlon_plan, plan_path)

    
    def _visualize_plan(self, plan, log_fp):
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

        plt.xlabel('Lon')
        plt.ylabel('Lat')
        plt.title(f'Passenger {self.my_id} Plan Visualization')
        plt.xlim(self.env_lon_min, self.env_lon_max)
        plt.ylim(self.env_lat_max, self.env_lat_min)
        plt.legend()
        # plt.gca().invert_yaxis()
        plt.grid(True)
        plt.savefig(log_fp.replace(".txt", ".png"))
        plt.show()
        plt.close()


    def _get_dist_feature_value(self, feature: torch.tensor):
        """ Compute distance feature value. Is 1.0 if feature is in same cell as agent. """
        
        # Return 0.0 if feature tensor is empty
        if feature.numel() == 0:
            return 0.0

        my_loc_tensor = torch.tensor(self.my_location).unsqueeze(0)  # Shape (1, 2)
        
        dists = torch.norm(my_loc_tensor - feature, dim=-1)
        
        min_dist, _ = dists.min(dim=-1)

        if (dists < self.discrete_resolution/2).any(dim=-1):
            return 1.0
        else:
            return torch.clamp(1.0 - (min_dist/self.scaled_max_dist), min=0.0, max=self.scaled_max_dist).item()
        

    def update_location(self, lat=None, lon=None):
        """Update own current position from lat lon"""
        
        self.my_location = self.latlon_to_scaled(lat, lon)
        


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


base_ports = {
        "plan": 10000,
        "update": 11000,
        "coordinate": 12000
    }

def planning_listener(robot_id):
    global planning_trigger
    s = socket.socket()
    print("Starting planning listener on port:",  base_ports["plan"]+robot_id)
    s.bind(('localhost', base_ports["plan"]+robot_id))
    s.listen(1)
    while True:
        conn, _ = s.accept()
        planning_trigger = True
        conn.close()

def update_listener(robot_id):
    global update_trigger
    global passenger_latlon
    s = socket.socket()
    print("Starting update listener on port:",  base_ports["update"]+robot_id)
    s.bind(('localhost', base_ports["update"]+robot_id))
    s.listen(1)
    while True:
        conn, _ = s.accept()
        update_trigger = True
        data = conn.recv(1024)
        passenger_latlon = struct.unpack(f'<{2}f', data)
        print("Listener recv:", passenger_latlon)
        conn.close()

if __name__ == "__main__":
    """
    Launches Passenger agent, which will:
    - Automatically read new message files
    - Create new plan when python trigger_passenger.py is ran
    """

    planning_trigger = False
    update_trigger = False
    passenger_latlon = ()

    parser = argparse.ArgumentParser(description="Run simulated hardware agents")
    parser.add_argument("--config_fp", type=str, required=True, help="Path to problem config file")
    parser.add_argument("--robot_id", type=int, default=1, help="Passenger ID")
    parser.add_argument("--sim_comms", type=bool, default=False, help="Dummy comms bool. Defaults to False (no simulated comms)")

    args = parser.parse_args()

    # Create agent
    passenger = Passenger(args.robot_id)
    passenger.load_deployment_config(args.config_fp)

    # Trigger initialization
    threading.Thread(target=planning_listener, 
                     args=(args.robot_id,), 
                     daemon=True).start()
    threading.Thread(target=update_listener, 
                     args=(args.robot_id,), 
                     daemon=True).start()

    # Action loop
    while True:

        # Prepare new messages to send (periodically)
        if update_trigger:
            print("Updating triggered")
            update_trigger = False
            if len(passenger_latlon) > 0:
                passenger.update_location(passenger_latlon[0], passenger_latlon[1])
                passenger_latlon = ()
            passenger.send_location_obs_message()
            passenger.send_completed_task_message()
            passenger.send_cell_obs_message()
        else:
            print("Update socket waiting...")

        # Process any recieved messages
        passenger.receive_messages()

        # Process planning commands
        if planning_trigger:
            print("Planning triggered")
            planning_trigger = False
            passenger.create_plan() # create and save new plan
        else:
            print("Planning socket waiting...")
            
        # Simulate message sending if enabled
        if args.sim_comms:
            passenger.dummy_send_messages() 
        
        time.sleep(1)

