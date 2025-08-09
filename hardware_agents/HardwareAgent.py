import os

import yaml


class HardwareAgent():

    def __init__(self, id):
        
        self.my_id = id
        
        # Scaling parameters for converting from [-1,1] sim env scale to real coordinates
        self.env_lat_max = None
        self.env_lat_min = None
        self.env_lon_max = None
        self.env_lon_min = None
        self.lat_scale = None
        self.lat_offset = None
        self.lon_scale = None
        self.lon_offset = None

        self.my_latlon = [] # Real location in lat/lon
        self.my_location = [] # Location scaled to [-1,1]
        
        self.num_passengers = 0
        
        self.scaled_obs = {} # Observations scaled to [-1, 1]

        self.discrete_resolution = 0
        self.scaled_max_dist = 2.0
        
        # Filepaths to message tx/rx locations
        self.messaging_fp = ""
        self.msg_tx_fp = "" # outgoing message buffer
        self.msg_rx_fp = "" # incoming message buffer
        self.msg_tx_done_fp = "" # completed sent message storage
        self.msg_rx_done_fp = "" # completed received message storage
        self.tx_ct = 0
        self.rx_ct = 0


    def load_deployment_config(self, config_fp):
        """Load in deployment configuration details from config file."""

        with open(config_fp, 'r') as file:
            params = yaml.safe_load(file)

            self.env_lat_max = params["env_lat_max"]
            self.env_lat_min = params["env_lat_min"]
            self.env_lon_max = params["env_lon_max"]
            self.env_lon_min = params["env_lon_min"]
            
            self.num_passengers = params["num_passengers"]
            self.messaging_fp = params["messaging_fp"]
            logs_fp = params["logs_fp"]

            mothership_lat = params["mothership_lat"]
            mothership_lon = params["mothership_lon"]

            tasks_latlon = params["task_locs_latlon"]

        # Set up scaling parameters
        self.init_env_scaling()

        # Create scaled observations
        self.scaled_obs["mother_pos"] = self.latlon_to_scaled(mothership_lat, mothership_lon)
        tasks = {}
        for i, pos in enumerate(tasks_latlon):
            tasks[i] = self.latlon_to_scaled(pos[0], pos[1])
        self.scaled_obs["tasks_pos"] = tasks

        print(f"Init scaled mother_pos:", self.scaled_obs["mother_pos"])
        print(f"Init scaled tasks_pos:", self.scaled_obs["tasks_pos"])
        
        # Prepare messaging paths
        self.msg_tx_fp = self.messaging_fp+f"agent_{self.my_id}_tx"
        self.msg_rx_fp = self.messaging_fp+f"agent_{self.my_id}_rx"
        self.msg_tx_done_fp = self.messaging_fp+f"agent_{self.my_id}_tx_done"
        self.msg_rx_done_fp = self.messaging_fp+f"agent_{self.my_id}_rx_done"
        
        os.makedirs(self.msg_tx_fp, exist_ok=True)
        os.makedirs(self.msg_rx_fp, exist_ok=True)
        os.makedirs(self.msg_tx_done_fp, exist_ok=True)
        os.makedirs(self.msg_rx_done_fp, exist_ok=True)
        
        # Logging folder path
        self.logs_fp=logs_fp+"_"+str(self.my_id)

    
    def init_env_scaling(self):
        """Set scaling parameters for converting between sim dims and real"""

        self.lat_scale = 2.0 / (self.env_lat_max - self.env_lat_min)
        self.lat_offset = -1.0 - self.env_lat_min * self.lat_scale
        self.lon_scale = 2.0 / (self.env_lon_max - self.env_lon_min)
        self.lon_offset = -1.0 - self.env_lon_min * self.lon_scale

    def latlon_to_scaled(self, lat, lon):
        scaled_y = round(self.lat_scale * lat + self.lat_offset, 4)
        scaled_x = round(self.lon_scale * lon + self.lon_offset, 4)
        return [scaled_y, scaled_x]

    def scaled_to_latlon(self, scaled_lat, scaled_lon):
        lat = (scaled_lat - self.lat_offset) / self.lat_scale
        lon = (scaled_lon - self.lon_offset) / self.lon_scale
        return [lat, lon]


    def prepare_message(self, msg_type, target_id, contents):
        """
        Prepare contents to be transmitted by comms modems.

        Store in txt file format & in tx folder to be sent out by acoustic modems.
        
        Args:
            - msg_type (str): agent_pos, completed_task, spec_params, task_pos,
            - target_id (int): id of target entity
            - contents: (dict_id, content)            
        """
        # Create write contents
        text = msg_type+"|"+str(target_id)+"|"+str(contents)
        print("Prepared message text:", text)
        
        # Write to file
        msg_fp = os.path.join(self.msg_tx_fp, f"msg_{self.tx_ct}.txt")
        self.tx_ct += 1
        with open(msg_fp, "w") as file:
            file.write(text)
        
        print("Message saved at", msg_fp)


    def receive_messages(self):
        """
        Process messages in received messages (rx) folder.
        """
        if not os.path.exists(self.msg_rx_fp) or not os.listdir(self.msg_rx_fp):
            print("No messages received yet.")
            return
        
        # For each message in self.msg_rx_fp
        fps = []
        for msg_fp in os.listdir(self.msg_rx_fp):
            with open(os.path.join(self.msg_rx_fp, msg_fp), "r") as file:
                msg = file.read().strip().split("|")
                # Skip messages not meant for this agent
                if int(msg[1]) == self.my_id:
                    
                    print("Processing message", msg)
                    self.process_messages(msg[0],msg[-1])
            fps.append(msg_fp)
            
        for msg_fp in fps:
            # Move processed message to done folder
            done_fp = os.path.join(self.msg_rx_done_fp, msg_fp)
            os.rename(os.path.join(self.msg_rx_fp, msg_fp), done_fp)
        print("Messages processed, moved to done folder")


    def dummy_send_message(self, message):
        """
        Saves message to receiving agent's rx file location.
        """
        pass

    
    def process_messages(self, msg_type, msg):
        """Handle messages that apply both to Passengers and Mothership"""

        if "agent_pos" == msg_type: # Agent position updates
            # msg = (msg_type} {agent_id, agent_pos) MESSAGE FORMAT
            # scaled_obs[agents_pos] = {agent_id: agent_pos} STORED OBS FORMAT
            self.scaled_obs["agents_pos"][msg[0]] = msg[1]
        elif "completed_task" == msg_type:
            # msg = (msg_type, task_id)
            if self.scaled_obs["tasks_pos"][msg[0]]:
                self.scaled_obs["tasks_pos"].pop(msg[0])
        elif "new_task" == msg_type:
                # msg = (msg_type, task_id, task_pos) MESSAGE FORMAT
                # scaled_obs[tasks_pos] = {task_id: task_pos} STORED OBS FORMAT
                self.scaled_obs["tasks_pos"][msg[0]] = msg[1]
        elif "spec_params" == msg_type:
            # msg = (msg_type, agent_id, spec_params)
            if self.my_id == msg[0]:
                self.my_specializations = msg[1]
        elif "obs_vec" == msg_type:
                # msg = (msg_type, obs_vec)
                obs_vec = msg[1]
                self._update_env_cell(obs_vec)
                
                
    def _update_env_cell(self, obs_vec):
        """
        Creates or updates environment cell with observation vector information.
        
        Selects environment cell nearest to feature location
        """

        # TODO cell key should be nearest discretized position to pos in obs_vec (last 2 elements)
        
        # TODO cell value should be features in obs_vec

        pass
                

