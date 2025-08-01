import yaml
import copy

class HardwareAgent():

    def __init__(self):
        
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
        
        self.scaled_obs = {} # Observations scaled to [-1, 1]

        self.message_buffer = []
        self.received_message_buffer = []

        self.discrete_resolution = 0
        self.scaled_max_dist = 2.0


    def load_deployment_config(self, config_fp):
        """Load in deployment configuration details from config file."""

        with open(config_fp, 'r') as file:
            params = yaml.safe_load(file)

            self.env_lat_max = params["env_lat_max"]
            self.env_lat_min = params["env_lat_min"]
            self.env_lon_max = params["env_lon_max"]
            self.env_lon_min = params["env_lon_min"]

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

    
    def init_env_scaling(self):
        """Set scaling parameters for converting between sim dims and real"""

        self.lat_scale = 2.0 / (self.env_lat_max - self.env_lat_min)
        self.lat_offset = -1.0 - self.env_lat_min * self.lat_scale
        self.lon_scale = 2.0 / (self.env_lon_max - self.env_lon_min)
        self.lon_offset = -1.0 - self.env_lon_min * self.lon_scale

    def latlon_to_scaled(self, lat, lon):
        scaled_y = self.lat_scale * lat + self.lat_offset
        scaled_x = self.lon_scale * lon + self.lon_offset
        return [scaled_y, scaled_x]

    def scaled_to_latlon(self, scaled_lat, scaled_lon):
        lat = (scaled_lat - self.lat_offset) / self.lat_scale
        lon = (scaled_lon - self.lon_offset) / self.lon_scale
        return [lat, lon]


    def prepare_message(self, contents):
        """
        Prepare contents to be transmitted by comms modems.

        Store in file format to be sent out by acoustic modems.
        """

        # TODO

        pass


    def receive_message(self, message):
        """
        Process message received from comms modem.

        Adds message contents to received_message_buffer.
        """

        # TODO

        self.process_universal_messages()

    
    def process_universal_messages(self):
        """Handle messages that apply both to Passengers and Mothership"""

        for i, msg in enumerate(self.received_message_buffer):
            if "agent_pos" == msg[0]: # Agent position updates
                self.received_message_buffer.pop(i)
                # msg = (msg_type, agent_id, agent_pos) MESSAGE FORMAT
                # scaled_obs[agents_pos] = {agent_id: agent_pos} STORED OBS FORMAT
                self.scaled_obs["agents_pos"][msg[1]] = msg[2]
            if "completed_task" == msg[0]:
                self.received_message_buffer.pop(i)
                # msg = (msg_type, task_id)
                if self.scaled_obs["tasks_pos"][msg[1]]:
                    self.scaled_obs["tasks_pos"].pop(msg[1])
                

