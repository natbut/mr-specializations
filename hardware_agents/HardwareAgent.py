

class HardwareAgent():

    def __init__(self):
        
        # Scaling parameters for converting from [-1,1] sim env scale to real coordinates
        self.env_lat_max = None
        self.env_lat_min = None
        self.env_lon_max = None
        self.env_lon_min = None

        self.my_latlon = [] # Real location in lat/lon
        self.my_location = [] # Location scaled to [-1,1]
        
        self.scaled_obs = {} # Observations scaled to [-1, 1]

        self.message_buffer = []
        self.received_message_buffer = []

        self.discrete_resolution = 0
        self.scaled_max_dist = 2.0


    def load_deployment_config(self, config_file):
        """Load in deployment configuration details from config file."""

        pass

    
    def update_env_scaling(self,
                           lat_max: float,
                           lat_min: float, 
                           lon_max: float, 
                           lon_min: float
                           ):
        """Set scaling parameters for converting between sim dims and real"""
        self.env_lat_max = lat_max
        self.env_lat_min = lat_min
        self.env_lon_max = lon_max
        self.env_lon_min = lon_min


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
                

