class State:
	"""State of one intersection after a completed timestep. 
	This includes state of traffic lights and amount of cars waiting in each lane together with
	their accumulative waiting time"""
	def __init__(self, intersection):
		#configuration of traffic lights during the current time step
		self.trafficLights = []
		#12 numbers indicating the accumulative waiting time of the cars of each lane
		self.laneWaitingTimes = []
		#the reward that was granted from setting the current configuration of traffic lights
		self.currentReward = 0;

		#self.state_list = []

	def get_state(self):
		return self.state_list
