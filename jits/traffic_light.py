
from random import choice

class TrafficLight:
	maxQueueSize = 30
	def __init__(self):
		# False = red, True = green
		self.lights = [False, False, False]  # left, straigh ahead, right


	def update(self, setup):
		for i in range(3):
			if setup[i] == 0:
				self.lights[i] = False
			elif setup[i] == 1:
				self.lights[i] = True
			else:
				print("At traffic_light.update: given setup is wrong")

	def random_update(self):
		self.lights = [choice([True, False]), choice([True, False]), choice([True, False])]

	def green(self, origin, destination):
		direction = self.direction(origin, destination)
		if direction == -1:
			return None
		return self.lights[direction]

	def red(self, origin, destination):
		if self.green(origin, destination) is None:
			return None
		elif self.green(origin, destination):
			return False
		return True

	@staticmethod
	def direction(origin, destination):
		direction = 0
		for i in range(origin + 1, origin + 4):
			if (i % 4) == destination:
				return direction
			direction += 1
		print("traffic_light.direction failed. Origin: {0}, destination: {1}".format(origin, destination))
		return -1

		
