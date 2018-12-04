from jits.node import Node
from jits.car import Car

# Node that exists on the edge of the system. From this point cars enter/leave the environment.
# Connects to one intersection
class BorderNode(Node):
	def __init__(self, name):
		super().__init__(name, "border")
		self.connection = None

	def set_connection(self, neighbour):
		self.connection = neighbour

	def spawn_car(self, path):
		car = Car(path)
		self.connection.push_to_queue(car)

	def __str__(self):
		return "Border node: " + super().__str__()
