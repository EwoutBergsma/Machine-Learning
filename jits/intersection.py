from jits.node import Node

# Intersection node, connects to four other nodes in each direction
class Intersection(Node):
	def __init__(self, name):
		super().__init__(name, "intersection")
		self.north_side = None
		self.east_side = None
		self.south_side = None
		self.west_side = None

	def set_connections(self, north, east, south, west):
		self.north_side = north
		self.east_side = east
		self.south_side = south
		self.west_side = west

	def push_to_queue(self, car):
		pass


	def __str__(self):
		return "Intersection: " + super().__str__()
