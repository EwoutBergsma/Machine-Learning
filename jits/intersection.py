from jits.node import Node
from jits.car_queue import CarQueue


# Intersection node, connects to four other nodes in each direction
class Intersection(Node):
	def __init__(self, name):
		super().__init__(name, "intersection")
		self.neighbours = [None, None, None, None]  # north, east, south, west
		self.qs = [CarQueue(), CarQueue(), CarQueue(), CarQueue()]  # north, east, south, west

	def set_connections(self, north, east, south, west):
		self.neighbours = [north, east, south, west]

	def push_to_queue(self, origin, car):
		dir = self.get_direction(origin)
		self.qs[dir].add_car(car)

	def get_direction(self, origin):
		for i in range(4):
			if self.neighbours[i] == origin:
				return i
		print("Origin {0} is not a neighbouring node of {1}".format(origin, self))
		return None

	def update(self):
		for q in self.qs:
			self.move_car(q)

	def move_car(self, q):
		car = q.get_car()
		if not car is None:
			dir = car.get_direction()
			self.neighbours[dir].transfer_car(self, car)

	def __str__(self):
		return "Intersection: " + super().__str__()
