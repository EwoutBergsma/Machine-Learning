from jits.node import Node
from jits.car_queue import CarQueue


# Intersection node, connects to four other nodes in each direction
class Intersection(Node):
	def __init__(self, name, max_q_size):
		super().__init__(name, "intersection")
		self.neighbours = [None, None, None, None]  # north, east, south, west
		self.qs = [CarQueue(max_q_size), CarQueue(max_q_size), CarQueue(max_q_size), CarQueue(max_q_size)]

	def set_connections(self, north, east, south, west):
		self.neighbours = [north, east, south, west]

	def push_to_queue(self, origin, car):
		dir = self.get_direction(origin)
		if self.qs[dir].add_car(car):
			return True
		else:  # the queue of the direction it wants to go to is full
			return False

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
		if car is not None:
			dir = car.get_direction()
			if not self.neighbours[dir].transfer_car(self, car):
				if not q.add_car_back(car):
					print("CAR COULD NOT BE ADDED BACK TO QUEUE")

	def number_of_cars(self):
		cars = 0
		for q in self.qs:
			cars += q.number_of_cars()
		return cars

	def __str__(self):
		return "Intersection: " + super().__str__()
