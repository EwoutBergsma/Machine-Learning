from jits.node import Node
from jits.car import Car
from jits.car_queue import CarQueue


# Node that exists on the edge of the system. From this point cars enter/leave the environment.
# Connects to one intersection
class BorderNode(Node):
	def __init__(self, name):
		super().__init__(name, "border")
		self.connection = None
		self.q = CarQueue(-1)  # -1 for no maximum q-size

	def set_connection(self, neighbour):
		self.connection = neighbour

	def spawn_car(self, path):
		self.q.add_car(Car(path))

	def update(self):
		car = self.q.get_car()
		if car is not None:
			self.connection.transfer_car(self, car)

	def number_of_cars(self):
		return self.q.number_of_cars()

	@staticmethod
	def car_leaves_system(car):
		Car.removed_cars += 1
		return True

	def __str__(self):
		return "Border node: " + super().__str__()
