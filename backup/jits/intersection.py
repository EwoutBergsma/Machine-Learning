from node import Node
from car_queue import CarQueue


# Intersection node, connects to four other nodes in each direction
class Intersection(Node):
	def __init__(self, name, max_q_size):
		super().__init__(name, "intersection")
		self.neighbours = [None, None, None, None]  # north, east, south, west
		self.qs = [CarQueue(max_q_size), CarQueue(max_q_size), CarQueue(max_q_size), CarQueue(max_q_size)]

	def set_connections(self, north, east, south, west):
		self.neighbours = [north, east, south, west]

	def push_to_queue(self, origin, car):
		direction = self.get_direction(origin)
		if self.qs[direction].add_car(car):
			return True
		else:  # the queue of the direction it wants to go to is full
			return False

	def get_direction(self, origin):
		for i in range(4):
			if self.neighbours[i] == origin:
				return i
		print("Origin {0} is not a neighbouring node of {1}".format(origin, self))
		return None

	def update(self, time_step):
		for q in self.qs:
			self.move_car(q, time_step)

	def move_car(self, q, time_step):
		car = q.get_car()  # pop car from queue
		if car is not None:  # there was a car in the queue
			if time_step == car.get_last_move():
				if not q.add_car_back(car):  # car is added back to queue
					print("CAR COULD NOT BE ADDED BACK TO QUEUE")
			else:
				direction = car.get_direction(time_step)  # first direction of the car
				if not self.neighbours[direction].transfer_car(self, car):
					# car has already moved or car could not be moved towards its direction
					car.put_direction_back(direction)
					if not q.add_car_back(car):  # car is added back to queue
						print("CAR COULD NOT BE ADDED BACK TO QUEUE")

	def count_cars_at(self, origin_str):
		for index in range(len((self.neighbours))):
			if self.neighbours[index].name == origin_str:
				return self.qs[index].number_of_cars()


	def number_of_cars(self):
		cars = 0
		for q in self.qs:
			cars += q.number_of_cars()
		return cars

	def __str__(self):
		return "Intersection: " + super().__str__() + " has {0} cars".format(self.number_of_cars())
