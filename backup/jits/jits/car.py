from queue import Queue
from random import randint


class Car:
	added_cars = 0
	removed_cars = 0

	def __init__(self, path):
		Car.added_cars += 1
		self.path_q = Queue()
		self.make_queue(path)

	def make_queue(self, path):
		for direction in path:
			self.path_q.put(direction)

	def get_number_of_cars(self):
		return "{0} cars were added to the system, {1} cars have left the system".format(self.added_cars, self.removed_cars)

	def get_direction(self):
		if self.path_q.empty():
			return randint(0, 3)
		return self.path_q.get()

