from collections import deque
from random import randint


class Car:
	added_cars = 0
	removed_cars = 0

	def __init__(self, path):
		Car.added_cars += 1
		self.path_q = deque()
		self.make_queue(path)

	def make_queue(self, path):
		for direction in path:
			self.path_q.appendleft(direction)

	def get_number_of_cars(self):
		return [self.added_cars, self.removed_cars]

	def get_direction(self):
		if len(self.path_q) == 0:
			return randint(0, 3)
		return self.path_q.pop()

