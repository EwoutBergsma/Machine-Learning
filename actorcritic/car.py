from collections import deque
from random import randint


class Car:
	added_cars = 0
	removed_cars = 0
	random_direction = 0

	def __init__(self, path):

		Car.added_cars += 1
		self.path_q = deque()
		self.make_queue(path)
		self.last_move = -1

		self.initial_path = path
		self.total_moves = 0
		self.waiting_time = 0

	def make_queue(self, path):
		for direction in path:
			self.path_q.appendleft(direction)

	def get_number_of_cars(self):
		return [self.added_cars, self.removed_cars]

	def get_direction(self, time_step):
		self.last_move = time_step  # store the last time a car has moved so it won't move until next time step
		if len(self.path_q) == 0:
			Car.random_direction += 1
			return randint(0, 3)
		direction = self.path_q.pop()
		return direction

	def get_last_move(self):
		return self.last_move

	def put_direction_back(self, direction):
		self.path_q.append(direction)

	def reset_waiting_time(self):
		self.waiting_time = 0

	def increment_waiting_time(self):
		self.waiting_time += 1

	def get_waiting_time(self):
		return self.waiting_time