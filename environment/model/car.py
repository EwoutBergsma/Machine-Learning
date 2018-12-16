from collections import deque
from random import randint


class Car:
	added_cars = 0
	removed_cars = 0
	random_direction = 0


	def __init__(self, destination):
		Car.added_cars += 1
		self.last_move = -1

		self.dest_x = destination[0]
		self.dest_y = destination[1]

	       # self.initial_path = path
		self.total_moves = 0
		self.waiting_time = 0

	def make_queue(self, path):
		for direction in path:
			self.path_q.appendleft(direction)

	def get_number_of_cars(self):
		return [self.added_cars, self.removed_cars]

	def get_directions(self, time_step, x, y):
		self.last_move = time_step  # store the last time a car has moved so it won't move until next time step
		directions = []
		if x > self.dest_x:
			directions.append(3)
		elif x < self.dest_x:
			directions.append(1)

		if y > self.dest_y:
			directions.append(0)
		elif y < self.dest_y:
			directions.append(2)

		if len(directions) > 1 and randint(0, 1) == 1:
			# swap directions with p = 0.5
			a = directions[0]
			b = directions[1]
			directions = [b, a]
		return directions

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
