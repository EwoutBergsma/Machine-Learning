from collections import deque


class CarQueue:
	def __init__(self, max_size):
		if max_size >= 0:
			self.q = deque(maxlen=max_size)
		else:
			self.q = deque()

	def add_car(self, car):
		if self.q.maxlen is None or len(self.q) < self.q.maxlen:
			self.q.appendleft(car)
			return True
		return False

	def add_car_back(self, car):
		if self.q.maxlen is None or len(self.q) < self.q.maxlen:
			self.q.append(car)
			return True
		return False

	def get_car(self):
		if len(self.q) == 0:
			return None
		car = self.q.pop()
		return car

	def number_of_cars(self):
		return len(self.q)

	def iterate_queue(self):
		total_waiting_time = 0
		for car in self.q:
			waiting_time = car.get_waiting_time()
			total_waiting_time += waiting_time
		return total_waiting_time

	def get_direction_amounts(self, time_step, x, y):
		directions = []
		direction_amounts = [0,0,0,0]
		for car in self.q:
			direction = car.get_directions(time_step, x, y)[0]
			directions.append(direction)

		# count each direction
		for direction in directions:
			if direction == 0:
				direction_amounts[0] += 1
			elif direction == 1:
				direction_amounts[1] += 1
			elif direction == 2:
				direction_amounts[2] += 1
			elif direction == 3:
				direction_amounts[3] += 1

		# Direction amounts is a list containing for n,e,s,w how many cars are waiting for that direction in the queue
		# Can be used to determine which light should be put on green
		print(direction_amounts)
		return direction_amounts




