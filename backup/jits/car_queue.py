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

