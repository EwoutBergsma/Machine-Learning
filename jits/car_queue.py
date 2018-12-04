from queue import Queue


class CarQueue:
	def __init__(self):
		self.q = Queue()

	def add_car(self, car):
		self.q.put(car)

	def get_car(self):
		if self.q.empty():
			return None
		car = self.q.get()
		return car

