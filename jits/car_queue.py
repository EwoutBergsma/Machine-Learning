from queue import Queue


class CarQueue:
	def __init__(self):
		self.q = Queue(maxsize=20)

	def add_car(self, car):
		self.q.put(car)

	def get_car(self):
		car = self.q.get()
		return car

