class Node:
	def __init__(self, name, type, x, y):
		self.name = name
		self.type = type
		self.x = x
		self.y = y

	def is_intersection(self):
		if self.type == "intersection":
			return True
		return False

	def is_border(self):
		if self.type == "border":
			return True
		return False

	def get_position(self):
		return [self.x, self.y]

	def get_type(self):
		return self.type

	def transfer_car(self, origin, car):
		if self.type == "intersection":
			return self.push_to_queue(origin, car)
		if self.type == "border":
			return self.car_leaves_system(car)

	def __str__(self):
		return self.name
