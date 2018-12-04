class Node:
	def __init__(self, name, type):
		self.name = name
		self.type = type

	def is_intersection(self):
		if self.type == "intersection":
			return True
		return False

	def is_border(self):
		if self.type == "border":
			return True
		return False

	def get_type(self):
		return self.type

	def transfer_car(self, origin, car):
		# print("movement from {0} to {1}".format(origin.name, self.name))
		if self.type == "intersection":
			return self.push_to_queue(origin, car)
		if self.type == "border":
			return self.car_leaves_system(car)

	def __str__(self):
		return self.name
