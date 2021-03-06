from intersection import Intersection
from border_node import BorderNode
from random import randint

# Map class, contains all nodes and connections between them.
class Map:
	def __init__(self, max_q_size):
		border_names = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
		intersection_names = ["A", "B", "C", "D"]
		self.intersections = []
		self.borders = []
		for name in border_names:
			self.borders.append(BorderNode(name))
		for name in intersection_names:
			self.intersections.append(Intersection(name, max_q_size))
		self.set_connections()

	def set_connections(self):
		self.connect_intersection("A", "I", "B", "C", "III")
		self.connect_intersection("B", "II", "IV", "D", "A")
		self.connect_intersection("C", "A", "D", "VII", "V")
		self.connect_intersection("D", "B", "VI", "VIII", "C")
		self.connect_border("I", "A")
		self.connect_border("II", "B")
		self.connect_border("III", "A")
		self.connect_border("IV", "B")
		self.connect_border("V", "C")
		self.connect_border("VI", "D")
		self.connect_border("VII", "C")
		self.connect_border("VIII", "D")

	def connect_intersection(self, origin, north, east, south, west):
		self.get_node_at(origin).set_connections(
			self.get_node_at(north), self.get_node_at(east), self.get_node_at(south), self.get_node_at(west))

	def connect_border(self, origin, destination):
		self.get_node_at(origin).set_connection(self.get_node_at(destination))

	def get_node_at(self, name):
		for intersection in self.intersections:
			if intersection.name == name:
				return intersection
		for border in self.borders:
			if border.name == name:
				return border
		return None

	def update_cars(self, time_step):
		for intersection in self.intersections:
			intersection.update(time_step)
		for border in self.borders:
			border.update(time_step)

	def get_index(self, path_key):
		border_names = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
		index = 0
		for border_name in border_names:
			if border_name == path_key:
				return index
			index += 1

	def spawn_car(self, path_key, path):
		starting_point = path_key.split(":")[0]
		index = self.get_index(starting_point)
		self.borders[index].spawn_car(path)

	def number_of_cars(self):
		cars = 0
		for intersection in self.intersections:
			cars += intersection.number_of_cars()
		for border in self.borders:
			cars += border.number_of_cars()
		return cars

	def print_status(self):
		for intersection in self.intersections:
			print(intersection)
		for border in self.borders:
			print(border)

	def cars_at(self, origin_str, destination_str):
		for intersection in self.intersections:
			if intersection.name == destination_str:
				cars = intersection.count_cars_at(origin_str)
				if cars < 1000:
					if cars < 100:
						if cars < 10:
							return "00" + str(cars)
						return "0" + str(cars)
					return cars
				return 999
		if origin_str == destination_str:
			for border in self.borders:
				if border.name == destination_str:
					cars = border.number_of_cars()
					if cars < 1000:
						if cars < 100:
							if cars < 10:
								return "00" + str(cars)
							return "0" + str(cars)
						return cars
					return 999
		return "000"


	def display_map(self):
		print("        {0}  000       {1}  000        ".format(self.cars_at("I", "I"), self.cars_at("II", "II")))
		print("        {0}  000       {1}  000        ".format(self.cars_at("I", "A"), self.cars_at("A", "I")))
		print("000|000           {0}          {1}|{2}".format(self.cars_at("B", "A"), self.cars_at("IV", "B"), self.cars_at("IV", "IV")))
		print("{0}|{1}           {2}          000|000".format(self.cars_at("III", "III"), self.cars_at("III", "A"), self.cars_at("A", "B")))
		print("        {0}  {1}       {2}  {3}      ".format(self.cars_at("A", "C"), self.cars_at("C", "A"), self.cars_at("B", "D"), self.cars_at("D", "B")))
		print("000|000           {0}          {1}|{2}".format(self.cars_at("D", "C"), self.cars_at("VI", "D"), self.cars_at("VI", "VI")))
		print("{0}|{1}           {2}          000|000".format(self.cars_at("V", "V"), self.cars_at("V", "C"), self.cars_at("C", "D")))
		print("        000  {0}       000  {1}       ".format(self.cars_at("VII", "C"), self.cars_at("VIII", "D")))
		print("        000  {0}       000  {1}       ".format(self.cars_at("VII", "VII"), self.cars_at("VIII", "VIII")))
		print("------------------------------------")
