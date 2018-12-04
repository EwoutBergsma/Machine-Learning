from jits.intersection import Intersection
from jits.border_node import BorderNode
from random import randint


class Map:
	def __init__(self):
		border_names = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
		intersection_names = ["A", "B", "C", "D"]
		self.intersections = []
		self.borders = []
		for name in border_names:
			self.borders.append(BorderNode(name))
		for name in intersection_names:
			self.intersections.append(Intersection(name))
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

	def spawn_car(self, path):
		index = randint(0, len(self.borders)-1)
		self.borders[index].spawn_car(path)