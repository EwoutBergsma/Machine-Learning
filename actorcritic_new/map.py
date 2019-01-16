from intersection import Intersection
from border_node import BorderNode
from nodes import border_data, intersection_data
from global_traffic_light_combinations import combinations
from random import choice, randint
from paths import path_dict, border_names
from car import Car

import numpy as np
# Map class, contains all nodes and connections between them.
class Map:
	def __init__(self, max_q_size):

		self.intersections = []
		self.borders = []
		for border in border_data:
			self.borders.append(BorderNode(border[0], border[1], border[2]))
		for intersection in intersection_data:
			self.intersections.append(Intersection(intersection[0], max_q_size, intersection[1], intersection[2]))
		self.set_connections()
		self.global_state = []
		self.global_reward = 0

		# self.action_space = combinations
		# self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		# self.observation_space = 
		#self.action_size = 7
		self.action_size = 210
		#self.state_size = 8
		self.state_size = 32
		#self.state_size = 96


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
			intersection.update_cars(time_step)
		for border in self.borders:
			border.update_cars(time_step)

	def update_traffic_lights(self,action):
		#action = choice(combinations)
		for index,intersection in enumerate(self.intersections):
			#single_action = choice(combinations)
			#print("SINGLE ACTION: ")
			#print(single_action)
			intersection.update_traffic_lights(action[index])
			#action.append(single_action)

		#self.step(action)

	def time_step(self):
		n_cars = 4
		for c in range(n_cars):
			start = choice(border_names)
			end = choice(border_names)
			while start == end:
				end = choice(border_names)
			self.spawn_car(start, end)

	@staticmethod
	def get_index(path_key):
		border_names = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
		index = 0
		for border_name in border_names:
			if border_name == path_key:
				return index
			index += 1

	def spawn_car(self, start, end):
		starting_point = start
		end_position = self.borders[self.get_index(end)].get_position()
		index = self.get_index(starting_point)
		self.borders[index].spawn_car(end_position)

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

	def reset(self):
		#print("resetting")
		for intersection in self.intersections:
			intersection.reset()
		for border in self.borders:
			border.reset()
		#state = [0,0,0,0,0,0,0,0]
		state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		Car.reset_number_of_cars(Car)
		return state

	def step(self,action, t):
		

		#self.display_map()

		self.time_step()

		
		#print(a)
		#if (t % 3 == 0):
		#done = True
		#old_global_reward = self.global_reward
		#old_global_state = self.global_state
		#print("REWARD BEFORE RESET: ", self.global_reward)
		a = combinations[action]
		#print(a)
		#if (self.global_reward == 0):
		#	self.display_map()

		#print("Reward: ", self.global_reward)
		for intersection in self.intersections:
			intersection.reset_reward()
		
		self.global_reward = 0
		self.update_traffic_lights(a)

		self.global_state = []

		#return np.array(old_global_state),old_global_reward, done, {}

		self.update_cars(t)
		

		#print(a)

	#	self.model_reward = self.episode_reward
		#self.state = self.get_intersection_state()
		for index,intersection in enumerate(self.intersections):
			state,reward = (intersection.step(a[index]))
			self.global_state.extend(state)
			self.global_reward += reward
		done = True
		#print(global_state, global_reward, "\n")
		
		#self.update_cars(t);

		return np.array(self.global_state),self.global_reward, done, {}
		#return None
		#return np.array(old_global_state),old_global_reward, done, {}

	def display_map(self):
		print("        {0}  000       {1}  000        ".format(self.cars_at("I", "I"), self.cars_at("II", "II")))
		print("        {0}  000       {1}  000        ".format(self.cars_at("I", "A"), self.cars_at("II", "B")))
		print("000|000           {0}           {1}|{2}".format(self.cars_at("B", "A"), self.cars_at("IV", "B"), self.cars_at("IV", "IV")))
		print("{0}|{1}           {2}           000|000".format(self.cars_at("III", "III"), self.cars_at("III", "A"), self.cars_at("A", "B")))
		print("        {0}  {1}       {2}  {3}      ".format(self.cars_at("A", "C"), self.cars_at("C", "A"), self.cars_at("B", "D"), self.cars_at("D", "B")))
		print("000|000           {0}           {1}|{2}".format(self.cars_at("D", "C"), self.cars_at("VI", "D"), self.cars_at("VI", "VI")))
		print("{0}|{1}           {2}           000|000".format(self.cars_at("V", "V"), self.cars_at("V", "C"), self.cars_at("C", "D")))
		print("        000  {0}       000  {1}       ".format(self.cars_at("VII", "C"), self.cars_at("VIII", "D")))
		print("        000  {0}       000  {1}       ".format(self.cars_at("VII", "VII"), self.cars_at("VIII", "VIII")))
		print("------------------------------------")
