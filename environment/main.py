import sys
from random import choice, randint
from map.map import Map
from tqdm import tqdm
from parameters import *
from model.car_queue import CarQueue
from dictionaries.paths import border_names
from model.car import Car


def main(argv):
	"""Simulation of traffic flow representing cars passing trough intersections. Parameters of the simulation
	are read from parameters.py."""

	#Initialize the map
	traffic_map = Map(max_q_size)

	#Run simulation and show progress bar	
	for t in tqdm(range(n_time_steps)):
		spawn_cars(traffic_map)
		traffic_map.update_traffic_lights()
		traffic_map.update_cars(t)

	traffic_map.display_map()

	n_cars = Car.get_number_of_cars(Car)
	print("")
	print(f"{n_cars[0]} cars were added to the system, {n_cars[1]} cars have left the system")
	print(f"{traffic_map.number_of_cars()} cars are still in system")
	print(f"{n_cars[0] - n_cars[1] - traffic_map.number_of_cars()} cars have disappeared")
	print(f"random dirs: {Car.random_direction}")


def spawn_cars(traffic_map):
	for c in range(n_cars_spawned):
		start = choice(border_names)
		end = choice(border_names)
		while start == end:
			end = choice(border_names)
		traffic_map.spawn_car(start, end)


if __name__ == "__main__":
	main(sys.argv)
