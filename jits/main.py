import sys
from random import choice, randint
from map import Map
from car import Car
from paths import path_dict, border_names
from tqdm import tqdm


def main(argv):

	max_q_size = 5000
	traffic_map = Map(max_q_size)

	n_time_steps = 10000

	for t in tqdm(range(0, n_time_steps)):

		time_step(traffic_map)

		if t % 10 == 0:  # update traffic lights once every 10 time steps
			traffic_map.update_traffic_lights()
		traffic_map.update_cars(t)

	traffic_map.display_map()

	n_cars = Car.get_number_of_cars(Car)
	print("")
	print("{0} cars were added to the system, {1} cars have left the system".format(n_cars[0], n_cars[1]))
	print("{0} cars are still in system".format(traffic_map.number_of_cars()))
	print("{0} cars have disappeared".format(n_cars[0] - n_cars[1] - traffic_map.number_of_cars()))
	print("random dirs: " + str(Car.random_direction))


def time_step(traffic_map):
	n_cars = 1
	for c in range(n_cars):
		start = choice(border_names)
		end = choice(border_names)
		while start == end:
			end = choice(border_names)
		traffic_map.spawn_car(start, end)


if __name__ == "__main__":
	main(sys.argv)
