import sys
from jits.map import Map
from jits.car import Car


def main(argv):
	traffic_map = Map()

	n_time_steps = 1000

	for t in range(0, n_time_steps):
		time_step(traffic_map)

	print(Car.get_number_of_cars(Car))


def time_step(traffic_map):
	n_cars = 5
	for c in range(n_cars):
		traffic_map.spawn_car([])

	traffic_map.update_cars()


if __name__ == "__main__":
	main(sys.argv)