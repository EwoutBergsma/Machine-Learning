import sys
from jits.map import Map
from jits.car import Car


def main(argv):
	max_q_size = 5
	traffic_map = Map(max_q_size)

	n_time_steps = 100

	prev_disappeared_cars = 0
	for t in range(0, n_time_steps):
		time_step(traffic_map)
		n_cars = Car.get_number_of_cars(Car)
		if (n_cars[0] - n_cars[1] - traffic_map.number_of_cars()) > prev_disappeared_cars:
			prev_disappeared_cars = n_cars[0] - n_cars[1] - traffic_map.number_of_cars()
			print("car disappeared at time {0}".format(t))


	n_cars = Car.get_number_of_cars(Car)
	print("{0} cars were added to the system, {1} cars have left the system".format(n_cars[0], n_cars[1]))
	print("{0} cars are still in system".format(traffic_map.number_of_cars()))
	print("{0} cars have disappeared".format(n_cars[0] - n_cars[1] - traffic_map.number_of_cars()))


def time_step(traffic_map):
	n_cars = 50  # should not be greater than 8 (maximum number of cars to leave per time step)
	for c in range(n_cars):
		traffic_map.spawn_car([])

	traffic_map.update_cars()


if __name__ == "__main__":
	main(sys.argv)