import sys
from jits.intersection import Intersection
from jits.border_node import BorderNode
from jits.map import Map
from jits.node import Node


def main(argv):
	traffic_map = Map()

	traffic_map.spawn_car([1, 2, 1, 3])


if __name__ == "__main__":
	main(sys.argv)