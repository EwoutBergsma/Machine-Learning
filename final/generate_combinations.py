from traffic_light_combinations import combinations
import sys
import itertools

def main(argv):
	combos = list(itertools.combinations_with_replacement(combinations,4))
	print(combos)

if __name__ == "__main__":
	main(sys.argv)
