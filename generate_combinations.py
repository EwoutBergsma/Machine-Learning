from traffic_light_combinations import combinations
import sys
import itertools

def main(argv):
	#combos = list(itertools.permutations(combinations))
	combos = list(itertools.combinations_with_replacement(combinations,4))
	#print(len(combos))
	print(combos)

	"""
	for combo in combos:
		for c in combo:
			print(c)
	#	print(combo, "\n")
		print("\n")
	"""
if __name__ == "__main__":
	main(sys.argv)
