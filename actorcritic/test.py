import sys
from new_global_traffic_light_combinations import combinations
from numpy.random import randint

def main(argv):
# 	combinations = [
# 		# One lane fully green, rest red
# 		[[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
# 		[[0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
# 		[[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
# 		[[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
#
# 		# Two opposite lanes both straight ahead and right green
# 		[[0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 1]],
# 		[[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0]],
#
# 		# Every lane right
# 		[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
# 	]
#
# 	# action = combinations[randint(210)]
# 	all_combs = []
# 	for i in range(7):
# 		for j in range(7):
# 			for k in range(7):
# 				for l in range(7):
# 					row_comb = [combinations[i], combinations[j], combinations[k], combinations[l]]
# 					all_combs.append(row_comb)


	index = 0
	for comb in combinations:
		print(index)
		print(comb)
		index += 1




if __name__ == "__main__":
	main(sys.argv)