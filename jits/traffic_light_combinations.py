# Each item of the combinations list contains a list [north, east, south, west].
# Each of these lists contains three items [left, straight ahead, right] (1 for green, 0 for red)

combinations = [
	
	# One lane fully green, rest red
	[[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
	[[0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
	[[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
	[[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],

	# Two opposite lanes both straight ahead and right green
	[[0, 1, 1], [0, 0, 0], [0, 0, 0], [0, 1, 1]],
	[[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0]],

	# Every lane right
	[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],

]

# combinations = [
# 	[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
#  ]