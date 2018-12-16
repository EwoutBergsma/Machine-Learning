""" Each intersection has three traffic lights for each incoming direction.
Each item of the combinations list contains a list [north, east, south, west].
Each of these lists contains three items [left, straight ahead, right] (1 for green, 0 for red) """
combinations = [
	#One weather direction has all three traffic lights green
	[[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
	[[0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
	[[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
	[[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],

	#One weather direction has straight, right and same for the opposite weather direction
	[[0, 1, 1], [0, 0, 0], [0, 1, 1], [0, 0, 0]],
	[[0, 0, 0], [0, 1, 1], [0, 0, 0], [0, 1, 1]],

	#One weather direction has right turn green, the weather direction right of the former
	#has left turn green, and the opposite has turn right green.
	[[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]],
	[[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]],
	[[0, 0, 1], [0, 0, 0], [0, 0, 1], [1, 0, 0]],
	[[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1]],

	#One weather direction has right turn green, the weather direction right of the former
	#has right turn and straight green
	[[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]],
	[[0, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
	[[0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 0, 1]],
	[[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 1, 1]],

	#All weather direction have right turn green
	[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
]