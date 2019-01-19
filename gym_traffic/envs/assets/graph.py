import heapq # could use this, but don't right now
import queue # this is simpler and what we need
# Graph and nodes
# Making a custom graph data type because I don't know of any good python frameworks for it

class Graph():
	'''
	This is the graph datatype.
	Requirements:
		TODO: General graph stucture
		TODO: Implement reading from JSON
		TODO: Implement writing to JSON - I expect this is not going to be easy
	'''
	def __init__(self):
		self.nodes = [] # initialize empty list of nodes

	def fromJson(self,filename):
		# from file with filename read a graph

class Node():
	'''
	Parent class for both types of possible nodes
	'''
	def __init__(self,raw):
		self.raw = raw
		self.parse()
		self.clearRaw()

	def parse(self):
		pass

	def clearRaw(self):
		self.raw = None # Hopefully reduces memory usage?

class Intersection(Node):
	''' Intersection node class '''

	def parse(self):
		super().parse()
		self.node_id = self.raw.get("NodeID")
		self.node_type = self.raw.get("NodeType")
		self.connections = self.raw.get("Connections") # we can just keep it as a dictionary
		self.cars_up = queue.Queue(maxsize=20) # fifo queue
		self.cars_down = queue.Queue(maxsize=20)
		self.cars_left = queue.Queue(maxsize=20)
		self.cars_right= queue.Queue(maxsize=20)
		

