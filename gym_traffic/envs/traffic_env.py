import gym
from gym import error, spaces, utils
import os

class TrafficEnv(gym.Env):
	metadata = {'render.modes':['human']}

	def __init__(self):
		# initialize the environment. Load from file perhaps?
		self._read_map()
		self._test_working()

	def step(self,action):
		# action may have to be a list of actions because we're dealing with multiple agents
		# reward may also have to be a list (can we embed a list in a tuple?)
		
		'''
		This was copied from: https://stackoverflow.com/questions/45068568/is-it-possible-to-create-a-new-gym-environment-in-openai
		Parameters
		----------
		action :

		Returns
		-------
		ob, reward, episode_over, info : tuple
			ob (object) :
				an environment-specific object representing your observation of
				the environment.
			reward (float) :
				amount of reward achieved by the previous action. The scale
				varies between environments, but the goal is always to increase
				your total reward.
			episode_over (bool) :
				whether it's time to reset the environment again. Most (but not
				all) tasks are divided up into well-defined episodes, and done
				being True indicates the episode has terminated. (For example,
				perhaps the pole tipped too far, or you lost your last life.)
			info (dict) :
				 diagnostic information useful for debugging. It can sometimes
				 be useful for learning (for example, it might contain the raw
				 probabilities behind the environment's last state change).
				 However, official evaluations of your agent are not allowed to
				 use this for learning.
		'''
		# for a test simply take action (a number) and print it
		print("Action = {}".format(action))
		return None,None,None,None

	def render(self, mode='human', close=False):
		# we're not going to render this for now
		pass

	def _take_action(self, action):
		# action validation will have to happen either here or in the step function
		pass

	def _read_map(self,file='traffic_map.json'):
		pass

	def _test_working(self):
		# simply print something to see if this is all functioning properly
		print("Well at least the init ran.")
		print("Working directory is: {}".format(os.getcwd()))
