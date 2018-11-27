import gym
import gym_traffic # noqa

env = gym.make('traffic-v0')

for i in range(5):
	env.step(i) # do we want to make an enum class for what actions you can take?