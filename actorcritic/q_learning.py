import argparse
import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
from gym.envs.registration import register
from map import Map

def main(args):
	register(
		id='FrozenLake-v3',
		entry_point='gym.envs.toy_text:FrozenLakeEnv',
		kwargs={'map_name': '4x4', 'is_slippery': False}
	)
	pos_actions = [" LEFT  ", " DOWN  ", " RIGHT ", "  UP   "]

	# env = gym.make('FrozenLakeNotSlippery-v0')

	# max_q_size = 50
	# env = Map(max_q_size)
	# n_input = env.state_size
	# n_output = env.action_size
	env = gym.make('FrozenLake-v3')

	tf.reset_default_graph()

	# ----- parameters -----

	# number of input units
	n_input = 16

	# number of output units
	n_output = 4

	# set number of hidden units as mean of input units and output units
	n_hidden = int(np.ceil(np.mean([n_input, n_output])))

	# learning rate
	n = 0.1

	# number of steps per epoch
	n_time_steps = 200

	# number of epochs
	num_episodes = 10000

	# Set reinforcement learning parameters
	# gamma
	y = 0.5

	# epsilon
	e = 1.
	min_e = 0.01

	# ----- Initialize network -----

	# initialize input layer
	input_layer = tf.placeholder(shape=[1, n_input], dtype=tf.float32)

	# initialize weights
	hidden_weights = tf.Variable(tf.random_uniform([n_input, n_hidden], 0, 0.01))
	output_weights = tf.Variable(tf.random_uniform([n_hidden, n_output], 0, 0.01))
	# output_weights = tf.Variable(tf.random_uniform([n_hidden + 1, n_output], 0, 0.01))

	# Construct model
	# hidden_layer = tf.matmul(input_layer, hidden_weights)
	# output_layer = tf.matmul(hidden_layer, output_weights)
	# output_layer = tf.matmul(tf.matmul(input_layer, hidden_weights), output_weights)
	output_layer = tf.matmul(tf.nn.relu(tf.matmul(input_layer, hidden_weights)), output_weights)
	predict = tf.argmax(output_layer, 1)

	# Initialize next Q-value
	next_q = tf.placeholder(shape=[1, n_output], dtype=tf.float32)

	# Loss is defined as the sum of squares of the difference between target and precited Q-values
	loss = tf.reduce_sum(tf.square(next_q - output_layer))

	# Initialize Model
	trainer = tf.train.GradientDescentOptimizer(learning_rate=n)
	update_model = trainer.minimize(loss)
	init = tf.global_variables_initializer()

	# create lists to contain total rewards and steps per episode
	jList = []
	rList = []
	with tf.Session() as sess:
		sess.run(init)
		finish_flag = False
		for i in tqdm(range(num_episodes)):
			# if finish_flag:
			# 	break
			# Reset environment and get first new observation
			s = env.reset()
			r_all = 0
			d = False
			# The Q-Network
			t = 0
			while t < n_time_steps:

				t += 1
				# Choose an action by greedily (with e chance of random action) from the Q-network
				a, all_q = sess.run([predict, output_layer], feed_dict={input_layer: get_state(s, n_input)})
				# print("Q-values of state 14 = {0}".format(all_q))
				# if s == 14:
				# 	print("Q-values of state {0} = {1}".format(s, all_q))

				if np.random.rand(1) < e:
					a[0] = env.action_space.sample()
				# Get new state and reward from environment
				s1, r, d, _ = env.step(a[0])
				r_all += r
				if r > 0 and e == 0:
					print("reached goal")
					finish_flag = True
					break
				if d:
					# Reduce chance of random action as we train the model.
					e = max(min_e, e - 1/num_episodes)
					# exploitation at the last 30% of episodes
					if i > 0.7 * num_episodes:
						e = 0

				# elif s1 == s:
				# 	r -= 2
				# else:
				# 	r -= 1
				# Obtain the Q' values by feeding the new state through our network
				Q1 = sess.run(output_layer, feed_dict={input_layer: get_state(s1, n_input)})
				# Obtain maxQ' and set our target value for chosen action.
				maxQ1 = np.max(Q1)
				target_q = all_q

				if d:
					if r > 0:
						target_q[0, a[0]] = r
					else:
						target_q[0, a[0]] = 0
				else:
					target_q[0, a[0]] = r + y * maxQ1
				# if s == 14:
				# 	print("Max q1 = {0}".format(maxQ1))
				# 	print("Target Q-values of state {0},{1} = {2}".format(s, s1, target_q))

				# Train our network using target and predicted Q values
				_, = sess.run([update_model], feed_dict={input_layer: get_state(s, n_input), next_q: target_q})

				# if s == 14:
				# 	a, all_q = sess.run([predict, output_layer], feed_dict={input_layer: get_state(s, n_input)})
				# 	print("new Q-values of state {0} = {1}".format(s, all_q))
				if d:
					break
				s = s1

			rList.append(r_all)

		for row in range(4):
			this_row = ""
			for col in range(4):
				state = row * 4 + col
				# print("state = {0}".format(state))
				action, all_q = sess.run([predict, output_layer], feed_dict={input_layer: get_state(state, n_input)})
				this_row = this_row + pos_actions[action[0]]
				# print(pos_actions[action[0]])
				# print("Q-values of state {0} = {1}".format(state, all_q))
			print(this_row)

		cont = input("Continue?")
		d = False
		s = env.reset()
		while not d and not cont == 'n':
			print("state = {0}".format(s))
			action, all_q = sess.run([predict, output_layer], feed_dict={input_layer: get_state(s, n_input)})
			print("perform action {0}".format(pos_actions[action[0]]))

			s1, r, d, _ = env.step(action[0])
			print("new state = {0}".format(s1))
			s = s1
			cont = input("Continue? (y/n) -- ")

		print("Percent of successful episodes: {0:.2f}%".format(100 * (sum(rList[int(np.ceil(0.8*num_episodes)):]) / (0.2*num_episodes))))

		plt.plot(moving_average(rList, 100))

		plt.show()


def get_state(s, n_input):
	# returns input vector
	# currently the state is a zero-vector with a 1 for the active state
	return np.identity(n_input)[s:s + 1]

def moving_average(given_list, N):
	cumsum, moving_aves = [0], []
	for i, x in enumerate(given_list, 1):
		cumsum.append(cumsum[i - 1] + x)
		if i >= N:
			moving_ave = (cumsum[i] - cumsum[i - N]) / N
			# can do stuff with moving_ave here
			moving_aves.append(moving_ave)
	return moving_aves



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Run Q-learning algorithm')
	parser.add_argument('--algorithm', default='q', type=str,
											help='Choose between \'q\' and \'random\'.')
	parser.add_argument('--train', dest='train', action='store_true',
											help='Train our model.')
	parser.add_argument('--lr', default=0.05,
											help='Learning rate for the mlp.')
	parser.add_argument('--update-freq', default=20, type=int,
											help='How often to update the global model.')
	parser.add_argument('--max-eps', default=10000, type=int,
											help='Global maximum number of episodes to run.')
	parser.add_argument('--gamma', default=0.99,
											help='Discount factor of rewards.')
	parser.add_argument('--save-dir', default='tmp/q/', type=str,
											help='Directory in which you desire to save the model.')
	parser.add_argument('--environment', default='traffic', type=str,
											help='Environment for which the algorithm is trained')
	args = parser.parse_args()

	main(args)