import argparse
import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym


def main(args):
	env = gym.make('FrozenLake-v0')

	tf.reset_default_graph()

	# parameters
	n_input = 999
	n_hidden = 50
	n_output = 2401

	# learning rate
	n = 0.01

	n_time_steps = 200


	# These lines establish the feed-forward part of the network used to choose actions
	# input_layer = tf.placeholder(shape=[1, 16], dtype=tf.float32)
	input_layer = tf.placeholder("float", [None, n_input])

	# W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
	weights = {
		'h': tf.Variable(tf.random_uniform([n_input, n_hidden])),
		'out': tf.Variable(tf.random_uniform([n_hidden, n_output]))
	}
	biases = {
		'b': tf.Variable(tf.random_normal([n_hidden])),
		'out': tf.Variable(tf.random_normal([n_output]))
	}

	def multilayer_perceptron(x, weights, biases):
		# Hidden layer with RELU activation
		hidden_layer = tf.add(tf.matmul(x, weights['h']), biases['b'])
		hidden_layer = tf.nn.relu(hidden_layer)
		# Output layer with linear activation
		out_layer = tf.matmul(hidden_layer, weights['out']) + biases['out']
		return out_layer

	# Construct model
	output_layer = multilayer_perceptron(input_layer, weights, biases)
	predict = tf.argmax(output_layer, 1)
	# Qout = tf.matmul(input_layer, W)
	# predict = tf.argmax(Qout, 1)

	# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
	nextQ = tf.placeholder(shape=[1, n_input], dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - output_layer))
	trainer = tf.train.GradientDescentOptimizer(learning_rate=n)
	updateModel = trainer.minimize(loss)

	# init = tf.initialize_all_variables()
	init = tf.global_variables_initializer()

	# Set learning parameters
	# gamma
	y = .99
	# epsilon
	e = 0.1
	num_episodes = 2000
	# create lists to contain total rewards and steps per episode
	jList = []
	rList = []
	with tf.Session() as sess:
		sess.run(init)
		for i in tqdm(range(num_episodes)):
			# Reset environment and get first new observation
			s = env.reset()
			rAll = 0
			d = False
			j = 0
			# The Q-Network
			while j < n_time_steps:
				j += 1
				# Choose an action by greedily (with e chance of random action) from the Q-network
				a, allQ = sess.run([predict, output_layer], feed_dict={input_layer: np.identity(16)[s:s + 1]})
				if np.random.rand(1) < e:
					a[0] = env.action_space.sample()
				# Get new state and reward from environment
				s1, r, d, _ = env.step(a[0])
				# Obtain the Q' values by feeding the new state through our network
				Q1 = sess.run(output_layer, feed_dict={input_layer: np.identity(16)[s1:s1 + 1]})
				# Obtain maxQ' and set our target value for chosen action.
				maxQ1 = np.max(Q1)
				targetQ = allQ
				targetQ[0, a[0]] = r + y * maxQ1
				# Train our network using target and predicted Q values
				_, W1 = sess.run([updateModel, W], feed_dict={input_layer: np.identity(16)[s:s + 1], nextQ: targetQ})
				rAll += r
				s = s1
				if d == True:
					# Reduce chance of random action as we train the model.
					e = 1. / ((i / 50) + 10)
					break
			jList.append(j)
			rList.append(rAll)
	print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")
	plt.plot(rList)

	plt.plot(jList)

	plt.show()


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
																							 'Cartpole.')
	parser.add_argument('--algorithm', default='q', type=str,
											help='Choose between \'a3c\' and \'random\'.')
	parser.add_argument('--train', dest='train', action='store_true',
											help='Train our model.')
	parser.add_argument('--lr', default=0.001,
											help='Learning rate for the shared optimizer.')
	parser.add_argument('--update-freq', default=20, type=int,
											help='How often to update the global model.')
	parser.add_argument('--max-eps', default=1000, type=int,
											help='Global maximum number of episodes to run.')
	parser.add_argument('--gamma', default=0.99,
											help='Discount factor of rewards.')
	parser.add_argument('--save-dir', default='tmp/q/', type=str,
											help='Directory in which you desire to save the model.')
	args = parser.parse_args()

	main(args)