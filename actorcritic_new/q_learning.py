import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm


import sys

# directory in which the folder /actorcritic_new is stored
sys.path.append('/Users/jits/git/Machine-Learning/actorcritic_new')
from map import Map

def main(args):
	max_q_size = 50
	traffic_map = Map(max_q_size)
	env = traffic_map

	tf.reset_default_graph()

	# ----- parameters -----

	# number of input units
	n_input = env.state_size

	# number of output units
	n_output = env.action_size

	# set number of hidden units as mean of input units and output units
	n_hidden = int(np.ceil(np.mean([n_input, n_output])))

	# learning rate
	n = 0.01

	# number of steps per epoch
	n_time_steps = 3000

	# number of epochs
	num_episodes = 250

	# Set reinforcement learning parameters
	# gamma
	y = 0.1
	max_y = 0.95

	# epsilon
	e = 1.
	min_e = 0.01

	# ----- Initialize network -----

	# initialize input layer
	input_layer = tf.placeholder(shape=[1, n_input], dtype=tf.float32)

	# initialize weights
	hidden_weights = tf.Variable(tf.random_uniform([n_input, n_hidden], -0.5, 0.5))#0.01))
	output_weights = tf.Variable(tf.random_uniform([n_hidden, n_output], -0.5, 0.5))#0.01))

	# initialize bias
	hidden_bias = tf.Variable(tf.random_normal([n_hidden]))
	output_bias = tf.Variable(tf.random_normal([n_output]))

	# Construct model
	hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(input_layer, hidden_weights), hidden_bias))
	output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_bias)
	predict = tf.argmax(output_layer, 1)

	# Initialize next Q-value
	next_q = tf.placeholder(shape=[1, n_output], dtype=tf.float32)

	# Loss is defined as the sum of squares of the difference between target and predicted Q-values
	loss = tf.reduce_sum(tf.square(next_q - output_layer))
	# loss = tf.reduce_sum(next_q - output_layer)


	# Initialize Model
	trainer = tf.train.GradientDescentOptimizer(learning_rate=n)
	update_model = trainer.minimize(loss)
	init = tf.global_variables_initializer()

	# create lists to contain total rewards and steps per episode
	e_list = []
	r_list = []
	t_list = []

	a_list = []
	with tf.Session() as sess:
		sess.run(init)
		error_flag = False
		for i in tqdm(range(num_episodes)):
			if error_flag:
				print("Q_values contained 'nan'")
				break
			# Reset environment and get first new observation
			s = env.reset()
			r_all = 0
			d = False
			# The Q-Network
			t = 0
			while t < n_time_steps:
				t += 1
				# Choose an action by greedily (with e chance of random action) from the Q-network
				# print(s)
				a, all_q = sess.run([predict, output_layer], feed_dict={input_layer: get_state(s, max_q_size, n_time_steps)})
				a_list.append(a[0])
				if contains_nan(all_q):
					error_flag = True
					break
				if np.random.rand(1) < e:
					# a[0] = env.action_space.sample()
					a[0] = np.random.randint(n_output)
				# Get new state and reward from environment
				s1, r, d, _ = env.step(a[0], t)
				r_all += r

				# Obtain the Q' values by feeding the new state through our network
				new_q = sess.run(output_layer, feed_dict={input_layer: get_state(s1, max_q_size, n_time_steps)})
				# Obtain maxQ' and set our target value for chosen action.
				max_new_q = np.max(new_q)
				target_q = all_q

				if simulation_stuck(s1, max_q_size):
					# Train our network using target and predicted Q values
					print("simulation stuck at t = {0}".format(t))
					target_q[0, a[0]] = -100
					_, = sess.run([update_model], feed_dict={input_layer: get_state(s, max_q_size, n_time_steps), next_q: target_q})
					break
				else:
					# Train our network using target and predicted Q values
					target_q[0, a[0]] = ((r-5)/10) + y * max_new_q
					_, = sess.run([update_model], feed_dict={input_layer: get_state(s, max_q_size, n_time_steps), next_q: target_q})

				s = s1
			# Reduce chance of random action as we train the model.
			# exploitation at the last 30% of episodes
			if i > 0.7 * num_episodes:
				e = 0
			else:
				e = max(min_e, e - 1 / (0.7 * num_episodes))

			y = min(max_y, y + 2 / num_episodes)

			r_list.append(r_all)
			e_list.append(e)
			t_list.append(t)

		fig1, ax1 = plt.plot(np.arange(len(moving_average(r_list, 10))), moving_average(r_list, 10),'b',np.arange(len(moving_average(t_list, 10))), moving_average(t_list, 10), 'g')
		plt.show()
		n, bins, patches = plt.hist(a_list, bins=n_output)
		plt.show()


def simulation_stuck(s, max_q_size):
	full_qs = 0
	for i in range(len(s)):
		if i % 8 < 4 and s[i] >= max_q_size:
			full_qs += 1

	if full_qs >= 12:
		return True
	return False


def contains_nan(all_q):
	for number in all_q[0]:
		if np.isnan(number):
			return True
	return False


def get_state(s, max_q_size, n_time_steps):
	# returns input vector
	state = np.identity(len(s))[0:1]
	for i in range(len(state[0])):
		if i % 8 < 4:
			state[0][i] = max(min(s[i]/max_q_size, 1), 0)
		else:
			state[0][i] = max(min(s[i]/n_time_steps, 1), 0)
	return state

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