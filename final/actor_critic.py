import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import threading
import multiprocessing
import numpy as np
from queue import Queue
import argparse
import matplotlib.pyplot as plt
from map import Map
from random import choice, randint
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from car import Car
from paths import path_dict, border_names
from tqdm import tqdm
from car import Car
from q_learning import moving_average

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'Cartpole.')
parser.add_argument('--algorithm', default='a3c', type=str,
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
parser.add_argument('--save-dir', default='tmp/', type=str,
                    help='Directory in which you desire to save the model.')
args = parser.parse_args()

class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.dense1 = layers.Dense(100, activation='relu')
    self.policy_logits = layers.Dense(action_size)
    self.dense2 = layers.Dense(100, activation='relu')
    self.values = layers.Dense(1)

  def call(self, inputs):
    # Forward pass
    x = self.dense1(inputs)
    logits = self.policy_logits(x)
    v1 = self.dense2(inputs)
    values = self.values(v1)
    return logits, values

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
  """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: Global reward from episode
    result_queue: Queue storing episode scores
    total_loss: The total loss accumualted over the current episode
  """
  global_ep_reward = episode_reward
  """
  print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {int(global_ep_reward)} | "
      f"Episode Reward: {int(episode_reward)} | "
      f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx}"
  )
  """
  result_queue.put(global_ep_reward)
  return global_ep_reward


class RandomAgent:
  """Random Agent that will control the traffic lights
    Arguments:
      max_eps: Maximum number of episodes to run agent for.
  """
  def __init__(self, max_eps):
    self.max_q_size = 50
    self.traffic_map = Map(self.max_q_size)
    self.env = self.traffic_map
    self.max_episodes = max_eps
    self.global_moving_average_reward = 0
    self.res_queue = Queue()

  def run(self):
    reward_avg = 0
    #Number of time steps per episode. Change this if needed, depends on which A3C time step settings you compare it to
    time = 1000
    for episode in range(self.max_episodes):
      done = False
      self.env.reset()
      reward_sum = 0.0
      steps = 0
      for t in range(0,time):
        #sample and update environment randomly
        action = np.random.choice(self.env.action_size)
        new_state, reward, done, _ = self.env.step(action,t)
        reward_sum += reward
        steps+=1
        
      # Record statistics
      self.global_moving_average_reward = record(episode,
                                                 reward_sum,
                                                 0,
                                                 self.global_moving_average_reward,
                                                 self.res_queue, 0, steps)

      reward_avg += reward_sum
    final_avg = reward_avg / float(self.max_episodes)
    print("Average score across {} episodes: {}".format(self.max_episodes, final_avg))
    return final_avg


class MasterAgent():
  def __init__(self):

    save_dir = args.save_dir
    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    max_q_size = 50
    traffic_map = Map(max_q_size)
    env = traffic_map
    #self.state_size = env.observation_space.shape[0]
    #self.action_size = env.action_space.n
    self.state_size = env.state_size
    self.action_size = env.action_size
    self.opt = tf.train.AdamOptimizer(args.lr, use_locking=True)
  #  print(self.state_size, self.action_size)

    self.global_model = ActorCriticModel(self.state_size, self.action_size)  # global network
    self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

  def train(self):
    if args.algorithm == 'random':
      random_agent = RandomAgent(args.max_eps)
      random_agent.run()
      return

    res_queue = Queue()

    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i,
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = []  # record episode reward to plot
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]

    plt.plot(moving_average_rewards)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig(os.path.join(self.save_dir,
                             '{} A3C.png'.format("traffic")))
    plt.show()


class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


class Worker(threading.Thread):
  # Set up global variables across different threads
  global_episode = 0
  # Moving average reward
  global_moving_average_reward = 0
  best_score = 0
  save_lock = threading.Lock()

  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,
               save_dir='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ActorCriticModel(self.state_size, self.action_size)
    self.worker_idx = idx
    self.max_q_size = 50
    self.traffic_map = Map(self.max_q_size)
    self.env = self.traffic_map
    self.save_dir = save_dir
    self.ep_loss = 0.0


  def run(self):

    total_step = 1
    mem = Memory()
    while Worker.global_episode < args.max_eps:
      print("Epoch: {0}".format(Worker.global_episode))

      current_state = self.env.reset()
      mem.clear()
      ep_reward = 0.
      ep_steps = 0
      self.ep_loss = 0

      time_count = 0
      done = False

      n_time_steps = 1000
      
      for t in range(0, n_time_steps):

        #sample action based on current model and update environment
        logits, _ = self.local_model(

            tf.convert_to_tensor(current_state[None, :],
                               dtype=tf.float32))
        probs = tf.nn.softmax(logits)

        action = np.random.choice(self.action_size, p=probs.numpy()[0])
        new_state, reward, done, _ = self.env.step(action,t)

        ep_reward += reward
    
        mem.store(current_state, action, reward)

        if time_count == args.update_freq or done:
          # Calculate gradient wrt to local model. We do so by tracking the
          # variables involved in computing the loss by using tf.GradientTape
          with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done,
                                           new_state,
                                           mem,
                                           args.gamma)
          self.ep_loss += total_loss
          # Calculate local gradients
          grads = tape.gradient(total_loss, self.local_model.trainable_weights)
          # Push local gradients to global model
          self.opt.apply_gradients(zip(grads,
                                       self.global_model.trainable_weights))
          # Update local model with new weights
          self.local_model.set_weights(self.global_model.get_weights())

          mem.clear()
          time_count = 0
          
        ep_steps += 1

        time_count += 1
        current_state = new_state
        total_step += 1

      Worker.global_moving_average_reward = \
        record(Worker.global_episode, ep_reward, self.worker_idx,
               Worker.global_moving_average_reward, self.result_queue,
               self.ep_loss, ep_steps)
      #Use lock to save our model and to print to prevent data races.
      if ep_reward > Worker.best_score:
        with Worker.save_lock:
          print("Saving best model to {}, "
                "episode score: {}".format(self.save_dir, ep_reward))
          self.global_model.save_weights(
              os.path.join(self.save_dir,
                           'model_{}.h5'.format("traffic"))
          )
          Worker.best_score = ep_reward      
      Worker.global_episode += 1
      
    self.result_queue.put(None)


  def compute_loss(self,
                   done,
                   new_state,
                   memory,
                   gamma=0.99):
    if done:
      reward_sum = 0.  # terminal
    else:
      reward_sum = self.local_model(
          tf.convert_to_tensor(new_state[None, :],
                               dtype=tf.float32))[-1].numpy()[0]

    # Get discounted rewards
    discounted_rewards = []
    for reward in memory.rewards[::-1]:  # reverse buffer r
      reward_sum = reward + gamma * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    logits, values = self.local_model(
        tf.convert_to_tensor(np.vstack(memory.states),
                             dtype=tf.float32))
    # Get our advantages
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                            dtype=tf.float32) - values
    # Value loss
    value_loss = advantage ** 2

    # Calculate our policy loss
    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                 logits=logits)
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
    return total_loss


if __name__ == '__main__':
  print(args)
  master = MasterAgent()
  if args.train:
    master.train()
  else:
    print("Please use the --train option.")
    sys.exit()
