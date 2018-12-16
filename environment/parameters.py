""""Parameters of the traffic environment."""

#Number of iterations of the algorithm
n_time_steps = 10000

#Queue size indicating how many cars fit on the road.
max_q_size = 50

#number of new cars spawned per timestep
n_cars_spawned = 2

#reward per car passing trough an intersection
reward_per_car = 1

#multiplier of negative reward 
reward_coeff_waiting_time = 0.1