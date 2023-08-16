import numpy as np
import os

# TRAINING HYPERPARAMETERS

model_name = 'model_pot_19'
duration = 1000

# Learning rate parameters - Exponential decay
initial_learning_rate = 0.005
decay_steps = 4000
decay_rate = 0.90

fc_layer_params = (128,128)
discount_factor = 0.95
target_update_tau = 0.01
target_update_period = 1

replay_buffer_max_length = 100000
num_parallel_calls = 2
sample_batch_size = 128
num_steps = 2
prefetch = 3

max_episodes = 6000

epsilon_greedy_episodes = 1000
random_seed = 12345

DQN_update_time_steps = 10              # Updates DQN parameters every these many time steps
DQN_policy_store_frequency = 50        # Stores DQN policy every these many episodes
DQN_loss_avg_interval = 100             # Computes DQN loss and returns by averaging over these many episodes

def epsilon_greedy(ep_counter):
    # Epsilon greedy algorithm to balance exploration and exploitation
    if ep_counter <= epsilon_greedy_episodes:

        return np.minimum(1 - (ep_counter / epsilon_greedy_episodes), 0.8)
    else: 
        return 0#0.02 #((max_episodes - ep_counter) / (max_episodes - epsilon_greedy_episodes))*0.02

def print_params(path):
    fid = open(os.path.join(path,'parameters.txt'),'w')
    fid.write(f'model_name:{model_name}\nduration:{duration}\n\n')
    fid.write(f'initial_learning_rate:{initial_learning_rate}\ndecay_steps:{decay_steps}\ndecay_rate:{decay_rate}\n\n')
    fid.write(f'fc_layer_params:{fc_layer_params}\ndiscount_factor:{discount_factor}\ntarget_update_tau:{target_update_tau}\ntarget_update_period:{target_update_period}\n\n')
    fid.write(f'replay_buffer_max_length:{replay_buffer_max_length}\nnum_parallel_calls:{num_parallel_calls}\nsample_batch_size:{sample_batch_size}\nnum_steps:{num_steps}\nprefetch:{prefetch}\n\n')
    fid.write(f'max_episodes:{max_episodes}\nepsilon_greedy_episodes:{epsilon_greedy_episodes}\nrandom_seed:{random_seed}\n\n')
    fid.write(f'DQN_update_time_steps:{DQN_update_time_steps}\nDQN_policy_store_frequency:{DQN_policy_store_frequency}\nDQN_loss_avg_interval:{DQN_loss_avg_interval}\n')
    fid.close()
