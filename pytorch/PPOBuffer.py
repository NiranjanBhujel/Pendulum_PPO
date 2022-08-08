"""
Implementation of PPO buffer

Author:     Niranjan Bhujel
Date:       Aug 3, 2022
"""

import numpy as np


def compute_return_advantage(rewards, values, is_last_terminal, gamma, gae_lambda, last_value):
    """
    Computes returns and advantage based on generalized advantage estimation.
    """
    N = rewards.shape[0]
    advantages = np.zeros(
        (N, 1),
        dtype=np.float32
    )

    tmp = 0.0
    for k in reversed(range(N)):
        if k==N-1:
            next_non_terminal = 1 - is_last_terminal
            next_values = last_value
        else:
            next_non_terminal = 1
            next_values = values[k+1]

        delta = rewards[k] + gamma * next_non_terminal * next_values - values[k]
        tmp = delta + gamma * gae_lambda * next_non_terminal * tmp
        
        advantages[k] = tmp
    
    returns = advantages +  values

    return returns, advantages


class PPOBuffer:
    def __init__(self, obs_dim, action_dim, buffer_capacity, seed=None) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_capacity = buffer_capacity

        self.obs = np.zeros(
            shape=(self.buffer_capacity, self.obs_dim),
            dtype=np.float32
        )
        self.action = np.zeros(
            shape=(self.buffer_capacity, self.action_dim),
            dtype=np.float32
        )
        self.reward = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.log_prob = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.returns = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.advantage = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )
        self.values = np.zeros(
            shape=(self.buffer_capacity, 1),
            dtype=np.float32
        )


        self.rng = np.random.default_rng(seed=seed)
        self.start_index, self.pointer = 0, 0

    def record(self, obs, action, reward, values, log_prob):
        self.obs[self.pointer] = obs
        self.action[self.pointer] = action
        self.reward[self.pointer] = reward
        self.values[self.pointer] = values
        self.log_prob[self.pointer] = log_prob

        self.pointer += 1

    def process_trajectory(self, gamma, gae_lam, is_last_terminal, last_v):
        path_slice = slice(self.start_index, self.pointer)
        values_t = self.values[path_slice]

        self.returns[path_slice], self.advantage[path_slice] = compute_return_advantage(
            self.reward[path_slice],
            values_t,
            is_last_terminal,
            gamma,
            gae_lam,
            last_v
        )

        self.start_index = self.pointer

    def get_data(self):
        whole_slice = slice(0, self.pointer)
        return {
            'obs': self.obs[whole_slice],
            'action': self.action[whole_slice],
            'reward': self.reward[whole_slice],
            'values': self.values[whole_slice],
            'log_prob': self.log_prob[whole_slice],
            'return': self.returns[whole_slice],
            'advantage': self.advantage[whole_slice],
        }

    def get_mini_batch(self, batch_size):
        assert batch_size <= self.pointer, "Batch size must be smaller than number of data."
        indices = np.arange(self.pointer)
        self.rng.shuffle(indices)
        

        split_indices = []
        point = batch_size
        while point < self.pointer:
            split_indices.append(point)
            point += batch_size

        temp_data = {
            'obs': np.split(self.obs[indices], split_indices),
            'action': np.split(self.action[indices], split_indices),
            'reward': np.split(self.reward[indices], split_indices),
            'values': np.split(self.values[indices], split_indices),
            'log_prob': np.split(self.log_prob[indices], split_indices),
            'return': np.split(self.returns[indices], split_indices),
            'advantage': np.split(self.advantage[indices], split_indices),
        }

        n = len(temp_data['obs'])
        data_out = []
        for k in range(n):
            data_out.append(
                {
                    'obs': temp_data['obs'][k],
                    'action': temp_data['action'][k],
                    'reward': temp_data['reward'][k],
                    'values': temp_data['values'][k],
                    'log_prob': temp_data['log_prob'][k],
                    'return': temp_data['return'][k],
                    'advantage': temp_data['advantage'][k],
                }
            )
        
        return data_out

    def clear(self):
        self.start_index, self.pointer = 0, 0


