# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import pyaogmaneo as neo
import gymnasium as gym
import numpy as np

# squashing function
def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5

# create the environment
env = gym.make('CartPole-v1')

# get observation size
num_obs = env.observation_space.shape[0] # 4 values for Cart-Pole
num_actions = env.action_space.n # N actions (1 discrete value)
input_resolution = 32

# set the number of threads
neo.set_num_threads(4)

# define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(8): # layers with exponential memory. Not much memory is needed for Cart-Pole, so we only use 2 layers
    ld = neo.LayerDesc()

    # set some layer structural parameters
    ld.hidden_size = (5, 5, 64)
    
    lds.append(ld)

s = neo.Searcher((5, 5, 32), 256)
s.params.lr = 0.001
s.params.exploration = 0.1

# create the hierarchy
h = neo.Hierarchy([ neo.IODesc((2, 2, input_resolution), neo.none), neo.IODesc((1, 1, num_actions), neo.prediction) ], lds)

reward = 0.0
average_reward = 0.0
action = 0

for episode in range(10000):
    obs, _ = env.reset()

    # timesteps
    for t in range(500):
        # sensory CSDR creation through "squash and bin" method
        csdr = (sigmoid(obs * 3.0) * (input_resolution - 1) + 0.5).astype(np.int32)

        average_reward += 0.001 * (reward - average_reward)

        s.step(average_reward, True)

        h.step([ csdr, [ action ] ], s.get_config_cis(), True)

        # retrieve the action, the hierarchy already automatically applied exploration
        action = h.get_prediction_cis(1)[0] # First and only column

        if np.random.rand() < 0.04:
            action = np.random.randint(0, num_actions)

        obs, reward, term, trunc, _ = env.step(action)

        # re-define reward so that it is 0 normally and then -100 if terminated
        if term:
            reward = -100.0
        else:
            reward = 0.0

        if term or trunc:
            print(f"Episode {episode + 1} finished after {t + 1} timesteps, {average_reward} average reward.")

            break
