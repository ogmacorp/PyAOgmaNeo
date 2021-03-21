# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

# Simple Cart-Pole example

import pyaogmaneo as pyaon
import gym
import numpy as np

# Squashing function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class ScalarEncoder:
    def __init__(self, num_scalars, num_columns, cells_per_column, lower_bound=0.0, upper_bound=1.0):
        self.num_scalars = num_scalars
        self.cells_per_column = cells_per_column

        self.protos = []

        for i in range(num_columns):
            self.protos.append(np.random.rand(cells_per_column, num_scalars) * (upper_bound - lower_bound) + lower_bound)

    def encode(self, scalars):
        csdr = []

        for i in range(len(self.protos)):
            acts = -np.sum(np.square(np.repeat(scalars.T, self.cells_per_column, axis=0) - self.protos[i]), axis=1)

            csdr.append(np.asscalar(np.argmax(acts)))

        return csdr

    def decode(self, csdr):
        scalars = np.zeros(self.num_scalars)

        for i in range(len(self.protos)):
            scalars += self.protos[csdr[i]]

# Create the environment
env = gym.make('CartPole-v1')

# Get observation size
numObs = env.observation_space.shape[0] # 4 values for Cart-Pole
numActions = env.action_space.n # N actions (1 discrete value)

se = ScalarEncoder(4, 9, 16)

# Set the number of threads
pyaon.setNumThreads(4)

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(2): # Layers with exponential memory. Not much memory is needed for Cart-Pole, so we only use 2 layers
    ld = pyaon.LayerDesc(hiddenSize=(4, 4, 16), errorSize=(4, 4, 16))
    
    lds.append(ld)

# Create the hierarchy: Provided with input layer sizes (a single column in this case), and input types (a single predicted layer)
h = pyaon.Hierarchy()
h.initRandom([ pyaon.IODesc((3, 3, 16), pyaon.prediction), pyaon.IODesc((1, 1, numActions), pyaon.action) ], lds)

reward = 0.0

action = 0

for episode in range(1000):
    obs = env.reset()

    # Timesteps
    for t in range(500):
        csdr = se.encode(sigmoid(np.matrix(obs).T * 4.0))

        h.step([ csdr, [ action ] ], True, reward)

        # Retrieve the action, the hierarchy already automatically applied exploration
        action = h.getPredictionCIs(1)[0] # First and only column

        obs, reward, done, info = env.step(action)

        # Re-define reward so that it is 0 normally and then -1 if done
        if done:
            reward = -100.0

            print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))

            break
        else:
            reward = 0.0
