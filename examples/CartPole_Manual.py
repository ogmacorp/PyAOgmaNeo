# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

# Simple Cart-Pole example

import pyaogmaneo as neo
import gym
import numpy as np

# Squashing function
def sigmoid(x):
    return np.tanh(x) * 0.5 + 0.5

# An example of a pre-encoder. This one is just random projection. It's not very good, but will do for this task
class ScalarEncoder:
    def __init__(self, num_scalars, num_columns, cells_per_column, lower_bound=0.0, upper_bound=1.0):
        self.num_scalars = num_scalars
        self.cells_per_column = cells_per_column

        self.protos = []

        for _ in range(num_columns):
            self.protos.append(np.random.rand(cells_per_column, num_scalars) * (upper_bound - lower_bound) + lower_bound)

    def encode(self, scalars):
        csdr = []

        for i in range(len(self.protos)):
            acts = -np.sum(np.square(np.repeat(scalars.T, self.cells_per_column, axis=0) - self.protos[i]), axis=1)

            csdr.append(np.argmax(acts).item())

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

res = 16 # Resolution (column size) of encoding

se = ScalarEncoder(4, 9, res)

# Set the number of threads
neo.setNumThreads(8)

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(2): # Layers with exponential memory. Not much memory is needed for Cart-Pole
    ld = neo.LayerDesc(hiddenSize=(4, 4, 16))

    ld.eRadius = 2
    ld.dRadius = 2
    
    lds.append(ld)

# Create the hierarchy: Provided with input layer sizes (a single column in this case), and input types (a single predicted layer)
h = neo.Hierarchy()
h.initRandom([ neo.IODesc((3, 3, res), neo.none, eRadius=2, dRadius=2), neo.IODesc((1, 1, numActions), neo.action, eRadius=0, dRadius=2, historyCapacity=64) ], lds)

# Set some parameters for the actor IO layer (index 1)
h.setAVLR(1, 0.01)
h.setAALR(1, 0.01)
h.setADiscount(1, 0.99)
h.setAMinSteps(1, 16)
h.setAHistoryIters(1, 16)

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

        # Re-define reward so that it is 0 normally and then -100 if done
        if done:
            reward = -100.0

            print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))

            break
        else:
            reward = 0.0
