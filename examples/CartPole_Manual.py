# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
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
    return 1.0 / (1.0 + np.exp(-x))

# Create the environment
env = gym.make('CartPole-v1')

# Get observation size
numObs = env.observation_space.shape[0] # 4 values for Cart-Pole
numActions = env.action_space.n # N actions (1 discrete value)
inputResolution = 32

# Set the number of threads
neo.setNumThreads(4)

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(2): # Layers with exponential memory. Not much memory is needed for Cart-Pole, so we only use 2 layers
    ld = neo.LayerDesc()

    # Set some layer structural parameters
    ld.hiddenSize = (4, 4, 32)
    ld.ticksPerUpdate = 2 # How many ticks before a layer updates (compared to previous layer) - clock speed for exponential memory
    ld.temporalHorizon = 2 # Memory horizon of the layer. Must be greater or equal to ticksPerUpdate
    
    lds.append(ld)

# Create the hierarchy
h = neo.Hierarchy()
h.initRandom([ neo.IODesc((2, 2, inputResolution), neo.none), neo.IODesc((1, 1, numActions), neo.action) ], lds)

# Setting parameters
h.setAVLR(1, 0.01) # Parameters: IO index and value. Here, we set the actor's action learning rate.

reward = 0.0

for episode in range(1000):
    obs, _ = env.reset()

    # Timesteps
    for t in range(500):
        # CSDR through "squash and bin" method
        csdr = (sigmoid(obs * 3.0) * (inputResolution - 1) + 0.5).astype(np.int32).tolist()

        h.step([ csdr, h.getPredictionCIs(1) ], True, reward)

        # Retrieve the action, the hierarchy already automatically applied exploration
        action = h.getPredictionCIs(1)[0] # First and only column

        obs, reward, term, trunc, _ = env.step(action)

        # Re-define reward so that it is 0 normally and then -1 if done
        if term:
            reward = -1.0

            print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))

            break
        else:
            reward = 0.0
