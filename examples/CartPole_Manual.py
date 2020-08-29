# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
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

# Create the environment
env = gym.make('CartPole-v1')

# Get observation size
numObs = env.observation_space.shape[0] # 4 values for Cart-Pole
numActions = env.action_space.n # N actions (1 discrete value)

# Squashing scale multiplier for observation
obsSquashScale = 1.0

# Define binning resolution
obsColumnSize = 32

# Set the number of threads
pyaon.setNumThreads(4)

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(2): # Layers with exponential memory. Not much memory is needed for Cart-Pole, so we only use 2 layers
    ld = pyaon.LayerDesc()

    # Set the hidden (encoder) layer size: width x height x columnSize
    ld.hiddenSize = pyaon.Int3(4, 4, 16)

    ld.ffRadius = 4 # Sparse coder radius onto visible layers
    ld.pRadius = 4 # Predictor radius onto sparse coder hidden layer (and feed back)
    ld.aRadius = 4 # Actor radius onto sparse coder hidden layer (and feed back)

    ld.ticksPerUpdate = 2 # How many ticks before a layer updates (compared to previous layer) - clock speed for exponential memory
    ld.temporalHorizon = 4 # Memory horizon of the layer. Must be greater or equal to ticksPerUpdate
    
    lds.append(ld)

# Create the hierarchy: Provided with input layer sizes (a single column in this case), and input types (a single predicted layer)
h = pyaon.Hierarchy([ pyaon.Int3(1, numObs, obsColumnSize), pyaon.Int3(1, 1, numActions) ], [ pyaon.inputTypeNone, pyaon.inputTypeAction ], lds)

reward = 0.0

for episode in range(1000):
    obs = env.reset()

    # Timesteps
    for t in range(500):
        # Bin the 4 observations. Since we don't know the limits of the observation, we just squash it
        binnedObs = (sigmoid(obs * obsSquashScale) * (obsColumnSize - 1) + 0.5).astype(np.int).ravel().tolist()

        h.step([ binnedObs, h.getPredictionCs(1) ], True, reward)

        # Retrieve the action, the hierarchy already automatically applied exploration
        action = h.getPredictionCs(1)[0] # First and only column

        obs, reward, done, info = env.step(action)

        # Re-define reward so that it is 0 normally and then -1 if done
        if done:
            reward = -1.0

            print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))

            break
        else:
            reward = 0.0