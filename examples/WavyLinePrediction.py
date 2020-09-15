# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import pyaogmaneo as pyaon
import numpy as np
import matplotlib.pyplot as plt

# Set the number of threads
pyaon.setNumThreads(8)

# This defines the resolution of the input encoding - we are using a simple single column that represents a bounded scalar through a one-hot encoding. This value is the number of "bins"
inputColumnSize = 64

# The bounds of the scalar we are encoding (low, high)
bounds = (-1.0, 1.0)

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(8): # Layers with exponential memory
    ld = pyaon.LayerDesc()

    # Set the hidden (encoder) layer size: width x height x columnSize
    ld.hiddenSize = pyaon.Int3(4, 4, 16)

    ld.ffRadius = 4 # Sparse coder radius onto visible layers
    ld.pRadius = 4 # Predictor radius onto sparse coder hidden layer (and feed back)

    ld.ticksPerUpdate = 2 # How many ticks before a layer updates (compared to previous layer) - clock speed for exponential memory
    ld.temporalHorizon = 4 # Memory horizon of the layer. Must be greater or equal to ticksPerUpdate
    
    lds.append(ld)

# Create the hierarchy: Provided with input layer sizes (a single column in this case), and input types (a single predicted layer)
h = pyaon.Hierarchy([ pyaon.Int3(1, 1, inputColumnSize) ], [ pyaon.inputTypePrediction ], lds)

# Present the wave sequence for some timesteps
iters = 1000

def wave(t):
    return float(t % 19 == 0)#np.sin(t * 0.02 * 2.0 * np.pi) * np.sin(t * 0.04 * 2.0 * np.pi + 0.45)

for t in range(iters):
    # The value to encode into the input column
    valueToEncode = wave(t) # Some wavy line

    valueToEncodeBinned = int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (inputColumnSize - 1) + 0.5)

    # Step the hierarchy given the inputs (just one here)
    h.step([ [ valueToEncodeBinned ] ], True) # True for enabling learning

    # Print progress
    if t % 100 == 0:
        print(t)

# Recall the sequence
ts = [] # Time step
vs = [] # Predicted value
trgs = [] # True value

for t2 in range(500):
    t = t2 + iters # Continue where previous sequence left off

    # New, continued value for comparison to what the hierarchy predicts
    valueToEncode = wave(t) # Some wavy line

    # Bin the value into the column and write into the input buffer. We are simply rounding to the nearest integer location to "bin" the scalar into the column
    valueToEncodeBinned = int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (inputColumnSize - 1) + 0.5)

    # Run off of own predictions with learning disabled
    h.step([ h.getPredictionCs(0) ], False) # Learning disabled

    predIndex = h.getPredictionCs(0)[0] # First (only in this case) input layer prediction
    
    # Decode value (de-bin)
    value = predIndex / float(inputColumnSize - 1) * (bounds[1] - bounds[0]) + bounds[0]

    # Append to plot data
    ts.append(t2)
    vs.append(value)
    trgs.append(valueToEncode)

    # Show predicted value
    print(value)

# Show plot
plt.plot(ts, vs, ts, trgs)
plt.show()


