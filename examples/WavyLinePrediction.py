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
inputColumnSize = 32

# The bounds of the scalar we are encoding (low, high)
bounds = (-1.0, 1.0)

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(9): # Layers with exponential memory
    ld = pyaon.LayerDesc()

    ld.hiddenSize = (4, 4, 16) # Size of the encoder (SparseCoder)

    lds.append(ld)

# Create the hierarchy
h = pyaon.Hierarchy()
h.initRandom([ pyaon.IODesc(size=(1, 1, inputColumnSize), type=pyaon.prediction, ffRadius=0) ], lds)

# Present the wave sequence for some timesteps
iters = 40000

def wave(t):
    return float(t % 21 == 0)#np.sin(t * 0.01 * 2.0 * np.pi - 1.0) * 0.9 + np.sin(t * 0.03 * 2.0 * np.pi - 1.0) * 0.1

for t in range(iters):
    # The value to encode into the input column
    valueToEncode = wave(t) # Some wavy line

    valueToEncodeBinned = int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (inputColumnSize - 1) + 0.5)

    # Step the hierarchy given the inputs (just one here)
    h.step([ [ valueToEncodeBinned ] ], True) # True for enabling learning

    print(h.getHiddenCIs(5))

    # Print progress
    if t % 100 == 0:
        print(t)

# Recall the sequence
ts = [] # Time step
vs = [] # Predicted value
units = []

for i in range(3):
    units.append([])

trgs = [] # True value

for t2 in range(300):
    t = t2 + iters # Continue where previous sequence left off

    # New, continued value for comparison to what the hierarchy predicts
    valueToEncode = wave(t) # Some wavy line

    # Bin the value into the column and write into the input buffer. We are simply rounding to the nearest integer location to "bin" the scalar into the column
    valueToEncodeBinned = int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (inputColumnSize - 1) + 0.5)

    # Run off of own predictions with learning disabled
    h.step([ h.getPredictionCIs(0) ], False) # Learning disabled

    predIndex = h.getPredictionCIs(0)[0] # First (only in this case) input layer prediction
    
    # Decode value (de-bin)
    value = predIndex / float(inputColumnSize - 1) * (bounds[1] - bounds[0]) + bounds[0]

    # Append to plot data
    ts.append(t2)
    vs.append(value)

    l = 0

    for i in range(len(units)):
        units[i].append(float(h.getHiddenCIs(l)[i]) / float(h.getHiddenSize(l)[2] - 1))

    trgs.append(valueToEncode)

    # Show predicted value
    #print(value)

# Show plot
plt.plot(ts, vs, ts, trgs)

#for i in range(len(units)):
#    plt.plot(ts, units[i])

plt.show()


