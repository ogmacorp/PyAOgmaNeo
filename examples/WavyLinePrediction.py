# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import pyaogmaneo as neo
import numpy as np
import matplotlib.pyplot as plt
import struct

# Set the number of threads
neo.setNumThreads(4)

# Encoding method to get 2 columns with 16 cells each from a byte
def Unorm8ToCSDR(x : float):
    assert(x >= 0.0 and x <= 1.0)

    i = int(x * 255.0 + 0.5) & 0xff

    return [ int(i & 0x0f), int((i & 0xf0) >> 4) ]

# Reverse transform of CSDRToUnorm8
def CSDRToUnorm8(csdr):
    return (csdr[0] | (csdr[1] << 4)) / 255.0

# Dimensions of the encoding
numInputColumns = 2
inputColumnSize = 16

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(7): # Layers with exponential memory
    ld = neo.LayerDesc()

    ld.hiddenSize = (4, 4, 16) # Size of the encoder

    ld.ticksPerUpdate = 2
    ld.temporalHorizon = 2

    lds.append(ld)

# Create the hierarchy
h = neo.Hierarchy()
h.initRandom([ neo.IODesc(size=(1, 2, 16), type=neo.prediction) ], lds)

# Present the wave sequence for some timesteps
iters = 10000

# The function we are modeling
def wave(t):
    if t % 40 == 0:
        return 1.0
    return 0.0
    return (np.sin(t * 0.05 * 2.0 * np.pi + 0.5)) * 0.5 + 0.5

for t in range(iters):
    # The value to encode into the input column
    valueToEncode = wave(t) # Some wavy line

    # Encode
    csdr = Unorm8ToCSDR(float(valueToEncode))

    # Step the hierarchy given the inputs (just one here)
    h.step([ csdr ], True) # True for enabling learning

    # Print progress
    if t % 100 == 0:
        print(t)

# Recall the sequence
ts = [] # Time step
vs = [] # Predicted value

trgs = [] # True value

for t2 in range(100):
    t = t2 + iters # Continue where previous sequence left off

    # New, continued value for comparison to what the hierarchy predicts
    valueToEncode = wave(t) # Some wavy line

    # Run off of own predictions with learning disabled
    h.step([ h.getPredictionCIs(0) ], False) # Learning disabled

    # Decode value (de-bin)
    value = CSDRToUnorm8(h.getPredictionCIs(0))

    # Append to plot data
    ts.append(t2)
    vs.append(value)

    trgs.append(valueToEncode)

    # Show predicted value
    #print(value)

# Show plot
plt.plot(ts, vs, ts, trgs)

plt.show()


