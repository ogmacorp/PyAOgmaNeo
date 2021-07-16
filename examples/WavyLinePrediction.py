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
import struct

# Set the number of threads
pyaon.setNumThreads(8)

# Converts an IEEE float to 8 columns with 16 cells each
def IEEEToCSDR(x : float):
    b = struct.pack("<f", x)

    csdr = []

    for i in range(4):
        csdr.append(b[i] & 0x0f)
        csdr.append((b[i] & 0xf0) >> 4)

    return csdr

# Reverse transform of IEEEToCSDR
def CSDRToIEEE(csdr):
    bs = []

    for i in range(4):
        bs.append(csdr[i * 2 + 0] | (csdr[i * 2 + 1] << 4))

    return struct.unpack("<f", bytes(bs)) 

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(8): # Layers with exponential memory
    ld = pyaon.LayerDesc()

    ld.hiddenSize = (4, 4, 16) # Size of the encoder (SparseCoder)

    lds.append(ld)

# Create the hierarchy
h = pyaon.Hierarchy()
h.initRandom([ pyaon.IODesc(size=(2, 4, 16), type=pyaon.prediction) ], lds)

# Present the wave sequence for some timesteps
iters = 10000

def wave(t):
    return (np.sin(t * 0.05 * 2.0 * np.pi + 0.5) * np.sin(t * 0.1 * 2.0 * np.pi - 0.5) * np.sin(t * 0.02 * 2.0 * np.pi)) * 0.5 + 0.5

for t in range(iters):
    # The value to encode into the input column
    valueToEncode = wave(t) # Some wavy line

    csdr = IEEEToCSDR(float(valueToEncode))

    # Step the hierarchy given the inputs (just one here)
    h.step([ csdr ], True) # True for enabling learning

    print(h.getHiddenCIs(3))
    # Print progress
    if t % 100 == 0:
        print(t)

# Recall the sequence
ts = [] # Time step
vs = [] # Predicted value

trgs = [] # True value

for t2 in range(3000):
    t = t2 + iters # Continue where previous sequence left off

    # New, continued value for comparison to what the hierarchy predicts
    valueToEncode = wave(t) # Some wavy line

    # Run off of own predictions with learning disabled
    h.step([ h.getPredictionCIs(0) ], False) # Learning disabled

    # Decode value (de-bin)
    value = CSDRToIEEE(h.getPredictionCIs(0))

    # Append to plot data
    ts.append(t2)
    vs.append(value)

    trgs.append(valueToEncode)

# Show plot
plt.plot(ts, vs, ts, trgs)

plt.show()
