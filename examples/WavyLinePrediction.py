# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import pyaogmaneo as neo
import numpy as np
import matplotlib.pyplot as plt
import struct
import time

# Set the number of threads
neo.setNumThreads(8)

def fToCSDR(x, num_columns, cells_per_column, scale_factor=0.25):
    csdr = []

    scale = 1.0

    for i in range(num_columns):
        s = (x / scale) % (1.0 if x > 0.0 else -1.0)

        csdr.append(int((s * 0.5 + 0.5) * (cells_per_column - 1) + 0.5))

        rec = scale * (float(csdr[i]) / float(cells_per_column - 1) * 2.0 - 1.0)
        x -= rec

        scale *= scale_factor

    return csdr

def CSDRToF(csdr, cells_per_column, scale_factor=0.25):
    x = 0.0

    scale = 1.0

    for i in range(len(csdr)):
        x += scale * (float(csdr[i]) / float(cells_per_column - 1) * 2.0 - 1.0)

        scale *= scale_factor

    return x

def Unorm8ToCSDR(x : float):
    assert(x >= 0.0 and x <= 1.0)

    i = int(x * 255.0 + 0.5) & 0xff

    return [ int(i & 0x0f), int((i & 0xf0) >> 4) ]

# Reverse transform of IEEEToCSDR
def CSDRToUnorm8(csdr):
    return (csdr[0] | (csdr[1] << 4)) / 255.0

# This defines the resolution of the input encoding - we are using a simple single column that represents a bounded scalar through a one-hot encoding. This value is the number of "bins"
numInputColumns = 2
inputColumnSize = 16

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(6): # Layers with exponential memory
    ld = neo.LayerDesc()

    ld.hiddenSize = (5, 5, 16) # Size of the encoder (SparseCoder)

    lds.append(ld)

# Create the hierarchy
h = neo.Hierarchy()
h.initRandom([ neo.IODesc(size=(1, 2, 16)) ], [ neo.GDesc(size=(1, 1, 16)) ], lds)

# Present the wave sequence for some timesteps
iters = 1000

def wave(t):
    return (np.sin(t * 0.05 * 2.0 * np.pi + 0.5)) * 0.5 + 0.5

total = 0.0

for t in range(iters):
    # The value to encode into the input column
    valueToEncode = wave(t) # Some wavy line

    #csdr = fToCSDR(valueToEncode, numInputColumns, inputColumnSize)
    csdr = Unorm8ToCSDR(float(valueToEncode))

    start = time.time()

    # Step the hierarchy given the inputs (just one here)
    h.step([ csdr ], [ [ 0 ] ], [ [ 0 ] ], True) # True for enabling learning

    end = time.time()

    total += end - start

    # Print progress
    if t % 100 == 0:
        print(t)

print("Total: " + str(total))

# Recall the sequence
ts = [] # Time step
vs = [] # Predicted value

trgs = [] # True value

for t2 in range(300):
    t = t2 + iters # Continue where previous sequence left off

    # New, continued value for comparison to what the hierarchy predicts
    valueToEncode = wave(t) # Some wavy line

    start = time.time()

    # Run off of own predictions with learning disabled
    h.step([ h.getPredictionCIs(0) ], [ [ 0 ] ], [ [ 0 ] ], False) # Learning disabled

    end = time.time()

    total += end - start

    # Decode value (de-bin)
    #value = CSDRToF(h.getPredictionCIs(0), inputColumnSize) * maxRange
    value = CSDRToUnorm8(h.getPredictionCIs(0))

    # Append to plot data
    ts.append(t2)
    vs.append(value)

    trgs.append(valueToEncode)

    # Show predicted value
    #print(value)

# Show plot
plt.plot(ts, vs, ts, trgs)

#for i in range(len(units)):
#    plt.plot(ts, units[i])

plt.show()


