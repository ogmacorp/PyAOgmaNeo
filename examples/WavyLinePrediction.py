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

# Scalar encoding used in this example, take a byte and convert 4 consective bits into 2 one-hot columns with 16 cells in them

def Unorm8ToCSDR(x : float):
    assert(x >= 0.0 and x <= 1.0)

    i = int(x * 255.0 + 0.5) & 0xff

    return [ int(i & 0x0f), int((i & 0xf0) >> 4) ]

# Reverse transform of IEEEToCSDR
def CSDRToUnorm8(csdr):
    return (csdr[0] | (csdr[1] << 4)) / 255.0

# Some other ways of encoding individual scalers:

# Multi-scale embedding
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

# Convert an IEEE float to 8 columns with 16 cells each (similar to first approach but on floating-point data)
def IEEEToCSDR(x : float):
    b = struct.pack("<f", x)

    csdr = []

    for i in range(4):
        csdr.append(b[i] & 0x0f)
        csdr.append((b[i] & 0xf0) >> 4)

    return csdr

def CSDRToIEEE(csdr):
    bs = []

    for i in range(4):
        bs.append(csdr[i * 2 + 0] | (csdr[i * 2 + 1] << 4))

    return struct.unpack("<f", bytes(bs))[0]

# This defines the resolution of the input encoding
numInputColumns = 1
inputColumnSize = 32

# Define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(9): # Layers with exponential memory
    ld = neo.LayerDesc()

    ld.hiddenSize = (4, 4, 64) # Size of the encoder(s) in the layer

    lds.append(ld)

# Create the hierarchy
h = neo.Hierarchy()
h.initRandom([ neo.IODesc(size=(1, numInputColumns, inputColumnSize), type=neo.prediction) ], lds)

# Present the (noisy) wave sequence for some timesteps
iters = 50000

def wave(t):
    if t % 100 == 0:
        return 1.0
    return 0.0
    return min(1.0, max(0.0, (np.sin(t * 0.05 * 2.0 * np.pi + 0.5)) * np.sin(t * 0.04 * 2.0 * np.pi - 0.4) * 0.5 + 0.5 + np.random.randn() * 0.02))

for t in range(iters):
    valueToEncode = wave(t)

    csdr = [ int(valueToEncode * (inputColumnSize - 1) + 0.5) ]#Unorm8ToCSDR(float(valueToEncode))

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

for t2 in range(1000):
    t = t2 + iters

    valueToEncode = wave(t)

    csdr = [ int(valueToEncode * (inputColumnSize - 1) + 0.5) ]#Unorm8ToCSDR(float(valueToEncode))

    # Run off of own predictions with learning disabled
    h.step([ h.getPredictionCIs(0) ], False) # Learning disabled

    # Decode value (de-bin)
    value = h.getPredictionCIs(0)[0] / float(inputColumnSize - 1)

    # Append to plot data
    ts.append(t2)
    vs.append(value + 1.1)

    trgs.append(valueToEncode)

    # Show predicted value
    #print(value)

# Show plot
plt.plot(ts, vs, ts, trgs)

plt.show()


