# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import pyaogmaneo as neo
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import struct

#matplotlib.use('TkAgg')

# set the number of threads
neo.set_num_threads(4)

# scalar encoding used in this example, take a byte and convert 4 consective bits into 2 one-hot columns with 16 cells in them

def unorm8_to_csdr(x : float):
    assert(x >= 0.0 and x <= 1.0)

    i = int(x * 255.0 + 0.5) & 0xff

    return [ int(i & 0x0f), int((i & 0xf0) >> 4) ]

# reverse transform of unorm8_to_csdr
def csdr_to_unorm8(csdr):
    return (csdr[0] | (csdr[1] << 4)) / 255.0

# some other ways of encoding individual scalers:

# multi-scale embedding
def f_to_csdr(x, num_columns, cells_per_column, scale_factor=0.25):
    csdr = []

    scale = 1.0

    for i in range(num_columns):
        s = (x / scale) % (1.0 if x > 0.0 else -1.0)

        csdr.append(int((s * 0.5 + 0.5) * (cells_per_column - 1) + 0.5))

        rec = scale * (float(csdr[i]) / float(cells_per_column - 1) * 2.0 - 1.0)
        x -= rec

        scale *= scale_factor

    return csdr

def csdr_to_f(csdr, cells_per_column, scale_factor=0.25):
    x = 0.0

    scale = 1.0

    for i in range(len(csdr)):
        x += scale * (float(csdr[i]) / float(cells_per_column - 1) * 2.0 - 1.0)

        scale *= scale_factor

    return x

# convert an ieee float to 8 columns with 16 cells each (similar to first approach but on floating-point data)
def ieee_to_csdr(x : float):
    b = struct.pack("<f", x)

    csdr = []

    for i in range(4):
        csdr.append(b[i] & 0x0f)
        csdr.append((b[i] & 0xf0) >> 4)

    return csdr

def csdr_to_ieee(csdr):
    bs = []

    for i in range(4):
        bs.append(csdr[i * 2 + 0] | (csdr[i * 2 + 1] << 4))

    return struct.unpack("<f", bytes(bs))[0]

# this defines the resolution of the input encoding
num_input_columns = 2
input_column_size = 16

# define layer descriptors: parameters of each layer upon creation
lds = []

for i in range(3): # layers
    ld = neo.LayerDesc()

    ld.hidden_size = (5, 5, 64) # size of the encoder(s) in the layer

    lds.append(ld)

# create the hierarchy with a single IO layer of size (1 x num_input_columns x input_column_size) and type prediction
h = neo.Hierarchy([ neo.IODesc(size=(1, num_input_columns, input_column_size), io_type=neo.prediction) ], lds)

# present the wave sequence for some timesteps, 1000 here
iters = 10000

# function for the wave
def wave(t):
    if t % 20 == 0 or t % 7 == 0:
        return 1.0
    return 0.0
    return np.sin(t * 0.05 * 2.0 * np.pi + 0.5) * 0.5 + 0.5

# iterate
for t in range(iters):
    value_to_encode = wave(t)

    # encode
    csdr = unorm8_to_csdr(float(value_to_encode))

    # step the hierarchy given the inputs (just one here)
    h.step([ csdr ], True) # true for enabling learning

    msg = ""

    if value_to_encode > 0.5:
        msg = ">>>>>>>>>>>>>>>>>"

    msg += str(h.get_hidden_cis(0))

    print(msg)

    # print progress
    if t % 100 == 0:
        print(t)

# recall the sequence and plot the result
ts = [] # time step
vs = [] # predicted value

trgs = [] # true value

for t2 in range(1000):
    t = t2 + iters # get "continued" timestep (relative to previous training iterations)

    value_to_encode = wave(t)

    csdr = unorm8_to_csdr(float(value_to_encode))

    # run off of own predictions with learning disabled
    h.step([ h.get_prediction_cis(0) ], False) # learning disabled for recall

    # decode value from latest prediction
    value = csdr_to_unorm8(h.get_prediction_cis(0))

    # append to plot data
    ts.append(t2)
    vs.append(value + 1.1) # offset the plot by 1.1 so we can see it better

    trgs.append(value_to_encode)

    # show predicted value
    #print(value)

# show plot
plt.plot(ts, vs, ts, trgs)

plt.show()


