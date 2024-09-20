# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import pyaogmaneo as neo
import numpy as np
import matplotlib.pyplot as plt

# set the number of threads
neo.set_num_threads(4)

v1 = neo.Vec64_32.randomized()

print(v1)


# this defines the resolution of the input encoding
num_input_columns = 2
input_column_size = 16

# define layer descriptors: parameters of each layer upon creation
lds = []

for i in range(10): # layers with exponential memory
    ld = neo.LayerDesc()

    ld.hidden_size = (5, 5, 16) # size of the encoder(s) in the layer

    lds.append(ld)

# create the hierarchy with a single IO layer of size (1 x num_input_columns x input_column_size) and type prediction
h = neo.Hierarchy([ neo.IODesc(size=(1, num_input_columns, input_column_size), io_type=neo.prediction) ], lds)

h.params.anticipation = True # Anticipation mode, faster learning of long sequences at the cost of some extra compute

# present the wave sequence for some timesteps, 1000 here
iters = 50000

# function for the wave
def wave(t):
    return float(t % 50 == 0 or t % 7 == 0)#np.sin(t * 0.05 * 2.0 * np.pi + 0.5) * np.sin(t * 0.04 * 2.0 * np.pi - 0.4) * 0.5 + 0.5

# iterate
for t in range(iters):
    value_to_encode = wave(t)

    # encode
    csdr = unorm8_to_csdr(float(value_to_encode))

    # step the hierarchy given the inputs (just one here)
    h.step([ csdr ], True) # true for enabling learning

    print(h.get_hidden_cis(0))
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


