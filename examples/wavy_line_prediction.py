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

# set types
Vec = neo.Vec64_32
Bundle = neo.Bundle64_32
Hierarchy = neo.Hierarchy64_32

# set the number of threads
neo.set_num_threads(4)

vecs = []

for i in range(2):
    vecs.append(Vec.randomized())

# define layer descriptors: parameters of each layer upon creation
lds = []

for i in range(8):
    ld = neo.LayerDesc()

    ld.hidden_size = (2, 2) # size of the layer

    lds.append(ld)

h = Hierarchy([ neo.IODesc(size=(1, 1), io_type=neo.prediction) ], lds)

# present the wave sequence for some timesteps
iters = 10000

# function for the wave
def wave(t):
    return float(t % 20 == 0 or t % 7 == 0)#np.sin(t * 0.05 * 2.0 * np.pi + 0.5) * np.sin(t * 0.04 * 2.0 * np.pi - 0.4) * 0.5 + 0.5

# iterate
last_index = 0

for t in range(iters):
    value_to_encode = wave(t)

    # encode
    index = int(value_to_encode * (len(vecs) - 1) + 0.5)

    # step the hierarchy given the inputs (just one here)
    h.step([[ vecs[index] ]], True) # true for enabling learning

    v = h.get_prediction_vecs(0)[0]

    # decode
    max_index = 0
    max_similarity = -99999

    for i in range(len(vecs)):
        d = vecs[i].dot(v)

        if d > max_similarity:
            max_similarity = d
            max_index = i

    value = max_index / (len(vecs) - 1)

    last_index = max_index

    #print(h.get_hidden_vecs(0))

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

    # run off of own predictions with learning disabled
    h.step([[ vecs[last_index] ]], False) # learning disabled for recall

    v = h.get_prediction_vecs(0)[0]

    # decode
    max_index = 0
    max_similarity = -99999

    for i in range(len(vecs)):
        d = vecs[i].dot(v)

        if d > max_similarity:
            max_similarity = d
            max_index = i

    value = max_index / (len(vecs) - 1)

    last_index = max_index

    # append to plot data
    ts.append(t2)
    vs.append(value + 1.1) # offset the plot by 1.1 so we can see it better

    trgs.append(value_to_encode)

    # show predicted value
    #print(value)

# show plot
plt.plot(ts, vs, ts, trgs)

plt.show()


