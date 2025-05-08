# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import pyaogmaneo as neo
import gymnasium as gym
import numpy as np
import struct

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

    return np.array(csdr, dtype=np.int32)

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

    return np.array(csdr, dtype=np.int32)

def csdr_to_ieee(csdr):
    bs = []

    for i in range(4):
        bs.append(csdr[i * 2 + 0] | (csdr[i * 2 + 1] << 4))

    return struct.unpack("<f", bytes(bs))[0]

# squashing function
def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5

# create the environment
env = gym.make('CartPole-v1')

# get observation size
num_obs = env.observation_space.shape[0] # 4 values for Cart-Pole
num_actions = env.action_space.n # N actions (1 discrete value)
input_resolution = 16

# set the number of threads
neo.set_num_threads(4)

# define layer descriptors: Parameters of each layer upon creation
lds = []

for i in range(1): # layers with exponential memory. Not much memory is needed for Cart-Pole, so we only use 2 layers
    ld = neo.LayerDesc()

    # set some layer structural parameters
    ld.hidden_size = (5, 5, 64)
    
    lds.append(ld)

# create the hierarchy
h = neo.Hierarchy([ neo.IODesc((2, 2, input_resolution), neo.none), neo.IODesc((1, 1, num_actions), neo.prediction), neo.IODesc((2, 4, 16), neo.prediction) ], lds)

input_history = []
max_history = 500
action = 0
reward = 0.0
future_state = h.serialize_state_to_buffer()
reward_bump = 0.1
exploration = 0.04

for episode in range(10000):
    obs, _ = env.reset()

    # timesteps
    for t in range(500):
        # sensory CSDR creation through "squash and bin" method
        csdr = (sigmoid(obs * 3.0) * (input_resolution - 1) + 0.5).astype(np.int32)

        input_history.append((csdr, action, reward))

        if len(input_history) > max_history:
            input_history = input_history[len(input_history) - max_history:]

        if len(input_history) == max_history:
            average_reward = 0.0

            for i in range(max_history):
                average_reward += input_history[i][2] * pow(0.97, i)

            average_reward /= max_history

            h.step([input_history[0][0], [input_history[0][1]], ieee_to_csdr(average_reward)], True)

        # save state
        old_state = h.serialize_state_to_buffer()

        h.set_state_from_buffer(future_state)

        pred_reward = csdr_to_ieee(h.get_prediction_cis(2))

        h.step([csdr, h.get_prediction_cis(1), ieee_to_csdr(pred_reward + reward_bump)], False)

        action = h.get_prediction_cis(1)[0]

        if np.random.rand() < exploration:
            action = np.random.randint(0, num_actions)

        future_state = h.serialize_state_to_buffer()

        h.set_state_from_buffer(old_state)

        obs, reward, term, trunc, _ = env.step(action)

        # re-define reward so that it is 0 normally and then -100 if terminated
        if term:
            reward = -100.0
        else:
            reward = 0.0

        if term or trunc:
            print(f"Episode {episode + 1} finished after {t + 1} timesteps")

            break
