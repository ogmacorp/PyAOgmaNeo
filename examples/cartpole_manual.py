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

def unorm8_to_csdr(x : float):
    assert(x >= 0.0 and x <= 1.0)

    v = int(x * 255.0 + 0.5) & 0xff

    return [(v >> (2 * i)) & 0x03 for i in range(4)]

# reverse transform of unorm8_to_csdr
def csdr_to_unorm8(csdr):
    r = int(0)

    for i in range(4):
        r |= int(csdr[i]) << (2 * i)

    return r / 255.0

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
env = gym.make('CartPole-v1')#, render_mode='human')

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

delay_capacity = 128

# create the hierarchy
h = neo.Hierarchy([neo.IODesc((2, 2, input_resolution), neo.none), neo.IODesc((1, 1, num_actions), neo.prediction), neo.IODesc((2, 2, 4), neo.prediction)], lds, delay_capacity)

rewards = []
pred_cumm_rewards = []

action = 0
pred_cumm_reward = 0.0
reward_bump = 1.0 / 255.0
exploration = 0.03
discount = 0.9
pred_bound = 999

for episode in range(10000):
    obs, _ = env.reset()

    # timesteps
    for t in range(500):
        # sensory CSDR creation through "squash and bin" method
        csdr = (sigmoid(obs * 4.0) * (input_resolution - 1) + 0.5).astype(np.int32)

        if h.delay_ready():
            r = 0.0
            d = 1.0

            for i in range(len(rewards)):
                r += rewards[i] * d
                d *= discount

            target = (r + d * pred_cumm_reward / (1.0 - discount)) * (1.0 - discount)

            td_error = target - pred_cumm_rewards[0]

            h.step_delayed([h.get_next_input_cis(0), h.get_next_input_cis(1), unorm8_to_csdr(min(1.0, max(0.0, target)))], True)

        h.step([csdr, [action], unorm8_to_csdr(min(1.0, pred_cumm_reward + reward_bump))], False)

        pred_cumm_reward = min(pred_bound, max(-pred_bound, csdr_to_unorm8(h.get_prediction_cis(2))))

        action = h.sample_prediction(1, 0.5)[0]

        if np.random.rand() < exploration:
            action = np.random.randint(0, num_actions)

        obs, reward, term, trunc, _ = env.step(action)

        # re-define reward
        if term:
            reward = -2.0
        else:
            reward = 1.0

        rewards.append(reward)
        pred_cumm_rewards.append(pred_cumm_reward)

        if len(rewards) > delay_capacity:
            rewards = rewards[1:]
            pred_cumm_rewards = pred_cumm_rewards[1:]

        if term or trunc:
            print(f"Episode {episode + 1} finished after {t + 1} timesteps")

            break
