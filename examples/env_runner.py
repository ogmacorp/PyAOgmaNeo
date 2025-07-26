# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import pyaogmaneo as neo
import numpy as np
import gymnasium as gym
import tinyscaler
import os
import time

def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5

input_type_none = neo.none
input_type_prediction = neo.prediction
input_type_action = neo.action

class EnvRunner:
    def _handle_nodict_obs_space(self, obs_space, obs_resolution, hidden_size, image_scale, image_radius, key=None):
        match type(obs_space):
            case gym.spaces.Discrete:
                self.input_sizes.append((1, 1, obs_space.n))
                self.input_types.append(input_type_none)
                self.input_lows.append([0.0])
                self.input_highs.append([0.0])
                self.input_encs.append(-1)
            case gym.spaces.multi_discrete:
                square_size = int(np.ceil(np.sqrt(len(obs_space.nvec))))
                high = np.max(obs_space.nvec)

                self.input_sizes.append((square_size, square_size, high))
                self.input_types.append(input_type_none)
                self.input_lows.append([0.0])
                self.input_highs.append([0.0])
                self.input_encs.append(-1)
            case gym.spaces.Box:
                match obs_space.shape:
                    case ():
                        return

                if len(obs_space.shape) == 1 or len(obs_space.shape) == 0:
                    square_size = int(np.ceil(np.sqrt(len(obs_space.low))))
                    self.input_sizes.append((square_size, square_size, obs_resolution))
                    self.input_types.append(input_type_none)
                    lows = obs_space.low
                    highs = obs_space.high
                    
                    # detect large numbers/inf
                    for i in range(len(lows)):
                        if abs(lows[i]) > 10000 or abs(highs[i]) > 10000:
                            # indicate inf by making low greater than high
                            lows[i] = 1.0
                            highs[i] = -1.0

                    self.input_lows.append(lows)
                    self.input_highs.append(highs)
                    self.input_encs.append(-1)
                elif len(obs_space.shape) == 2 or len(obs_space.shape) == 3:
                    scaled_size = (int(obs_space.shape[0] * image_scale), int(obs_space.shape[1] * image_scale), 1 if len(obs_space.shape) == 2 else obs_space.shape[2])

                    self.image_sizes.append(scaled_size)

                    image_enc = neo.ImageEncoder(hidden_size, [neo.ImageVisibleLayerDesc(scaled_size, image_radius)])

                    self.input_sizes.append(hidden_size)
                    self.input_types.append(input_type_none)
                    self.input_lows.append([0.0])
                    self.input_highs.append([1.0])
                    self.input_encs.append(len(self.image_encs))

                    self.image_encs.append(image_enc)
                else:
                    raise Exception("unsupported Box input: dimensions too high " + str(obs_space.shape))
            case _:
                raise Exception("unsupported input type " + str(type(obs_space)))

        self.input_keys.append(key)

    def __init__(self, env, layer_sizes=2 * [(5, 5, 64)],
        num_dendrites_per_cell=4,
        input_radius=2, layer_radius=2, hidden_size=(10, 10, 16),
        image_radius=8, image_scale=0.5, obs_resolution=16, action_resolution=16, action_importance=0.5,
        reward_scale=1.0, terminal_reward=0.0, inf_sensitivity=2.0, n_threads=4
    ):
        self.env = env

        neo.set_num_threads(n_threads)
        neo.set_global_state(int(time.time()))

        self.im_enc = None
        self.im_enc_index = -1

        self.input_sizes = []
        self.input_lows = []
        self.input_highs = []
        self.input_types = []
        self.input_keys = []
        self.input_encs = []
        self.image_encs = []
        self.image_sizes = []
        self.action_indices = []

        self.reward_scale = reward_scale
        self.terminal_reward = terminal_reward

        self.inf_sensitivity = inf_sensitivity

        obs_space = env.observation_space

        if type(obs_space) is gym.spaces.Dict:
            for key, value in obs_space.items():
                self._handle_nodict_obs_space(value, obs_resolution, hidden_size, image_scale, image_radius, key=key)
        else:
            self._handle_nodict_obs_space(obs_space, obs_resolution, hidden_size, image_scale, image_radius)

        # actions
        if type(self.env.action_space) is gym.spaces.Discrete:
            self.action_indices.append(len(self.input_sizes))
            self.input_sizes.append((1, 1, self.env.action_space.n))
            self.input_types.append(input_type_action)
            self.input_lows.append([0.0])
            self.input_highs.append([0.0])
            self.input_encs.append(-1)
            self.input_keys.append(None)
        elif type(self.env.action_space) is gym.spaces.multi_discrete:
            square_size = int(np.ceil(np.sqrt(len(self.env.action_space.nvec))))
            high = np.max(self.env.action_space.nvec)

            self.action_indices.append(len(self.input_sizes))
            self.input_sizes.append((square_size, square_size, high))
            self.input_types.append(input_type_action)
            self.input_lows.append([0.0])
            self.input_highs.append([0.0])
            self.input_encs.append(-1)
            self.input_keys.append(None)
        elif type(self.env.action_space) is gym.spaces.Box:
            if len(self.env.action_space.shape) < 3:
                if len(self.env.action_space.shape) == 2:
                    self.action_indices.append(len(self.input_sizes))
                    self.input_sizes.append((self.env.action_space.shape[0], self.env.action_space.shape[1], action_resolution))
                    self.input_types.append(input_type_action)
                    self.input_keys.append(None)
                    lows = self.env.action_space.low
                    highs = self.env.action_space.high

                    self.input_lows.append(lows)
                    self.input_highs.append(highs)
                    self.input_encs.append(-1)
                else:
                    square_size = int(np.ceil(np.sqrt(len(self.env.action_space.low))))
                    self.action_indices.append(len(self.input_sizes))
                    self.input_sizes.append((square_size, square_size, action_resolution))
                    self.input_types.append(input_type_action)
                    self.input_keys.append(None)
                    lows = self.env.action_space.low
                    highs = self.env.action_space.high

                    self.input_lows.append(lows)
                    self.input_highs.append(highs)
                    self.input_encs.append(-1)
            else:
                raise Exception("unsupported Box action: dimensions too high " + str(self.env.action_space.shape))
        else:
            raise Exception("unsupported action type " + str(type(self.env.action_space)))

        lds = []

        for i in range(len(layer_sizes)):
            ld = neo.LayerDesc()

            ld.hidden_size = layer_sizes[i]
            ld.num_dendrites_per_cell = num_dendrites_per_cell
            ld.up_radius = layer_radius
            ld.down_radius = layer_radius

            lds.append(ld)

        io_descs = []

        for i in range(len(self.input_sizes)):
            io_descs.append(neo.IODesc(self.input_sizes[i], self.input_types[i], num_dendrites_per_cell=num_dendrites_per_cell, up_radius=input_radius, down_radius=layer_radius))

        self.h = neo.Hierarchy(io_descs, lds)

        self.actions = []

        for i in range(len(self.action_indices)):
            index = self.action_indices[i]

            self.h.params.ios[index].importance = action_importance

            size = self.h.get_io_size(index)[0] * self.h.get_io_size(index)[1]

            start_act = []

            for _ in range(size):
                start_act.append(np.random.randint(0, self.input_sizes[index][2]))

            self.actions.append(start_act)

        self.actions = np.array(self.actions, np.int32)

        self.obs_space = obs_space

        self.learn_enabled = True

    def _feed_observation(self, obs):
        self.inputs = []

        action_index = 0
        image_enc_index = 0

        for i in range(len(self.input_sizes)):
            sub_obs = obs

            if self.input_keys[i] is not None:
                sub_obs = sub_obs[self.input_keys[i]]

            if self.input_types[i] == input_type_action:
                self.inputs.append(self.actions[action_index])

                action_index += 1
            elif self.input_encs[i] != -1:
                # format image
                img = tinyscaler.scale((sub_obs - self.input_lows[i]) / (self.input_highs[i][0] - self.input_lows[i][0]),
                                       (self.image_sizes[image_enc_index][1], self.image_sizes[image_enc_index][0]))
                
                # encode image
                self.image_encs[image_enc_index].step([img.astype(np.uint8).ravel()], True)

                self.inputs.append(self.image_encs[image_enc_index].get_hidden_cis())

                image_enc_index += 1
            else:
                sub_obs = sub_obs.ravel()

                indices = []

                for j in range(len(self.input_lows[i])):
                    if self.input_lows[i][j] < self.input_highs[i][j]:
                        # rescale
                        #indices.append(int(min(1.0, max(0.0, (sub_obs[j] - self.input_lows[i][j]) / (self.input_highs[i][j] - self.input_lows[i][j]))) * (self.input_sizes[i][2] - 1) + 0.5))
                        indices.append(int(sigmoid(sub_obs[j] * self.inf_sensitivity) * (self.input_sizes[i][2] - 1) + 0.5))
                    elif self.input_lows[i][j] > self.input_highs[i][j]: # Inf
                        # Rescale
                        indices.append(int(sigmoid(sub_obs[j] * self.inf_sensitivity) * (self.input_sizes[i][2] - 1) + 0.5))
                    else:
                        if type(self.env.observation_space) is gym.spaces.multi_discrete:
                            indices.append(int(sub_obs[j]) % self.sub_obs_space.nvec[j])
                        else:
                            indices.append(int(sub_obs[j]))

                if len(indices) < self.input_sizes[i][0] * self.input_sizes[i][1]:
                    indices += ((self.input_sizes[i][0] * self.input_sizes[i][1]) - len(indices)) * [int(0)]

                self.inputs.append(np.array(indices, dtype=np.int32))

    def act(self, epsilon=0.02, obs_preprocess=None):
        feed_actions = []

        for i in range(len(self.action_indices)):
            index = self.action_indices[i]

            assert(self.input_types[index] == input_type_action)

            if self.input_lows[index][0] < self.input_highs[index][0]:
                feed_action = []

                # explore
                for j in range(len(self.input_lows[index])):
                    if np.random.rand() < epsilon:
                        self.actions[i][j] = np.random.randint(0, self.input_sizes[index][2])

                    if self.input_lows[index][j] < self.input_highs[index][j]:
                        feed_action.append(self.actions[i][j] / float(self.input_sizes[index][2] - 1) * (self.input_highs[index][j] - self.input_lows[index][j]) + self.input_lows[index][j])
                    else:
                        feed_action.append(self.actions[i][j])

                feed_actions.append(feed_action)
            else:
                if type(self.env.action_space) is gym.spaces.multi_discrete:
                    for j in range(len(self.env.action_space.nvec)):
                        if np.random.rand() < epsilon:
                            self.actions[i][j] = np.random.randint(0, self.input_sizes[index][2])

                        feed_actions.append(int(self.actions[i][j]))
                else:
                    if np.random.rand() < epsilon:
                        self.actions[i][0] = np.random.randint(0, self.input_sizes[index][2])

                    feed_actions.append(int(self.actions[i][0]))

        # remove outer array if needed
        if len(feed_actions) == 1:
            feed_actions = feed_actions[0]

        obs, reward, term, trunc, info = self.env.step(feed_actions)

        if obs_preprocess is not None:
            obs = obs_preprocess(obs)

        if type(obs) is not np.array:
            obs = np.array(obs)

        self._feed_observation(obs)

        r = reward * self.reward_scale + float(term) * self.terminal_reward

        start_time = time.perf_counter()

        self.h.step(self.inputs, self.learn_enabled, r)

        end_time = time.perf_counter()

        #if term or trunc:
        #    print((end_time - start_time) * 1000.0)

        # retrieve actions
        for i in range(len(self.action_indices)):
            index = self.action_indices[i]

            assert self.input_types[index] == input_type_action

            self.actions[i] = self.h.get_prediction_cis(index)

        return term or trunc, reward
