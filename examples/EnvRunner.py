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
import gym
import cv2
import os
from copy import copy
import time

def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5

inputTypeNone = neo.none
inputTypePrediction = neo.prediction
inputTypeAction = neo.action

class EnvRunner:
    def __init__(self, env, layerSizes=2 * [ (5, 5, 32) ], layerRadius=2, hiddenSize=(8, 8, 16), imageRadius=8, imageScale=1.0, obsResolution=32, actionResolution=9, rewardScale=1.0, terminalReward=0.0, infSensitivity=2.0, nThreads=4):
        self.env = env

        neo.setNumThreads(nThreads)
        neo.setGlobalState(int(time.time()))

        self.imEnc = None
        self.imEncIndex = -1

        self.inputSizes = []
        self.inputLows = []
        self.inputHighs = []
        self.inputTypes = []
        self.imageSizes = []
        self.imgsPrev = []
        self.actionIndices = []

        self.rewardScale = rewardScale
        self.terminalReward = terminalReward

        self.infSensitivity = infSensitivity

        if type(self.env.observation_space) is gym.spaces.Discrete:
            self.inputSizes.append((1, 1, self.env.observation_space.n))
            self.inputTypes.append(inputTypeNone)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 0.0 ])
        elif type(self.env.observation_space) is gym.spaces.Box:
            if len(self.env.observation_space.shape) == 1 or len(self.env.observation_space.shape) == 0:
                squareSize = int(np.ceil(np.sqrt(len(self.env.observation_space.low))))
                self.inputSizes.append((squareSize, squareSize, obsResolution))
                self.inputTypes.append(inputTypeNone)
                lows = list(self.env.observation_space.low)
                highs = list(self.env.observation_space.high)
                
                # Detect large numbers/inf
                for i in range(len(lows)):
                    if abs(lows[i]) > 10000 or abs(highs[i]) > 10000:
                        # Indicate inf by making low greater than high
                        lows[i] = 1.0
                        highs[i] = -1.0

                self.inputLows.append(lows)
                self.inputHighs.append(highs)
            elif len(self.env.observation_space.shape) == 2:
                scaledSize = ( int(self.env.observation_space.shape[0] * imageScale), int(self.env.observation_space.shape[1] * imageScale), 1 )

                self.imageSizes.append(scaledSize)
            elif len(self.env.observation_space.shape) == 3:
                scaledSize = ( int(self.env.observation_space.shape[0] * imageScale), int(self.env.observation_space.shape[1] * imageScale), 3 )

                self.imageSizes.append(scaledSize)
            else:
                raise Exception("Unsupported Box input: Dimensions too high " + str(self.env.observation_space.shape))
        else:
            raise Exception("Unsupported input type " + str(type(self.env.observation_space)))

        if len(self.imageSizes) > 0:
            vlds = []

            for i in range(len(self.imageSizes)):
                vld = neo.ImageEncoderVisibleLayerDesc((self.imageSizes[i][0], self.imageSizes[i][1], self.imageSizes[i][2]), imageRadius)

                vlds.append(vld)

                self.imgsPrev.append(np.zeros(self.imageSizes[i]))

            self.imEnc = neo.ImageEncoder()
            self.imEnc.initRandom(hiddenSize, vlds)

            self.imEncIndex = len(self.inputSizes)
            self.inputSizes.append(hiddenSize)
            self.inputTypes.append(inputTypeNone)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 1.0 ])

        # Actions
        if type(self.env.action_space) is gym.spaces.Discrete:
            self.actionIndices.append(len(self.inputSizes))
            self.inputSizes.append((1, 1, self.env.action_space.n))
            self.inputTypes.append(inputTypeAction)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 0.0 ])
        elif type(self.env.action_space) is gym.spaces.Box:
            if len(self.env.action_space.shape) < 3:
                if len(self.env.action_space.shape) == 2:
                    self.actionIndices.append(len(self.inputSizes))
                    self.inputSizes.append((self.env.action_space.shape[0], self.env.action_space.shape[1], actionResolution))
                    self.inputTypes.append(inputTypeAction)
                    lows = list(self.env.action_space.low)
                    highs = list(self.env.action_space.high)

                    self.inputLows.append(lows)
                    self.inputHighs.append(highs)
                else:
                    squareSize = int(np.ceil(np.sqrt(len(self.env.action_space.low))))
                    self.actionIndices.append(len(self.inputSizes))
                    self.inputSizes.append((squareSize, squareSize, actionResolution))
                    self.inputTypes.append(inputTypeAction)
                    lows = list(self.env.action_space.low)
                    highs = list(self.env.action_space.high)

                    self.inputLows.append(lows)
                    self.inputHighs.append(highs)
            else:
                raise Exception("Unsupported Box action: Dimensions too high " + str(self.env.action_space.shape))
        else:
            raise Exception("Unsupported action type " + str(type(self.env.action_space)))

        lds = []

        for i in range(len(layerSizes)):
            ld = neo.LayerDesc(hiddenSize=layerSizes[i])

            ld.eRadius = layerRadius
            ld.dRadius = layerRadius

            lds.append(ld)

        self.h = neo.Hierarchy()

        ioDescs = []

        for i in range(len(self.inputSizes)):
            ioDescs.append(neo.IODesc(self.inputSizes[i], self.inputTypes[i], layerRadius, layerRadius))

        self.h.initRandom(ioDescs, lds)

        self.actions = []

        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            #self.h.setImportance(index, 0.01)

            size = self.h.getIOSize(index)[0] * self.h.getIOSize(index)[1]

            startAct = []

            for _ in range(size):
                startAct.append(np.random.randint(0, self.inputSizes[index][2]))

            self.actions.append(startAct)

        self.averageReward = -1.0
        self.averageRewardDecay = 0.01

    def _feedObservation(self, obs):
        self.inputs = []

        actionIndex = 0

        for i in range(len(self.inputSizes)):
            if self.inputTypes[i] == inputTypeAction:
                self.inputs.append(self.actions[actionIndex])

                actionIndex += 1
            elif i == self.imEncIndex:
                # Format image
                img = cv2.resize(obs, (self.imageSizes[0][0], self.imageSizes[0][1]))
                
                img = np.swapaxes(img, 0, 1)
                
                #delta = img - self.imgsPrev[0]
 
                self.imgsPrev[0] = copy(img)

                # Encode image
                self.imEnc.step([ img.ravel().tolist() ], True)

                self.inputs.append(list(self.imEnc.getHiddenCIs()))

            else:
                indices = []

                for j in range(len(self.inputLows[i])):
                    if self.inputLows[i][j] < self.inputHighs[i][j]:
                        # Rescale
                        indices.append(int(min(1.0, max(0.0, (obs[j] - self.inputLows[i][j]) / (self.inputHighs[i][j] - self.inputLows[i][j]))) * (self.inputSizes[i][2] - 1) + 0.5))
                        #v = obs[j]
                        #indices.append(int(sigmoid(v * self.infSensitivity) * (self.inputSizes[i][2] - 1) + 0.5))
                    elif self.inputLows[i][j] > self.inputHighs[i][j]: # Inf
                        v = obs[j]

                        # Rescale
                        indices.append(int(sigmoid(v * self.infSensitivity) * (self.inputSizes[i][2] - 1) + 0.5))
                    else:
                        indices.append(int(obs[j]))

                if len(indices) < self.inputSizes[i][0] * self.inputSizes[i][1]:
                    indices += ((self.inputSizes[i][0] * self.inputSizes[i][1]) - len(indices)) * [ int(0) ]

                self.inputs.append(indices)

    def act(self, epsilon=0.0, obsPreprocess=None):
        feedActions = []

        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            assert(self.inputTypes[index] == inputTypeAction)

            if self.inputLows[index][0] < self.inputHighs[index][0]:
                feedAction = []

                # Explore
                for j in range(len(self.inputLows[index])):
                    if np.random.rand() < epsilon:
                        self.actions[i][j] = np.random.randint(0, self.inputSizes[index][2])

                    if self.inputLows[index][j] < self.inputHighs[index][j]:
                        feedAction.append(self.actions[i][j] / float(self.inputSizes[index][2] - 1) * (self.inputHighs[index][j] - self.inputLows[index][j]) + self.inputLows[index][j])
                    else:
                        feedAction.append(self.actions[i][j])

                feedActions.append(feedAction)
            else:
                if np.random.rand() < epsilon:
                    self.actions[i][0] = np.random.randint(0, self.inputSizes[index][2])

                feedActions.append(int(self.actions[i][0]))

        # Remove outer array if needed
        if len(feedActions) == 1:
            feedActions = feedActions[0]

        obs, reward, done, _ = self.env.step(feedActions)

        if obsPreprocess is not None:
            obs = obsPreprocess(obs)

        self._feedObservation(obs)

        r = reward * self.rewardScale + float(done) * self.terminalReward

        self.averageReward += self.averageRewardDecay * (r - self.averageReward)

        self.h.step(self.inputs, True, r)

        # Retrieve actions
        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            assert(self.inputTypes[index] == inputTypeAction)

            self.actions[i] = list(self.h.getPredictionCIs(index))
        
        return done, reward
