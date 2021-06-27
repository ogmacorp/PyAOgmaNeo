# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import pyaogmaneo as pyaon
import numpy as np
import gym
import cv2
import os
from copy import copy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

inputTypePrediction = 0
inputTypeAction = 1

class EnvRunner:
    def __init__(self, env, layerSizes=8 * [ (5, 5, 16) ], layerRadius=2, hiddenSize=(8, 8, 16), imageRadius=8, imageScale=1.0, obsResolution=32, actionResolution=16, rewardScale=1.0, terminalReward=0.0, infSensitivity=1.0, nThreads=8):
        self.env = env

        pyaon.setNumThreads(nThreads)

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
            self.inputTypes.append(inputTypePrediction)
            self.inputLows.append([ 0.0 ])
            self.inputHighs.append([ 0.0 ])
        elif type(self.env.observation_space) is gym.spaces.Box:
            if len(self.env.observation_space.shape) == 1 or len(self.env.observation_space.shape) == 0:
                squareSize = int(np.ceil(np.sqrt(len(self.env.observation_space.low))))
                self.inputSizes.append((squareSize, squareSize, obsResolution))
                self.inputTypes.append(inputTypePrediction)
                lows = list(self.env.observation_space.low)
                highs = list(self.env.observation_space.high)
                
                # Detect large numbers/inf
                for i in range(len(lows)):
                    if abs(lows[i]) > 100000 or abs(highs[i]) > 100000:
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
                vld = pyaon.ImageEncoderVisibleLayerDesc((self.imageSizes[i][0], self.imageSizes[i][1], self.imageSizes[i][2]), imageRadius)

                vlds.append(vld)

                self.imgsPrev.append(np.zeros(self.imageSizes[i]))

            self.imEnc = pyaon.ImageEncoder()
            self.imEnc.initRandom(hiddenSize, vlds)

            self.imEncIndex = len(self.inputSizes)
            self.inputSizes.append(hiddenSize)
            self.inputTypes.append(inputTypePrediction)
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

        histCap = 8

        for i in range(len(layerSizes)):
            ld = pyaon.LayerDesc(hiddenSize=layerSizes[i])

            ld.eRadius = layerRadius
            ld.dRadius = layerRadius
            ld.historyCapacity = histCap

            ld.ticksPerUpdate = 2
            ld.temporalHorizon = 2

            lds.append(ld)

        self.h = pyaon.Hierarchy()

        ioDescs = []

        for i in range(len(self.inputSizes)):
            ioDescs.append(pyaon.IODesc(self.inputSizes[i], layerRadius, layerRadius, histCap))

        self.h.initRandom(ioDescs, lds)

        self.actions = []

        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            size = len(self.inputLows[index])

            startAct = []

            for _ in range(size):
                startAct.append(np.random.randint(0, self.inputSizes[index][2]))

            self.actions.append(startAct)

        goalSize = self.h.getHiddenSize(self.h.getNumLayers() - 1)
        self.goalSize = goalSize

        self.goal = np.random.randint(0, goalSize[2], size=(goalSize[0] * goalSize[1],))

        self.goalRewards = -np.random.randn(goalSize[0] * goalSize[1], goalSize[2]) * 0.0001 - 1.0

        self.goalLR = 0.01

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
                        indices.append(int((obs[j] - self.inputLows[i][j]) / (self.inputHighs[i][j] - self.inputLows[i][j]) * (self.inputSizes[i][2] - 1) + 0.5))
                    elif self.inputLows[i][j] > self.inputHighs[i][j]: # Inf
                        v = obs[j]

                        # Rescale
                        indices.append(int(sigmoid(v * self.infSensitivity) * (self.inputSizes[i][2] - 1) + 0.5))
                    else:
                        indices.append(int(obs[j]))

                if len(indices) < self.inputSizes[i][0] * self.inputSizes[i][1]:
                    indices += ((self.inputSizes[i][0] * self.inputSizes[i][1]) - len(indices)) * [ int(0) ]

                self.inputs.append(indices)

    def act(self, epsilon=0.05, obsPreprocess=None):
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

        # Update goal
        indices = copy(self.goal)#np.array(self.h.getHiddenCIs(self.h.getNumLayers() - 1), dtype=np.int)#

        indices = [ indices[i] + i * self.goalSize[2] for i in range(len(indices)) ]

        self.goalRewards.ravel()[indices] += self.goalLR * (r - self.goalRewards.ravel()[indices])

        if np.random.rand() < 0.01:
            self.goal = np.argmax(self.goalRewards, axis=1)
        
            mask = np.random.rand(self.goal.shape[0]) < 0.1

            self.goal[mask] = np.random.randint(0, self.goalSize[2], self.goal.shape[0])[mask]

        self.h.step(self.inputs, self.goal.ravel().tolist(), True)

        # Retrieve actions
        for i in range(len(self.actionIndices)):
            index = self.actionIndices[i]

            assert(self.inputTypes[index] is inputTypeAction)

            self.actions[i] = list(self.h.getPredictionCIs(index))
        
        return done, reward
