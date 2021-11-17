# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

# Simple Cart-Pole example using EnvRunner

import gym
from EnvRunner import EnvRunner # EnvRunner automatically creates an OgmaNeo2 hierarchy and appropriate encoders for most Gym environments

env = gym.make('LunarLander-v2')

runner = EnvRunner(env, terminalReward=0.0, rewardScale=0.01)

averageReward = 0.0
maxReward = 0.0

for episode in range(5000):
    env.reset()

    totalReward = 0.0

    # Timesteps
    for t in range(10000):
        done, reward = runner.act() # Step the environment and agent

        totalReward += reward

        if done:
            if episode == 0:
                averageReward = totalReward
                maxReward = totalReward
            else:
                averageReward = 0.99 * averageReward + 0.01 * totalReward
                maxReward = max(maxReward, totalReward)

            print("Episode {} finished after {} timesteps, receiving {} reward. Average: {} Max: {}".format(episode + 1, t + 1, totalReward, averageReward, maxReward))
            break
