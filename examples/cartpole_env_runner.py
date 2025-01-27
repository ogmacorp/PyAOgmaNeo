# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# Simple Cart-Pole example using EnvRunner

import gymnasium as gym
from env_runner import EnvRunner # EnvRunner automatically creates an AOgmaNeo hierarchy and appropriate encoders for most Gymnasium environments

env = gym.make('CartPole-v1')

runner = EnvRunner(env, terminal_reward=-100.0, reward_scale=0.0) # Cart-Pole environment always returns a reward of 1, so use a custom reward function: -1 if episode ends, 0 otherwise

for episode in range(10000):
    env.reset()

    # Timesteps
    for t in range(500):
        done, _ = runner.act() # Step the environment and agent

        if done:
            print(f"Episode {episode + 1} finished after {t + 1} timesteps")
            break
