# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# Lunar lander environment with EnvRunner

import gymnasium as gym
from env_runner import EnvRunner # EnvRunner automatically creates an AOgmaNeo hierarchy and appropriate encoders for most Gymnasium environments

env = gym.make('LunarLander-v3')#, render_mode='human')

runner = EnvRunner(env, terminal_reward=0.0, reward_scale=1.0)

average_reward = 0.0
max_reward = 0.0

for episode in range(5000):
    env.reset()

    total_reward = 0.0

    # timesteps
    for t in range(10000):
        done, reward = runner.act() # step the environment and agent

        total_reward += reward

        if done:
            if episode == 0:
                average_reward = total_reward
                max_reward = total_reward
            else:
                average_reward = 0.99 * average_reward + 0.01 * total_reward
                max_reward = max(max_reward, total_reward)

            print(f"Episode {episode + 1} finished after {t + 1} timesteps, receiving {total_reward} reward. Average: {average_reward} Max: {max_reward}")
            break
