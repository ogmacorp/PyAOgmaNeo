# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# lunar lander environment with env_runner

import gymnasium as gym
from env_runner import EnvRunner # EnvRunner automatically creates an AOgmaNeo hierarchy and appropriate encoders for most Gymnasium environments
import time

env = gym.make('LunarLander-v2')

runner = EnvRunner(env, terminal_reward=0.0, reward_scale=1.0)

average_reward = 0.0
max_reward = 0.0

reward_window = []

start = time.time()

for episode in range(5000):
    env.reset()

    total_reward = 0.0

    # timesteps
    for t in range(10000):
        done, reward = runner.act() # step the environment and agent

        total_reward += reward

        if done:
            end = time.time()

            if episode == 0:
                average_reward = total_reward
                max_reward = total_reward
            else:
                average_reward = 0.99 * average_reward + 0.01 * total_reward
                max_reward = max(max_reward, total_reward)

            reward_window.append(total_reward)

            if len(reward_window) > 100:
                reward_window = reward_window[-100:]
            
            if len(reward_window) > 0:
                average_window = sum(reward_window) / 100.0

                if average_window >= 195.0:
                    print(f"SOLVED with {average_window} average 100 episode reward")
                    print(f"TIME: {end - start} seconds")

            print(f"Episode {episode + 1} finished after {t + 1} timesteps, receiving {total_reward} reward. Average: {average_reward} Max: {max_reward}")
            break
