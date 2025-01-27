<!---
  PyAOgmaNeo
  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of PyAOgmaNeo is licensed to you under the terms described
  in the PYAOGMANEO_LICENSE.md file included in this distribution.
--->

# Examples

## The EnvRunner

The env_runner is a simple way to automatically create AOgmaNeo systems for [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) tasks. It will automatically create the hierarchy and appropriate pre-encoders, by guessing reasonable settings. This is good enough for initial experimentation, but further control requires manual usage. Image-based environments require [tinyscaler](https://github.com/Farama-Foundation/TinyScaler) in order to scale images to appropriate sizes.

## CartPole examples

The CartPole examples are simple tests on the Gymnasium CartPole environment. The manual one sets up a hierarchy manually, while the EnvRunner one does it automatically using EnvRunner.

## wave_line_prediction example

The wave_line_prediction example shows how to use a SPH for prediction of a simple waveform. Requires matplotlib.

## lunarlander_env_runner example

The lunar lander example shows a slightly more complicated reinforcement learning environment using EnvRunner. The Gymnasium LunarLander environment features a landing module that the agent must maneuver to the landing pad.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the  [PYAOGMANEO_LICENSE.md](./PYAOGMANEO_LICENSE.md) and [LICENSE.md](./LICENSE.md) file for further information.

Contact Ogma via licenses@ogmacorp.com to discuss commercial use and licensing options.

PyAOgmaNeo Copyright (c) 2020-2025 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
