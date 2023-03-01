Examples
======================================

The `examples/ <https://github.com/ogmacorp/PyAOgmaNeo/tree/master/examples>`_ directory contains a few usage examples.

EnvRunner
********************************************

The EnvRunner is a simple way to automatically create PyAOgmaNeo systems for Gymnasium tasks. It will automatically create the hierarchy and appropriate pre-encoders. This is good enough for initial experimentation, but further control requires manual usage.

CartPole EnvRunner example
********************************************

The CartPole example is a simple test on the Gymnasium CartPole environment. It uses EnvRunner.

CartPole Manual example
********************************************

The CartPole example is a simple test on the Gymnasium CartPole environment. It does not use EnvRunner, and creates the hierarchy manually.

WavyLinePrediction example
********************************************

The WavyLinePrediction example shows how to use a SPH for prediction of a simple waveform. Requires matplotlib.
