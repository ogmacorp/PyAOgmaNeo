Input-Ouput Descriptor (IODesc)
======================================

.. class:: pyaogmaneo.IODesc

Describes a single input/output channel for a hierarchy

.. attribute:: ((int32, int32, int32)) IODesc.size

    Size of the input/output

.. attribute:: (int32) IODesc.type

    Type of input/output layer. Can be:

        pyaogmaneo.none: No prediction will be generated (purely an input layer)

        pyaogmaneo.prediction: A prediction of the next timestep of values will be generated

        pyaogmaneo.action: An action will be generated (for use with reinforcement learning)

        The prediction of the specified type will be retrieved with Hierarchy.getPredictionCIs (regardless of type).

.. attribute:: (int32) IODesc.ffRadius

    Feed-forward (bottom-up) sparse coder (AKA encoder) radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

    Note: This value overrides that of the same name in pyaogmaneo.Hierarchy

.. attribute:: (int32) IODesc.pRadius

    Feed-back (top-down) predictor radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

    Note: This value overrides that of the same name in pyaogmaneo.Hierarchy

.. attribute:: (int32) IODesc.aRadius

    Feed-back (top-down) actor radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)
    
.. attribute:: (int32) IODesc.historyCapacity

    Maximum number of history samples (credit assignment horizon) an actor can have

.. function:: IODesc.__init__(self, size=(4, 4, 16), type=pyaogmaneo.none, ffRadius=2, pRadius=2, aRadius=2, historyCapacity=32)

    Initialize to given values
