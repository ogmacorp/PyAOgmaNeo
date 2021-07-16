Input-Ouput Descriptor (IODesc)
======================================

.. class:: pyaogmaneo.IODesc

Describes a single input/output channel for a hierarchy

.. attribute:: ((int32, int32, int32)) IODesc.size

    Size of the input/output

.. attribute:: (int32) IODesc.type

    Type of input/output layer. Can be:

        pyaogmaneo.prediction: A prediction of the next timestep of values will be generated

        pyaogmaneo.action: An action will be generated (for use with reinforcement learning)

        The prediction of the specified type will be retrieved with Hierarchy.getPredictionCIs (regardless of type).

.. attribute:: (int32) IODesc.eRadius

    Feed-forward (bottom-up) encoder radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

    Note: This value overrides that of the same name in pyaogmaneo.Hierarchy

.. attribute:: (int32) IODesc.dRadius

    Feed-back (top-down) decoder radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

    Note: This value overrides that of the same name in pyaogmaneo.Hierarchy

.. attribute:: (int32) IODesc.historyCapacity

    Maximum number of history samples (credit assignment horizon) an actor can have

.. function:: IODesc.__init__(self, size=(4, 4, 16), type=pyaogmaneo.prediction, eRadius=2, dRadius=2, aRadius=2, historyCapacity=64)

    Initialize to given values
