Input-Ouput Descriptor (IODesc)
======================================

.. class:: pyaogmaneo.IODesc

Describes a single input/output channel for a hierarchy

.. attribute:: (Int3) IODesc.size

    Size of the input/output

.. attribute:: (int32) IODesc.type

    Type of input/output layer. Can be:

        pyaogmaneo.typeNone: No prediction will be generated (purely an input layer)
        pyaogmaneo.typePrediction: A prediction of the next timestep of values will be generated
        pyaogmaneo.typeAction: A action will be generated (for use with reinforcement learning)

        The prediction of the specified type will be retrieved with Hierarchy.getPredictionCIs (regardless of type).

.. function:: IODesc.__init__(self)

    Initialize to sensible defaults

.. function:: IODesc.__init__(self, size, type)

    Initialize to given values
