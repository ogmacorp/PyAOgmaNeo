Layer Descriptor
======================================

.. class:: pyaogmaneo.LayerDesc

Describes the layers of the hierarchy during initialization. Fill these variables with the desired structure and properties of the hierarchy

.. attribute:: (Int3) LayerDesc.hiddenSize

    Hidden layer sparse coder (AKA encoder) size

.. attribute:: (int32) LayerDesc.ffRadius

    Feed-forward (bottom-up) sparse coder (AKA encoder) radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

.. attribute:: (int32) LayerDesc.pRadius

    Feed-back (top-down) predictor radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

.. attribute:: (int32) LayerDesc.aRadius

    Feed-back (top-down) actor radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

.. attribute:: (int32) LayerDesc.ticksPerUpdate

    Temporal horizon of sparse coder (AKA encoder). Must be greater than or equal to 1

.. attribute:: (int32) LayerDesc.temporalHorizon

    Temporal horizon of sparse coder (AKA encoder). Must be greater than or equal to LayerDesc.ticksPerUpdate. If no temporal window overshoot is desired, set equal to LayerDesc.ticksPerUpdate, otherwise the overshoot is how much large it is than LayerDesc.ticksPerUpdate

.. function:: LayerDesc.__init__(self)

    Initialize to sensible defaults
