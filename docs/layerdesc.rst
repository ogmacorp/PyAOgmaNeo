Layer Descriptor
======================================

.. class:: pyaogmaneo.LayerDesc

Describes the layers of the hierarchy during initialization. Fill these variables with the desired structure and properties of the hierarchy

.. attribute:: ((int32, int32, int32)) LayerDesc.hiddenSize

    Hidden layer sparse coder (AKA encoder) size

.. attribute:: (int32) LayerDesc.eRadius

    Feed-forward (bottom-up) encoder radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

.. attribute:: (int32) LayerDesc.dRadius

    Feed-back (top-down) decoder radius. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

.. attribute:: (int32) LayerDesc.ticksPerUpdate

    Temporal horizon of sparse coder (AKA encoder). Must be greater than or equal to 1

.. attribute:: (int32) LayerDesc.temporalHorizon

    Temporal horizon of sparse coder (AKA encoder). Must be greater than or equal to LayerDesc.ticksPerUpdate. If no temporal window overshoot is desired, set equal to LayerDesc.ticksPerUpdate, otherwise the overshoot is how much large it is than LayerDesc.ticksPerUpdate

.. function:: LayerDesc.__init__(self, hiddenSize=(4, 4, 16), eRadius=2, dRadius=2, ticksPerUpdate=2, temporalHorizon=2)

    Initialize to given values
