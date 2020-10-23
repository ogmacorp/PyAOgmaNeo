Image Encoder Visible Layer Desciptor
======================================

.. class:: pyaogmaneo.ImageEncoderVisibleLayerDesc

Describes a single visible layer of an ImageEncoder for use during initialization

.. attribute:: ((int32, int32, int32)) ImageEncoderVisibleLayerDesc.size

    Size of the input. Last value (z) will be 3 for RGB images

.. attribute:: (int32) ImageEncoderVisibleLayerDesc.radius

    Radius onto input. Must be 0 or greater. The diameter of the receptive field will be (2 * radius + 1)

.. function:: ImageEncoderVisibleLayerDesc.__init__(self)

    Initialize to sensible defaults

.. function:: ImageEncoderVisibleLayerDesc.__init__(self, size, radius)

    Initialize to given values
