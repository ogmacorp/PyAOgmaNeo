Image Encoder
======================================

.. class:: pyaogmaneo.ImageEncoder

The ImageEncoder is a pre-encoder used to convert images to CSDRs. Sometimes, it can also be used for non-image inputs, but images are the primary intended use.
It is implemented as a bunch of 1D SOMs.

.. function:: ImageEncoder.__init__(self):

    Does nothing.

.. function:: ImageEncoder.initRandom(self, hiddenSize, visibleLayerDescs)

    Initialize an image encoder of given structure.

    :param hiddenSize: (Int3) size of the output (hidden) layer that will be generated.
    :param visibleLayerDescs: ([ImageEncoderVisibleLayerDesc]) list of ImageEncoderVisibleLayerDesc describing each input (visible) layer

.. function:: ImageEncoder.initFromFile(self, name)

    Initialize an image encoder given a save file.

    :param name: (string) save file name

.. function:: ImageEncoder.initFromBuffer(self, buffer)

    Initialize a hierarchy given a byte buffer.

    :param buffer: ([uint8]) byte buffer to read from

.. function:: ImageEncoder.saveToFile(self, name)

    Save the image encoder to a file.

    :param name: (string) save file name

.. function:: ImageEncoder.serializeToBuffer(self)

    Serialize the hierarchy (write to byte list)

    :rtype: ([uint8]) byte buffer

.. function:: ImageEncoder.step(self, inputs, learnEnabled=True)

    Perform a simulation step of the ImageEncoder. This will both generate a CSDR from the images (visibleActivations) and learn to improve the representation (learning only if learnEnabled=True).

    :param inputs: ([ByteBuffer]) list of input byte buffers representing the image of the dimensions described in the initialization. If using regular RGB images, the values in this buffer should be in the range [0, 255])
    :param learnEnabled: (boolean) whether or not to enable learning (if False, will only perform inference). Defaults to True

.. function:: ImageEncoder.reconstruct(self, reconCIs)

    Reconstruct (reverse the encoding of) a given CSDR (reconCIs). This action perform the oppositive of regular inference - get the inputs given the output representation.
    After this function is called, the reconstructions for all inputs will be updated/generated. This can then be retrieved with ImageEncoder.getReconstruction (described below).

    :param reconCIs: (IntBuffer) CSDR to reconstruct

.. function:: ImageEncoder.getReconstruction(self, index)

    Return the reconstructed inputs (generated by ImageEncoder.reconstruct)

    :param index: (int32) index of the visible layer to retrieve the reconstruction from
    :rtype: (ByteBuffer) reconstruction of the input

.. function:: ImageEncoder.getNumVisibleLayers(self)

    Return the number of visible (input) layers the image encoder has. Will be equal to len(visibleLayerDescs) provided during initialization

    :rtype: (int32) number of layers

.. function:: ImageEncoder.getVisibleLayerDesc(self, index)

    Retrieve the ImageEncoderVisibleLayerDesc used to initialize the image encoder at a certain index

    :param index: (int32) index of the input (visible) layer
    :rtype: (ImageEncoderVisibleLayerDesc) the descriptor

.. function:: ImageEncoder.getHiddenCIs(self)

    Get the hidden encoded state (output CSDR)

    :rtype: (IntBuffer) the CSDR

.. function:: ImageEncoder.getHiddenSize(self)

    Get the size of the hidden state

    :rtype: (Int3) the CSDR size

.. function:: ImageEncoder.setLR(self, lr)

    Set the learning rate

    :param lr: (float32) value to set

.. function:: ImageEncoder.getLR(self)

    Get the learning rate

    :rtype: (float32) lr
