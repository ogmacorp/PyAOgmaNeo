Hierarchy
======================================

.. class:: pyaogmaneo.Hierarchy

The sparse predictive hierarchy (SPH). Can be thought of as the "agent" when used for reinforcement learning. This is the main component of any PyAOgmaNeo project

.. function:: Hierarchy.__init__(self):

    Does nothing.

.. function:: Hierarchy.initRandom(self, ioDescs, layerDescs)

    Initialize a hierarchy (random weights) of given structure.

    :param ioDescs: ([IODesc]) list of IODesc's (input-output (IO) descriptors, see IODesc). Defines the size of each IO layer and its type
    :param layerDescs: ([LayerDesc]) A list of filled-out descriptors (LayerDesc objects) describing how all the layers in the hierarchy should look

.. function:: Hierarchy.initFromFile(self, name)

    Initialize a hierarchy given a save file.

    :param name: (string) save file name

.. function:: Hierarchy.initFromBuffer(self, buffer)

    Initialize a hierarchy given a byte buffer.

    :param buffer: ([uint8]) byte buffer to read from

.. function:: Hierarchy.saveToFile(self, name)

    Save the hierarchy to a file.

    :param name: (string) save file name

.. function:: Hierarchy.serializeToBuffer(self)

    Serialize the hierarchy (write to byte list)

    :rtype: ([uint8]) byte buffer

.. function:: Hierarchy.setStateFromBuffer(self, buffer)

    Sets just the state (short term memory) of the hierarchy from a buffer.

    :param buffer: ([uint8]) byte buffer to read from

.. function:: Hierarchy.serializeStateToBuffer(self)

    Serialize just the state of the hierarchy (short term memory) (write to byte list)

    :rtype: ([uint8]) byte buffer

.. function:: Hierarchy.clearState(self)

    Clear the hidden state of the hierarchy (short term memory) by setting it to 0

.. function:: Hierarchy.step(self, inputCIs, learnEnabled=True, reward=0.0, mimic=False)

    Perform a simulation step of the hierarchy. This will produce new predictions or actions if those are being used (as specified in the IODesc's)

    :param inputCIs: ([IntBuffer]) list of input integer buffers representing the CSDRs of the dimensions described in the initialization
    :param learnEnabled: (boolean) whether or not to enable learning (if False, will only perform inference). Defaults to True
    :param reward: (float32) reward signal, if action IO layers (pyaogmaneo.IODesc type set to typeAction) are present this will be used to update those to maximize reward. Defaults to 0.0
    :param mimic: (boolean) If true, sets the actors (action generators for reinforcement learning) to behave like regular decoders (prediction). This is useful for imitation learning followed by reinforcement learning

.. function:: Hierarchy.getNumLayers(self)

    Return the number of layers (encoders with paired decoders) the hierarchy has. Will be equal to (len(layerDescs))

    :rtype: (int32) number of layers

.. function:: Hierarchy.getPredictionCIs(self, i)

    Get a prediction for a given IO layer index. If the IODesc for this index is of type action, it will return the action instead

    :param i: (int32) index of the IO layer
    :rtype: (IntBuffer) integer buffer containing predictions

.. function:: Hierarchy.getUpdate(self, l)

    Return whether a given layer has updated (clocked) in the last timestep

    :param l: (int32) index of the layer
    :rtype: (boolean) True if updated, False otherwise

.. function:: Hierarchy.getHiddenCIs(self, l)

    Get the CSDR of the encoder of a certain layer

    :param l: (int32) index of the layer
    :rtype: (IntBuffer) integer buffer containing encoder hidden layer states (CSDR)

.. function:: Hierarchy.getHiddenSize(self, l)

    Get the hidden size of the encoder of a certain layer

    :param l: (int32) index of the layer
    :rtype: ((int32, int32, int32)) size of layer. Will always be equal to the size if was initialzed to in the layer descriptor (LayerDesc.hiddenSize)

.. function:: Hierarchy.getTicks(self, l)

    Get current tick of a layer (clock value)

    :param l: (int32) index of the layer
    :rtype: (int32) tick value. Will always be less the value produced by Hierarchy.getTicksPerUpdate(l). Will always be 0 for the first layer (l = 0)

.. function:: Hierarchy.getTicksPerUpdate(self, l)

    Number of ticks required before a layer clocks. Typically 2

    :param l: (int32) index of the layer
    :rtype: (int32) number of ticks. Will always be 1 for the first layer (l = 0)

.. function:: Hierarchy.getNumIO(self)

    Get the number of IO layers to the hierarchy (number of IODescs).

    :rtype: (int32) number of IO layers

.. function:: Hierarchy.getIOSize(self, i)

    Get the size of an IO layer

    :param i: (int32) index of the IO layer
    :rtype: ((int32, int32, int32)) size of the IO layer

.. function:: Hierarchy.getIOType(self, i)

    Get the type of an IO layer

    :param i: (int32) index of the IO layer
    :rtype: (IOType) type of IO layer

.. function:: Hierarchy.getNumDLayers(self, l)

    Get the number of decoder (top down) sub-layers at a given layer

    :param l: (int32) index of the layer. Must be greater than 0 as the first layer does not have regular decoders
    :rtype: (int32) number of decoder layers

.. function:: Hierarchy.getNumEVisibleLayers(self, l)

    Get the number of visible (sub) layers of a encoder. This will be equal to Hierarchy.getTicksPerUpdate(l) for all l except for 0, where it will be equal to Hierarchy.getNumIO()

    :param l: (int32) index of the layer
    :rtype: (int32) number of visible layers

.. function:: Hierarchy.setELR(self, l, lr)

    Set the learning rate of an encoder (E). This is an integer in this case

    :param l: (int32) index of the layer
    :param lr: (int32) value to set

.. function:: Hierarchy.getELR(self, l)

    Get the learning rate of an encoder (E). This is an integer in this case

    :param l: (int32) index of the layer
    :rtype: (int32) lr

.. function:: Hierarchy.setDLR(self, l, lr)

    Set the learning rate of a decoder (D)

    :param l: (int32) index of the layer
    :param lr: (float32) value to set

.. function:: Hierarchy.getDLR(self, l)

    Get the learning rate of a decoder (D)

    :param l: (int32) index of the layer
    :rtype: (float32) lr

.. function:: Hierarchy.setAVLR(self, i, vlr)

    Set the value (critic) learning rate of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :param vlr: (float32) value to set

.. function:: Hierarchy.getAVLR(self, i)

    Get the value (critic) learning rate of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :rtype: (float32) vlr

.. function:: Hierarchy.setAALR(self, i, alr)

    Set the action (actor) learning rate of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :param alr: (float32) value to set

.. function:: Hierarchy.getAALR(self, i)

    Get the action (actor) learning rate of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :rtype: (float32) alr

.. function:: Hierarchy.setADiscount(self, i, discount)

    Set the discount factor of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :param discount: (float32) value to set

.. function:: Hierarchy.getADiscount(self, i)

    Get the discount factor of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :rtype: (float32) discount
    
.. function:: Hierarchy.setAMinSteps(self, i, minSteps)

    Set the minSteps (minimum number of samples before actor can update) of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :param minSteps: (int32) value to set

.. function:: Hierarchy.getAMinSteps(self, i)

    Get the minSteps (minimum number of samples before actor can update) of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :rtype: (int32) minSteps

.. function:: Hierarchy.setAHistoryIters(self, i, historyIters)

    Set the historyIters (number of iterations of credit assignment) of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :param historyIters: (int32) value to set

.. function:: Hierarchy.getAHistoryIters(self, i)

    Get the historyIters (number of iterations of credit assignment) of an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :rtype: (int32) historyIters

.. function:: Hierarchy.setATemperature(self, i, temperature)

    Set the Boltzmann exploration temperature (0.0 for no exploration) on an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :param explore: (boolean) value to set

.. function:: Hierarchy.getATemperature(self, i)

    Get the Boltzmann exploration temperature (0.0 for no exploration) on an action layer (A) at the bottom of the hierarchy (IO layer)

    :param i: (int32) index of the IO layer
    :rtype: (boolean) temperature

.. function:: Hierarchy.getERadius(self, l)

    Get the encoder (E) radius of a layer

    :param l: (int32) index of the layer
    :rtype: (int32) encoder radius

.. function:: Hierarchy.getDRadius(self, l, i)

    Get the decoder (D) radius of a layer

    :param l: (int32) index of the layer
    :param i: (int32) index of the IO layer 
    :rtype: (int32) P radius

.. function:: Hierarchy.getARadius(self, i)

    Get the actor (A) radius onto the first hidden layer

    :param i: (int32) index of the IO layer 
    :rtype: (int32) history capacity

.. function:: Hierarchy.getAHistoryCapacity(self, i)

    Get the actor (A) history capacity

    :param i: (int32) index of the IO layer 
    :rtype: (int32) history capacity

