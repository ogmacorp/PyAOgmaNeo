// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyaon;

bool IODesc::checkInRange() const {
    if (std::get<0>(size) < 1) {
        throw std::runtime_error("Error: size[0] < 1 is not allowed!");
        return false;
    }

    if (std::get<1>(size) < 1) {
        throw std::runtime_error("Error: size[1] < 1 is not allowed!");
        return false;
    }

    if (std::get<2>(size) < 1) {
        throw std::runtime_error("Error: size[2] < 1 is not allowed!");
        return false;
    }

    if (eRadius < 0) {
        throw std::runtime_error("Error: eRadius < 0 is not allowed!");
        return false;
    }

    if (dRadius < 0) {
        throw std::runtime_error("Error: dRadius < 0 is not allowed!");
        return false;
    }

    if (historyCapacity < 2) {
        throw std::runtime_error("Error: historyCapacity < 2 is not allowed!");
        return false;
    }

    return true;
}

bool LayerDesc::checkInRange() const {
    if (std::get<0>(hiddenSize) < 1) {
        throw std::runtime_error("Error: hiddenSize[0] < 1 is not allowed!");
        return false;
    }

    if (std::get<1>(hiddenSize) < 1) {
        throw std::runtime_error("Error: hiddenSize[1] < 1 is not allowed!");
        return false;
    }

    if (std::get<2>(hiddenSize) < 1) {
        throw std::runtime_error("Error: hiddenSize[2] < 1 is not allowed!");
        return false;
    }

    if (eRadius < 0) {
        throw std::runtime_error("Error: eRadius < 0 is not allowed!");
        return false;
    }

    if (dRadius < 0) {
        throw std::runtime_error("Error: dRadius < 0 is not allowed!");
        return false;
    }

    if (ticksPerUpdate < 1) {
        throw std::runtime_error("Error: ticksPerUpdate < 1 is not allowed!");
        return false;
    }

    if (temporalHorizon < 1) {
        throw std::runtime_error("Error: temporalHorizon < 1 is not allowed!");
        return false;
    }

    if (temporalHorizon < ticksPerUpdate) {
        throw std::runtime_error("Error: temporalHorizon < ticksPerUpdate is not allowed!");
        return false;
    }

    return true;
}

void Hierarchy::encGetSetIndexCheck(
    int l
) const {
    if (l < 0 || l >= h.getNumLayers()) {
        throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");
        abort();
    }
}

void Hierarchy::decGetSetIndexCheck(
    int l, int i
) const {
    if (l < 0 || l >= h.getNumLayers()) {
        throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");
        abort();
    }

    if (l == 0 && (i < 0 || i >= h.getNumIO())) {
        throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");
        abort();
    }

    if (l > 0 && i == 0) {
        throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid decoder index!");
        abort();
    }

    if (l == 0 && (!h.ioLayerExists(i) || h.getIOType(i) != aon::prediction)) {
        throw std::runtime_error("Error: index " + std::to_string(i) + " does not have a decoder!");
        abort();
    }
}

void Hierarchy::actGetSetIndexCheck(
    int i
) const {
    if (i < 0 || i >= h.getNumIO()) {
        throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");
        abort();
    }

    if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
        throw std::runtime_error("Error: index " + std::to_string(i) + " does not have an actor!");
        abort();
    }
}

Hierarchy::Hierarchy(
    const std::vector<IODesc> &ioDescs,
    const std::vector<LayerDesc> &layerDescs,
    const std::string &name,
    const std::vector<unsigned char> &buffer
) {
    if (!buffer.empty())
        initFromBuffer(buffer);
    else if (!name.empty())
        initFromFile(name);
    else {
        if (ioDescs.empty() || layerDescs.empty())
            throw std::runtime_error("Error: Hierarchy constructor requires some non-empty arguments!");

        initRandom(ioDescs, layerDescs);
    }
}

void Hierarchy::initRandom(
    const std::vector<IODesc> &ioDescs,
    const std::vector<LayerDesc> &layerDescs
) {
    aon::Array<aon::Hierarchy::IODesc> cIODescs(ioDescs.size());

    bool allInRange = true;

    for (int i = 0; i < ioDescs.size(); i++) {
        if(!ioDescs[i].checkInRange()) {
            throw std::runtime_error(" - at ioDesc[" + std::to_string(i) + "]");
            allInRange = false;
        }

        cIODescs[i] = aon::Hierarchy::IODesc(
            aon::Int3(std::get<0>(ioDescs[i].size), std::get<1>(ioDescs[i].size), std::get<2>(ioDescs[i].size)),
            static_cast<aon::IOType>(ioDescs[i].type),
            ioDescs[i].eRadius,
            ioDescs[i].dRadius,
            ioDescs[i].historyCapacity
        );
    }
    
    aon::Array<aon::Hierarchy::LayerDesc> cLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        if(!layerDescs[l].checkInRange()) {
            throw std::runtime_error(" - at layerDescs[" + std::to_string(l) + "]");
            allInRange = false;
        }

        cLayerDescs[l] = aon::Hierarchy::LayerDesc(
            aon::Int3(std::get<0>(layerDescs[l].hiddenSize), std::get<1>(layerDescs[l].hiddenSize), std::get<2>(layerDescs[l].hiddenSize)),
            layerDescs[l].eRadius,
            layerDescs[l].dRadius,
            layerDescs[l].ticksPerUpdate,
            layerDescs[l].temporalHorizon
        );
    }

    if (!allInRange) {
        throw std::runtime_error(" - Hierarchy: Some parameters out of range!");
        abort();
    }

    h.initRandom(cIODescs, cLayerDescs);
    
    initialized = true;
}

void Hierarchy::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != hierarchyMagic) {
        throw std::runtime_error("Attempted to initialize Hierarchy from incompatible file - " + name);
        abort();
    }

    h.read(reader);

    initialized = true;
}

void Hierarchy::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != hierarchyMagic) {
        throw std::runtime_error("Attempted to initialize Hierarchy from incompatible buffer!");
        abort();
    }

    h.read(reader);

    initialized = true;
}

void Hierarchy::saveToFile(
    const std::string &name
) {
    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&hierarchyMagic, sizeof(int));

    h.write(writer);
}

std::vector<unsigned char> Hierarchy::serializeToBuffer() {
    BufferWriter writer(h.size() + sizeof(int));

    writer.write(&hierarchyMagic, sizeof(int));

    h.write(writer);

    return writer.buffer;
}

void Hierarchy::setStateFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != hierarchyMagic) {
        throw std::runtime_error("Attempted to set Hierarchy state from incompatible buffer!");
        abort();
    }

    h.readState(reader);
}

std::vector<unsigned char> Hierarchy::serializeStateToBuffer() {
    BufferWriter writer(h.stateSize() + sizeof(int));

    writer.write(&hierarchyMagic, sizeof(int));

    h.writeState(writer);

    return writer.buffer;
}

void Hierarchy::step(
    const std::vector<std::vector<int>> &inputCIs,
    bool learnEnabled,
    float reward,
    float mimic
) {
    if (inputCIs.size() != h.getNumIO()) {
        throw std::runtime_error("Incorrect number of inputCIs passed to step! Received " + std::to_string(inputCIs.size()) + ", need " + std::to_string(h.getNumIO()));
        abort();
    }

    aon::Array<aon::IntBuffer> cInputCIsBacking(inputCIs.size());
    aon::Array<const aon::IntBuffer*> cInputCIs(inputCIs.size());

    for (int i = 0; i < inputCIs.size(); i++) {
        int numColumns = h.getIOSize(i).x * h.getIOSize(i).y;

        if (inputCIs[i].size() != numColumns) {
            throw std::runtime_error("Incorrect CSDR size at index " + std::to_string(i) + " - expected " + std::to_string(numColumns) + " columns, got " + std::to_string(inputCIs[i].size()));
            abort();
        }

        cInputCIsBacking[i].resize(inputCIs[i].size());

        for (int j = 0; j < inputCIs[i].size(); j++) {
            if (inputCIs[i][j] < 0 || inputCIs[i][j] >= h.getIOSize(i).z) {
                throw std::runtime_error("Input CSDR at input index " + std::to_string(i) + " has an out-of-bounds column index (" + std::to_string(inputCIs[i][j]) + ") at column index " + std::to_string(j) + ". It must be in the range [0, " + std::to_string(h.getIOSize(i).z - 1) + "]");
                abort();
            }

            cInputCIsBacking[i][j] = inputCIs[i][j];
        }

        cInputCIs[i] = &cInputCIsBacking[i];
    }
    
    h.step(cInputCIs, learnEnabled, reward, mimic);
}

std::vector<int> Hierarchy::getPredictionCIs(
    int i
) const {
    if (i < 0 || i >= h.getNumIO()) {
        throw std::runtime_error("Prediction index " + std::to_string(i) + " out of range [0, " + std::to_string(h.getNumIO() - 1) + "]!");
        abort();
    }

    if (!h.ioLayerExists(i) || h.getIOType(i) == aon::none) {
        throw std::runtime_error("No decoder exists at index " + std::to_string(i) + " - did you set it to the correct type?");
        abort();
    }

    std::vector<int> predictions(h.getPredictionCIs(i).size());

    for (int j = 0; j < predictions.size(); j++)
        predictions[j] = h.getPredictionCIs(i)[j];

    return predictions;
}
