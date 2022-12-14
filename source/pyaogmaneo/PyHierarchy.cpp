// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyaon;

bool IODesc::checkInRange() const {
    if (std::get<0>(size) < 1) {
        std::cerr << "Error: size[0] < 1 is not allowed!" << std::endl;
        return false;
    }

    if (std::get<1>(size) < 1) {
        std::cerr << "Error: size[1] < 1 is not allowed!" << std::endl;
        return false;
    }

    if (std::get<2>(size) < 1) {
        std::cerr << "Error: size[2] < 1 is not allowed!" << std::endl;
        return false;
    }

    if (eRadius < 0) {
        std::cerr << "Error: eRadius < 0 is not allowed!" << std::endl;
        return false;
    }

    if (dRadius < 0) {
        std::cerr << "Error: dRadius < 0 is not allowed!" << std::endl;
        return false;
    }

    if (historyCapacity < 2) {
        std::cerr << "Error: historyCapacity < 2 is not allowed!" << std::endl;
        return false;
    }

    return true;
}

bool LayerDesc::checkInRange() const {
    if (std::get<0>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[0] < 1 is not allowed!" << std::endl;
        return false;
    }

    if (std::get<1>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[1] < 1 is not allowed!" << std::endl;
        return false;
    }

    if (std::get<2>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[2] < 1 is not allowed!" << std::endl;
        return false;
    }

    if (eRadius < 0) {
        std::cerr << "Error: eRadius < 0 is not allowed!" << std::endl;
        return false;
    }

    if (dRadius < 0) {
        std::cerr << "Error: dRadius < 0 is not allowed!" << std::endl;
        return false;
    }

    if (ticksPerUpdate < 1) {
        std::cerr << "Error: ticksPerUpdate < 1 is not allowed!" << std::endl;
        return false;
    }

    if (temporalHorizon < 1) {
        std::cerr << "Error: temporalHorizon < 1 is not allowed!" << std::endl;
        return false;
    }

    if (temporalHorizon < ticksPerUpdate) {
        std::cerr << "Error: temporalHorizon < ticksPerUpdate is not allowed!" << std::endl;
        return false;
    }

    return true;
}

void Hierarchy::initCheck() const {
    if (!initialized) {
        std::cerr << "Attempted to use the hierarchy uninitialized!" << std::endl;
        abort();
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
            std::cerr << " - at ioDesc[" << i << "]" << std::endl;
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
            std::cerr << " - at layerDescs[" << l << "]" << std::endl;
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
        std::cerr << " - Hierarchy: Some parameters out of range!" << std::endl;
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
        std::cerr << "Attempted to initialize Hierarchy from incompatible file - " << name << std::endl;
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
        std::cerr << "Attempted to initialize Hierarchy from incompatible buffer!" << std::endl;
        abort();
    }

    h.read(reader);

    initialized = true;
}

void Hierarchy::saveToFile(
    const std::string &name
) {
    initCheck();

    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&hierarchyMagic, sizeof(int));

    h.write(writer);
}

std::vector<unsigned char> Hierarchy::serializeToBuffer() {
    initCheck();

    BufferWriter writer(h.size() + sizeof(int));

    writer.write(&hierarchyMagic, sizeof(int));

    h.write(writer);

    return writer.buffer;
}

void Hierarchy::setStateFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    initCheck();

    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != hierarchyMagic) {
        std::cerr << "Attempted to set Hierarchy state from incompatible buffer!" << std::endl;
        abort();
    }

    h.readState(reader);
}

std::vector<unsigned char> Hierarchy::serializeStateToBuffer() {
    initCheck();

    BufferWriter writer(h.stateSize() + sizeof(int));

    writer.write(&hierarchyMagic, sizeof(int));

    h.writeState(writer);

    return writer.buffer;
}

void Hierarchy::step(
    const std::vector<std::vector<int>> &inputCIs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    initCheck();

    if (inputCIs.size() != h.getNumIO()) {
        std::cerr << "Incorrect number of inputCIs passed to step! Received " << inputCIs.size() << ", need " << h.getNumIO() << std::endl;
        abort();
    }

    aon::Array<aon::IntBuffer> cInputCIsBacking(inputCIs.size());
    aon::Array<const aon::IntBuffer*> cInputCIs(inputCIs.size());

    for (int i = 0; i < inputCIs.size(); i++) {
        int numColumns = h.getIOSize(i).x * h.getIOSize(i).y;

        if (inputCIs[i].size() != numColumns) {
            std::cerr << "Incorrect CSDR size at index " << i << " - expected " << numColumns << " columns, got " << inputCIs[i].size() << std::endl;
            abort();
        }

        cInputCIsBacking[i].resize(inputCIs[i].size());

        for (int j = 0; j < inputCIs[i].size(); j++) {
            if (inputCIs[i][j] < 0 || inputCIs[i][j] >= h.getIOSize(i).z) {
                std::cerr << "Input CSDR at input index " << i << " has an out-of-bounds column index (" << inputCIs[i][j] << ") at column index " << j << ". It must be in the range [0, " << (h.getIOSize(i).z - 1) << "]" << std::endl;
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
    initCheck();

    if (i < 0 || i >= h.getNumIO()) {
        std::cout << "Prediction index " << i << " out of range [0, " << (h.getNumIO() - 1) << "]!" << std::endl;
        abort();
    }

    if (!h.ioLayerExists(i) || h.getIOType(i) == aon::none) {
        std::cerr << "No decoder exists at index " << i << " - did you set it to the correct type?" << std::endl;
        abort();
    }

    std::vector<int> predictions(h.getPredictionCIs(i).size());

    for (int j = 0; j < predictions.size(); j++)
        predictions[j] = h.getPredictionCIs(i)[j];

    return predictions;
}
