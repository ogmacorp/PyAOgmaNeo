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
    bool allInRange = true;

    if (std::get<0>(size) < 0) {
        std::cerr << "Error: size[0] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<1>(size) < 0) {
        std::cerr << "Error: size[1] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<2>(size) < 0) {
        std::cerr << "Error: size[2] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (eRadius < 0) {
        std::cerr << "Error: eRadius < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (dRadius < 0) {
        std::cerr << "Error: dRadius < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (historyCapacity < 2) {
        std::cerr << "Error: historyCapacity < 2 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (!allInRange) {
        std::cerr << " - IODesc: Some parameters out of range!" << std::endl;
        abort();
    }

    return true;
}

bool LayerDesc::checkInRange() const {
    bool allInRange = true;

    if (std::get<0>(hiddenSize) < 0) {
        std::cerr << "Error: hiddenSize[0] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<1>(hiddenSize) < 0) {
        std::cerr << "Error: hiddenSize[1] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<2>(hiddenSize) < 0) {
        std::cerr << "Error: hiddenSize[2] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (gHiddenSizeZ < 0) {
        std::cerr << "Error: gHiddenSizeZ < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (eRadius < 0) {
        std::cerr << "Error: eRadius < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (dRadius < 0) {
        std::cerr << "Error: dRadius < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (historyCapacity < 2) {
        std::cerr << "Error: historyCapacity < 2 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (ticksPerUpdate < 1) {
        std::cerr << "Error: ticksPerUpdate < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (temporalHorizon < 1) {
        std::cerr << "Error: temporalHorizon < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (temporalHorizon < ticksPerUpdate) {
        std::cerr << "Error: temporalHorizon < ticksPerUpdate is not allowed!" << std::endl;
        allInRange = false;
    }

    if (!allInRange) {
        std::cerr << " - LayerDesc: Some parameters out of range!" << std::endl;
        abort();
    }

    return true;
}

bool GDesc::checkInRange() const {
    bool allInRange = true;

    if (std::get<0>(size) < 0) {
        std::cerr << "Error: size[0] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<1>(size) < 0) {
        std::cerr << "Error: size[1] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<2>(size) < 0) {
        std::cerr << "Error: size[2] < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (radius < 0) {
        std::cerr << "Error: radius < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (!allInRange) {
        std::cerr << " - GDesc: Some parameters out of range!" << std::endl;
        abort();
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
    const std::vector<GDesc> &gDescs,
    const std::vector<LayerDesc> &layerDescs
) {
    bool allInRange = true;

    aon::Array<aon::Hierarchy::IODesc> cIODescs(ioDescs.size());

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
    
    aon::Array<aon::Hierarchy::GDesc> cGDescs(gDescs.size());

    for (int i = 0; i < gDescs.size(); i++) {
        if(!gDescs[i].checkInRange()) {
            std::cerr << " - at gDesc[" << i << "]" << std::endl;
            allInRange = false;
        }

        cGDescs[i] = aon::Hierarchy::GDesc(
            aon::Int3(std::get<0>(gDescs[i].size), std::get<1>(gDescs[i].size), std::get<2>(gDescs[i].size)),
            gDescs[i].radius
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
            layerDescs[l].gHiddenSizeZ,
            layerDescs[l].eRadius,
            layerDescs[l].dRadius,
            layerDescs[l].historyCapacity,
            layerDescs[l].ticksPerUpdate,
            layerDescs[l].temporalHorizon
        );
    }

    if (!allInRange) {
        std::cerr << " - Hierarchy: Some parameters out of range!" << std::endl;
        abort();
    }

    h.initRandom(cIODescs, cGDescs, cLayerDescs);
    
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
    const std::vector<std::vector<int>> &goalCIs,
    const std::vector<std::vector<int>> &actualCIs,
    bool learnEnabled
) {
    initCheck();

    if (inputCIs.size() != h.getIOSizes().size()) {
        std::cerr << "Incorrect number of inputCIs passed to step! Received " << inputCIs.size() << ", need " << h.getIOSizes().size() << std::endl;
        abort();
    }

    if (goalCIs.size() != h.getGLayer(0).getNumVisibleLayers()) {
        std::cerr << "Incorrect number of goalCIs passed to step! Received " << goalCIs.size() << ", need " << h.getGLayer(0).getNumVisibleLayers() << std::endl;
        abort();
    }

    if (actualCIs.size() != h.getGLayer(0).getNumVisibleLayers()) {
        std::cerr << "Incorrect number of actualCIs passed to step! Received " << actualCIs.size() << ", need " << h.getGLayer(0).getNumVisibleLayers() << std::endl;
        abort();
    }

    aon::Array<aon::IntBuffer> cInputCIsBacking(inputCIs.size());
    aon::Array<const aon::IntBuffer*> cInputCIs(inputCIs.size());

    for (int i = 0; i < inputCIs.size(); i++) {
        int numColumns = h.getIOSizes()[i].x * h.getIOSizes()[i].y;

        if (inputCIs[i].size() != numColumns) {
            std::cerr << "Incorrect input CSDR size at index " << i << " - expected " << numColumns << " columns, got " << inputCIs[i].size() << std::endl;
            abort();
        }

        cInputCIsBacking[i].resize(inputCIs[i].size());

        for (int j = 0; j < inputCIs[i].size(); j++) {
            if (inputCIs[i][j] < 0 || inputCIs[i][j] >= h.getIOSizes()[i].z) {
                std::cerr << "Input CSDR at input index " << i << " has an out-of-bounds column index (" << inputCIs[i][j] << ") at column index " << j << ". It must be in the range [0, " << (h.getIOSizes()[i].z - 1) << "]" << std::endl;
                abort();
            }

            cInputCIsBacking[i][j] = inputCIs[i][j];
        }

        cInputCIs[i] = &cInputCIsBacking[i];
    }

    aon::Array<aon::IntBuffer> cGoalCIsBacking(goalCIs.size());
    aon::Array<const aon::IntBuffer*> cGoalCIs(goalCIs.size());

    for (int i = 0; i < goalCIs.size(); i++) {
        int numColumns = h.getGLayer(0).getVisibleLayerDesc(i).size.x * h.getGLayer(0).getVisibleLayerDesc(i).size.y;

        if (goalCIs[i].size() != numColumns) {
            std::cerr << "Incorrect goal CSDR size at index " << i << " - expected " << numColumns << " columns, got " << goalCIs[i].size() << std::endl;
            abort();
        }

        cGoalCIsBacking[i].resize(goalCIs[i].size());

        for (int j = 0; j < goalCIs[i].size(); j++) {
            if (goalCIs[i][j] < 0 || goalCIs[i][j] >= h.getGLayer(0).getVisibleLayerDesc(i).size.z) {
                std::cerr << "Goal CSDR at input index " << i << " has an out-of-bounds column index (" << goalCIs[i][j] << ") at column index " << j << ". It must be in the range [0, " << (h.getGLayer(0).getVisibleLayerDesc(i).size.z - 1) << "]" << std::endl;
                abort();
            }

            cGoalCIsBacking[i][j] = goalCIs[i][j];
        }

        cGoalCIs[i] = &cGoalCIsBacking[i];
    }
    
    aon::Array<aon::IntBuffer> cActualCIsBacking(actualCIs.size());
    aon::Array<const aon::IntBuffer*> cActualCIs(actualCIs.size());

    for (int i = 0; i < actualCIs.size(); i++) {
        int numColumns = h.getGLayer(0).getVisibleLayerDesc(i).size.x * h.getGLayer(0).getVisibleLayerDesc(i).size.y;

        if (actualCIs[i].size() != numColumns) {
            std::cerr << "Incorrect actual CSDR size at index " << i << " - expected " << numColumns << " columns, got " << actualCIs[i].size() << std::endl;
            abort();
        }

        cActualCIsBacking[i].resize(actualCIs[i].size());

        for (int j = 0; j < actualCIs[i].size(); j++) {
            if (actualCIs[i][j] < 0 || actualCIs[i][j] >= h.getGLayer(0).getVisibleLayerDesc(i).size.z) {
                std::cerr << "Actual CSDR at input index " << i << " has an out-of-bounds column index (" << actualCIs[i][j] << ") at column index " << j << ". It must be in the range [0, " << (h.getGLayer(0).getVisibleLayerDesc(i).size.z - 1) << "]" << std::endl;
                abort();
            }

            cActualCIsBacking[i][j] = actualCIs[i][j];
        }

        cActualCIs[i] = &cActualCIsBacking[i];
    }
    
    h.step(cInputCIs, cGoalCIs, cActualCIs, learnEnabled);
}

std::vector<int> Hierarchy::getPredictionCIs(
    int i
) const {
    initCheck();

    if (i < 0 || i >= h.getIOSizes().size()) {
        std::cout << "Prediction index " << i << " out of range [0, " << (h.getIOSizes().size() - 1) << "]!" << std::endl;
        abort();
    }

    if (!h.dLayerExists(i)) {
        std::cerr << "No decoder exists at index " << i << " - did you set it to the correct type?" << std::endl;
        abort();
    }

    std::vector<int> predictions(h.getPredictionCIs(i).size());

    for (int j = 0; j < predictions.size(); j++)
        predictions[j] = h.getPredictionCIs(i)[j];

    return predictions;
}
