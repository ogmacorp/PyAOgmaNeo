// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyStateAdapter.h"

using namespace pyaon;

void StateAdapter::initCheck() const {
    if (!initialized) {
        std::cerr << "Attempted to use the StateAdapter uninitialized!" << std::endl;
        abort();
    }
}

void StateAdapter::initRandom(
    const std::tuple<int, int, int> &hiddenSize,
    int radius,
    int historyCapacity
) {
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

    if (radius < 0) {
        std::cerr << "Error: radius < 0 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (historyCapacity < 2) {
        std::cerr << "Error: historyCapacity < 2 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (!allInRange) {
        std::cerr << " - StateAdapter: Some parameters out of range!" << std::endl;
        abort();
    }

    adapter.initRandom(aon::Int3(std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize)), radius, historyCapacity);

    initialized = true;
}

void StateAdapter::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != stateAdapterMagic) {
        std::cerr << "Attempted to initialize StateAdapter from incompatible file - " << name << std::endl;
        abort();
    }

    adapter.read(reader);

    initialized = true;
}

void StateAdapter::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != stateAdapterMagic) {
        std::cerr << "Attempted to initialize StateAdapter from incompatible buffer!" << std::endl;
        abort();
    }

    adapter.read(reader);

    initialized = true;
}

void StateAdapter::saveToFile(
    const std::string &name
) {
    initCheck();

    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&stateAdapterMagic, sizeof(int));

    adapter.write(writer);
}

std::vector<unsigned char> StateAdapter::serializeToBuffer() {
    initCheck();

    BufferWriter writer(adapter.size() + sizeof(int));

    writer.write(&stateAdapterMagic, sizeof(int));

    adapter.write(writer);

    return writer.buffer;
}

void StateAdapter::step(
    const std::vector<int> &goalCIs,
    const std::vector<int> &hiddenCIs,
    bool learnEnabled
) {
    initCheck();

    if (hiddenCIs.size() != adapter.getProgCIs().size()) {
        std::cerr << "Error: Incorrect hiddenCIs size passed to StateAdapter!" << std::endl;
        abort();
    }

    aon::IntBuffer cHiddenCIs(hiddenCIs.size());

    for (int j = 0; j < hiddenCIs.size(); j++) {
        if (hiddenCIs[j] < 0 || hiddenCIs[j] >= adapter.getHiddenSize().z) {
            std::cerr << "Error: StateAdapter hiddenCIs has an out-of-bounds column index (" << hiddenCIs[j] << ") at column index " << j << ". It must be in the range [0, " << (adapter.getHiddenSize().z - 1) << "]" << std::endl;
            abort();
        }

        cHiddenCIs[j] = hiddenCIs[j];
    }

    aon::IntBuffer cGoalCIs(hiddenCIs.size());

    for (int j = 0; j < goalCIs.size(); j++) {
        if (goalCIs[j] < 0 || goalCIs[j] >= adapter.getHiddenSize().z) {
            std::cerr << "Error: StateAdapter goalCIs has an out-of-bounds column index (" << goalCIs[j] << ") at column index " << j << ". It must be in the range [0, " << (adapter.getHiddenSize().z - 1) << "]" << std::endl;
            abort();
        }

        cGoalCIs[j] = goalCIs[j];
    }

    adapter.step(&cGoalCIs, &cHiddenCIs, learnEnabled);
}
