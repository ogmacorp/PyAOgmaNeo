// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyRLAdapter.h"

using namespace pyaon;

void RLAdapter::initCheck() const {
    if (!initialized) {
        std::cerr << "Attempted to use the RLAdapter uninitialized!" << std::endl;
        abort();
    }
}

void RLAdapter::initRandom(
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
        std::cerr << " - RLAdapter: Some parameters out of range!" << std::endl;
        abort();
    }

    adapter.initRandom(aon::Int3(std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize)), radius, historyCapacity);

    initialized = true;
}

void RLAdapter::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != rlAdapterMagic) {
        std::cerr << "Attempted to initialize RLAdapter from incompatible file - " << name << std::endl;
        abort();
    }

    adapter.read(reader);

    initialized = true;
}

void RLAdapter::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != rlAdapterMagic) {
        std::cerr << "Attempted to initialize RLAdapter from incompatible buffer!" << std::endl;
        abort();
    }

    adapter.read(reader);

    initialized = true;
}

void RLAdapter::saveToFile(
    const std::string &name
) {
    initCheck();

    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&rlAdapterMagic, sizeof(int));

    adapter.write(writer);
}

std::vector<unsigned char> RLAdapter::serializeToBuffer() {
    initCheck();

    BufferWriter writer(adapter.size() + sizeof(int));

    writer.write(&rlAdapterMagic, sizeof(int));

    adapter.write(writer);

    return writer.buffer;
}

void RLAdapter::step(
    float reward,
    const std::vector<int> &hiddenCIs,
    bool learnEnabled
) {
    initCheck();

    if (hiddenCIs.size() != adapter.getProgCIs().size()) {
        std::cerr << "Error: Incorrect hiddenCIs size passed to RLAdapter!" << std::endl;
        abort();
    }

    aon::IntBuffer cHiddenCIs(hiddenCIs.size());

    for (int j = 0; j < hiddenCIs.size(); j++) {
        if (hiddenCIs[j] < 0 || hiddenCIs[j] >= adapter.getHiddenSize().z) {
            std::cerr << "Error: RLAdapter hiddenCIs has an out-of-bounds column index (" << hiddenCIs[j] << ") at column index " << j << ". It must be in the range [0, " << (adapter.getHiddenSize().z - 1) << "]" << std::endl;
            abort();
        }

        cHiddenCIs[j] = hiddenCIs[j];
    }

    adapter.step(reward, &cHiddenCIs, learnEnabled);
}
