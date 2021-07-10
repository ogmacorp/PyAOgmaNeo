// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyRLAdapter.h"

using namespace pyaon;

void RLAdapter::initRandom(
    const std::tuple<int, int, int> &hiddenSize,
    int radius
) {
    adapter.initRandom(aon::Int3(std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize)), radius);
}

void RLAdapter::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    adapter.read(reader);
}

void RLAdapter::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    adapter.read(reader);
}

void RLAdapter::saveToFile(
    const std::string &name
) {
    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    adapter.write(writer);
}

std::vector<unsigned char> RLAdapter::serializeToBuffer() {
    BufferWriter writer(adapter.size());

    adapter.write(writer);

    return writer.buffer;
}

void RLAdapter::step(
    const std::vector<int> &hiddenCIs,
    float reward,
    bool learnEnabled
) {
    aon::IntBuffer cHiddenCIs(hiddenCIs.size());

    for (int j = 0; j < hiddenCIs.size(); j++)
        cHiddenCIs[j] = hiddenCIs[j];

    adapter.step(&cHiddenCIs, reward, learnEnabled);
}
