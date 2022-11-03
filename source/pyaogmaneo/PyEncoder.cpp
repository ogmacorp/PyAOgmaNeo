// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyEncoder.h"

using namespace pyaon;

bool EncoderVisibleLayerDesc::checkInRange() const {
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

    if (radius < 0) {
        std::cerr << "Error: radius < 0 is not allowed!" << std::endl;
        return false;
    }

    return true;
}

void Encoder::initCheck() const {
    if (!initialized) {
        std::cerr << "Attempted to use the Encoder uninitialized!" << std::endl;
        abort();
    }
}

void Encoder::initRandom(
    const std::tuple<int, int, int> &hiddenSize,
    const std::vector<EncoderVisibleLayerDesc> &visibleLayerDescs
) {
    bool allInRange = true;

    aon::Array<aon::Encoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        if (!visibleLayerDescs[v].checkInRange()) {
            std::cerr << " - at visibleLayerDescs[" << v << "]" << std::endl;
            allInRange = false;
        }

        cVisibleLayerDescs[v].size = aon::Int3(std::get<0>(visibleLayerDescs[v].size), std::get<1>(visibleLayerDescs[v].size), std::get<2>(visibleLayerDescs[v].size));
        cVisibleLayerDescs[v].radius = visibleLayerDescs[v].radius;
    }

    if (std::get<0>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[0] < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<1>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[1] < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (std::get<2>(hiddenSize) < 1) {
        std::cerr << "Error: hiddenSize[2] < 1 is not allowed!" << std::endl;
        allInRange = false;
    }

    if (!allInRange) {
        std::cerr << " - Encoder: Some parameters out of range!" << std::endl;
        abort();
    }

    enc.initRandom(aon::Int3(std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize)), cVisibleLayerDescs);

    initialized = true;
}

void Encoder::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != encoderMagic) {
        std::cerr << "Attempted to initialize Encoder from incompatible file - " << name << std::endl;
        abort();
    }

    enc.read(reader);

    initialized = true;
}

void Encoder::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != encoderMagic) {
        std::cerr << "Attempted to initialize Encoder from incompatible buffer!" << std::endl;
        abort();
    }

    enc.read(reader);

    initialized = true;
}

void Encoder::saveToFile(
    const std::string &name
) {
    initCheck();

    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&encoderMagic, sizeof(int));

    enc.write(writer);
}

std::vector<unsigned char> Encoder::serializeToBuffer() {
    initCheck();

    BufferWriter writer(enc.size() + sizeof(int));

    writer.write(&encoderMagic, sizeof(int));

    enc.write(writer);

    return writer.buffer;
}

void Encoder::step(
    const std::vector<std::vector<int>> &inputCIs,
    bool learnEnabled
) {
    initCheck();

    if (inputCIs.size() != enc.getNumVisibleLayers()) {
        std::cerr << "Incorrect number of inputs given to Encoder! Expected " << enc.getNumVisibleLayers() << ", got " << inputCIs.size() << std::endl;
        abort();
    }

    aon::Array<aon::IntBuffer> cInputCIsBacking(inputCIs.size());
    aon::Array<const aon::IntBuffer*> cInputCIs(inputCIs.size());

    for (int i = 0; i < inputCIs.size(); i++) {
        int numInputCIs = enc.getVisibleLayerDesc(i).size.x * enc.getVisibleLayerDesc(i).size.y;

        if (inputCIs[i].size() != numInputCIs) {
            std::cerr << "Incorrect number of inputCIs given to Encoder! At input " << i << ": Expected " << numInputCIs << ", got " << inputCIs[i].size() << std::endl;
            abort();
        }

        cInputCIsBacking[i].resize(inputCIs[i].size());
        
        for (int j = 0; j < inputCIs[i].size(); j++) {
            if (inputCIs[i][j] < 0 || inputCIs[i][j] >= enc.getVisibleLayerDesc(i).size.z) {
                std::cerr << "Input CSDR at input index " << i << " has an out-of-bounds column index (" << inputCIs[i][j] << ") at column index " << j << ". It must be in the range [0, " << (enc.getVisibleLayerDesc(i).size.z - 1) << "]" << std::endl;
                abort();
            }

            cInputCIsBacking[i][j] = inputCIs[i][j];
        }

        cInputCIs[i] = &cInputCIsBacking[i];
    }

    enc.step(cInputCIs, learnEnabled);
}
