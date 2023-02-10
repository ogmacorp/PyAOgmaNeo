// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyEncoder.h"

using namespace pyaon;

void EncoderVisibleLayerDesc::checkInRange() const {
    if (std::get<0>(size) < 1)
        throw std::runtime_error("Error: size[0] < 1 is not allowed!");

    if (std::get<1>(size) < 1)
        throw std::runtime_error("Error: size[1] < 1 is not allowed!");

    if (std::get<2>(size) < 1)
        throw std::runtime_error("Error: size[2] < 1 is not allowed!");

    if (radius < 0)
        throw std::runtime_error("Error: radius < 0 is not allowed!");
}

Encoder::Encoder(
    const std::tuple<int, int, int> &hiddenSize,
    const std::vector<EncoderVisibleLayerDesc> &visibleLayerDescs,
    const std::string &name,
    const std::vector<unsigned char> &buffer
) {
    if (!buffer.empty())
        initFromBuffer(buffer);
    else if (!name.empty())
        initFromFile(name);
    else {
        if (visibleLayerDescs.empty())
            throw std::runtime_error("Error: Encoder constructor requires some non-empty arguments!");

        initRandom(hiddenSize, visibleLayerDescs);
    }
}

void Encoder::initRandom(
    const std::tuple<int, int, int> &hiddenSize,
    const std::vector<EncoderVisibleLayerDesc> &visibleLayerDescs
) {
    bool allInRange = true;

    aon::Array<aon::Encoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        visibleLayerDescs[v].checkInRange();

        cVisibleLayerDescs[v].size = aon::Int3(std::get<0>(visibleLayerDescs[v].size), std::get<1>(visibleLayerDescs[v].size), std::get<2>(visibleLayerDescs[v].size));
        cVisibleLayerDescs[v].radius = visibleLayerDescs[v].radius;
    }

    if (std::get<0>(hiddenSize) < 1)
        throw std::runtime_error("Error: hiddenSize[0] < 1 is not allowed!");

    if (std::get<1>(hiddenSize) < 1)
        throw std::runtime_error("Error: hiddenSize[1] < 1 is not allowed!");

    if (std::get<2>(hiddenSize) < 1)
        throw std::runtime_error("Error: hiddenSize[2] < 1 is not allowed!");

    if (!allInRange)
        throw std::runtime_error(" - Encoder: Some parameters out of range!");

    enc.initRandom(aon::Int3(std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize)), cVisibleLayerDescs);
}

void Encoder::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != encoderMagic)
        throw std::runtime_error("Attempted to initialize Encoder from incompatible file - " + name);

    enc.read(reader);
}

void Encoder::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != encoderMagic)
        throw std::runtime_error("Attempted to initialize Encoder from incompatible buffer!");

    enc.read(reader);
}

void Encoder::saveToFile(
    const std::string &name
) {
    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&encoderMagic, sizeof(int));

    enc.write(writer);
}

std::vector<unsigned char> Encoder::serializeToBuffer() {
    BufferWriter writer(enc.size() + sizeof(int));

    writer.write(&encoderMagic, sizeof(int));

    enc.write(writer);

    return writer.buffer;
}

void Encoder::step(
    const std::vector<std::vector<int>> &inputCIs,
    bool learnEnabled
) {
    if (inputCIs.size() != enc.getNumVisibleLayers())
        throw std::runtime_error("Incorrect number of inputs given to Encoder! Expected " + std::to_string(enc.getNumVisibleLayers()) + ", got " + std::to_string(inputCIs.size()));

    aon::Array<aon::IntBuffer> cInputCIsBacking(inputCIs.size());
    aon::Array<const aon::IntBuffer*> cInputCIs(inputCIs.size());

    for (int i = 0; i < inputCIs.size(); i++) {
        aon::Int3 inputSize = enc.getVisibleLayerDesc(i).size;

        if (inputCIs[i].size() != inputSize.x * inputSize.y)
            throw std::runtime_error("Incorrect number of pixels given to Encoder! At input " + std::to_string(i) + ": Expected " + std::to_string(inputSize.x * inputSize.y) + ", got " + std::to_string(inputCIs[i].size()));

        cInputCIsBacking[i].resize(inputCIs[i].size());
        
        for (int j = 0; j < inputCIs[i].size(); j++)
            cInputCIsBacking[i][j] = inputCIs[i][j];

        cInputCIs[i] = &cInputCIsBacking[i];
    }

    enc.step(cInputCIs, learnEnabled);
}

std::vector<int> Encoder::reconstruct(
    const std::vector<int> &hiddenCIs,
    int vli
) {
    if (hiddenCIs.size() != enc.getHiddenCIs().size())
        throw std::runtime_error("Error: hiddenCIs must match the hiddenSize of the Encoder!");

    aon::IntBuffer cHiddenCIsBacking(hiddenCIs.size());

    for (int j = 0; j < hiddenCIs.size(); j++) {
        if (hiddenCIs[j] < 0 || hiddenCIs[j] >= enc.getHiddenSize().z)
            throw std::runtime_error("Hidden CSDR (hiddenCIs) has an out-of-bounds column index (" + std::to_string(hiddenCIs[j]) + ") at column index " + std::to_string(j) + ". It must be in the range [0, " + std::to_string(enc.getHiddenSize().z - 1) + "]");

        cHiddenCIsBacking[j] = hiddenCIs[j];
    }

    aon::Int3 reconSize = enc.getVisibleLayerDesc(vli).size;

    aon::IntBuffer cReconCIs(reconSize.x * reconSize.y);

    enc.reconstruct(&cHiddenCIsBacking, &cReconCIs, vli);

    std::vector<int> reconCIs(cReconCIs.size());

    for (int j = 0; j < reconCIs.size(); j++)
        reconCIs[j] = cReconCIs[j];

    return reconCIs;
}
