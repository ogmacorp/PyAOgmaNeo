// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyaon;

bool ImageEncoderVisibleLayerDesc::checkInRange() const {
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

    if (radius < 0) {
        throw std::runtime_error("Error: radius < 0 is not allowed!");
        return false;
    }

    return true;
}

ImageEncoder::ImageEncoder(
    const std::tuple<int, int, int> &hiddenSize,
    const std::vector<ImageEncoderVisibleLayerDesc> &visibleLayerDescs,
    const std::string &name,
    const std::vector<unsigned char> &buffer
) {
    if (!buffer.empty())
        initFromBuffer(buffer);
    else if (!name.empty())
        initFromFile(name);
    else {
        if (visibleLayerDescs.empty())
            throw std::runtime_error("Error: ImageEncoder constructor requires some non-empty arguments!");

        initRandom(hiddenSize, visibleLayerDescs);
    }
}

void ImageEncoder::initRandom(
    const std::tuple<int, int, int> &hiddenSize,
    const std::vector<ImageEncoderVisibleLayerDesc> &visibleLayerDescs
) {
    bool allInRange = true;

    aon::Array<aon::ImageEncoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        if (!visibleLayerDescs[v].checkInRange()) {
            throw std::runtime_error(" - at visibleLayerDescs[" + std::to_string(v) + "]");
            allInRange = false;
        }

        cVisibleLayerDescs[v].size = aon::Int3(std::get<0>(visibleLayerDescs[v].size), std::get<1>(visibleLayerDescs[v].size), std::get<2>(visibleLayerDescs[v].size));
        cVisibleLayerDescs[v].radius = visibleLayerDescs[v].radius;
    }

    if (std::get<0>(hiddenSize) < 1) {
        throw std::runtime_error("Error: hiddenSize[0] < 1 is not allowed!");
        allInRange = false;
    }

    if (std::get<1>(hiddenSize) < 1) {
        throw std::runtime_error("Error: hiddenSize[1] < 1 is not allowed!");
        allInRange = false;
    }

    if (std::get<2>(hiddenSize) < 1) {
        throw std::runtime_error("Error: hiddenSize[2] < 1 is not allowed!");
        allInRange = false;
    }

    if (!allInRange) {
        throw std::runtime_error(" - ImageEncoder: Some parameters out of range!");
        abort();
    }

    enc.initRandom(aon::Int3(std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize)), cVisibleLayerDescs);

    initialized = true;
}

void ImageEncoder::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != imageEncoderMagic) {
        throw std::runtime_error("Attempted to initialize ImageEncoder from incompatible file - " + name);
        abort();
    }

    enc.read(reader);

    initialized = true;
}

void ImageEncoder::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != imageEncoderMagic) {
        throw std::runtime_error("Attempted to initialize ImageEncoder from incompatible buffer!");
        abort();
    }

    enc.read(reader);

    initialized = true;
}

void ImageEncoder::saveToFile(
    const std::string &name
) {
    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&imageEncoderMagic, sizeof(int));

    enc.write(writer);
}

std::vector<unsigned char> ImageEncoder::serializeToBuffer() {
    BufferWriter writer(enc.size() + sizeof(int));

    writer.write(&imageEncoderMagic, sizeof(int));

    enc.write(writer);

    return writer.buffer;
}

void ImageEncoder::step(
    const std::vector<std::vector<unsigned char>> &inputs,
    bool learnEnabled
) {
    if (inputs.size() != enc.getNumVisibleLayers()) {
        throw std::runtime_error("Incorrect number of inputs given to ImageEncoder! Expected " + std::to_string(enc.getNumVisibleLayers()) + ", got " + std::to_string(inputs.size()));
        abort();
    }

    aon::Array<aon::ByteBuffer> cInputsBacking(inputs.size());
    aon::Array<const aon::ByteBuffer*> cInputs(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        if (inputs[i].size() != enc.getReconstruction(i).size()) {
            throw std::runtime_error("Incorrect number of pixels given to ImageEncoder! At input " + std::to_string(i) + ": Expected " + std::to_string(enc.getReconstruction(i).size()) + ", got " + std::to_string(inputs[i].size()));
            abort();
        }

        cInputsBacking[i].resize(inputs[i].size());
        
        for (int j = 0; j < inputs[i].size(); j++)
            cInputsBacking[i][j] = inputs[i][j];

        cInputs[i] = &cInputsBacking[i];
    }

    enc.step(cInputs, learnEnabled);
}

void ImageEncoder::reconstruct(
    const std::vector<int> &reconCIs
) {
    if (reconCIs.size() != enc.getHiddenCIs().size()) {
        throw std::runtime_error("Error: reconCIs must match the outputSize of the ImageEncoder!");
        abort();
    }

    aon::IntBuffer cReconCIsBacking(reconCIs.size());

    for (int j = 0; j < reconCIs.size(); j++) {
        if (reconCIs[j] < 0 || reconCIs[j] >= enc.getHiddenSize().z) {
            throw std::runtime_error("Recon CSDR (reconCIs) has an out-of-bounds column index (" + std::to_string(reconCIs[j]) + ") at column index " + std::to_string(j) + ". It must be in the range [0, " + std::to_string(enc.getHiddenSize().z - 1) + "]");
            abort();
        }

        cReconCIsBacking[j] = reconCIs[j];
    }

    enc.reconstruct(&cReconCIsBacking);
}
