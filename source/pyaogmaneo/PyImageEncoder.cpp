// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyaon;

bool ImageEncoderVisibleLayerDesc::checkInRange() const {
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

void ImageEncoder::initCheck() const {
    if (!initialized) {
        std::cerr << "Attempted to use the ImageEncoder uninitialized!" << std::endl;
        abort();
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
        std::cerr << " - ImageEncoder: Some parameters out of range!" << std::endl;
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
        std::cerr << "Attempted to initialize ImageEncoder from incompatible file - " << name << std::endl;
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
        std::cerr << "Attempted to initialize ImageEncoder from incompatible buffer!" << std::endl;
        abort();
    }

    enc.read(reader);

    initialized = true;
}

void ImageEncoder::saveToFile(
    const std::string &name
) {
    initCheck();

    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&imageEncoderMagic, sizeof(int));

    enc.write(writer);
}

std::vector<unsigned char> ImageEncoder::serializeToBuffer() {
    initCheck();

    BufferWriter writer(enc.size() + sizeof(int));

    writer.write(&imageEncoderMagic, sizeof(int));

    enc.write(writer);

    return writer.buffer;
}

void ImageEncoder::step(
    const std::vector<std::vector<unsigned char>> &inputs,
    bool learnEnabled
) {
    initCheck();

    if (inputs.size() != enc.getNumVisibleLayers()) {
        std::cerr << "Incorrect number of inputs given to ImageEncoder! Expected " << enc.getNumVisibleLayers() << ", got " << inputs.size() << std::endl;
        abort();
    }

    aon::Array<aon::ByteBuffer> cInputsBacking(inputs.size());
    aon::Array<const aon::ByteBuffer*> cInputs(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        if (inputs[i].size() != enc.getReconstruction(i).size()) {
            std::cerr << "Incorrect number of pixels given to ImageEncoder! At input " << i << ": Expected " << enc.getReconstruction(i).size() << ", got " << inputs[i].size() << std::endl;
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
    initCheck();

    if (reconCIs.size() != enc.getHiddenCIs().size()) {
        std::cerr << "Error: reconCIs must match the outputSize of the ImageEncoder!" << std::endl;
        abort();
    }

    aon::IntBuffer cReconCIsBacking(reconCIs.size());

    for (int j = 0; j < reconCIs.size(); j++) {
        if (reconCIs[j] < 0 || reconCIs[j] >= enc.getHiddenSize().z) {
            std::cerr << "Recon CSDR (reconCIs) has an out-of-bounds column index (" << reconCIs[j] << ") at column index " << j << ". It must be in the range [0, " << (enc.getHiddenSize().z - 1) << "]" << std::endl;
            abort();
        }

        cReconCIsBacking[j] = reconCIs[j];
    }

    enc.reconstruct(&cReconCIsBacking);
}
