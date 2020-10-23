// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyaon;

void PyImageEncoder::initRandom(
    std::array<int, 3> hiddenSize,
    const std::vector<PyImageEncoderVisibleLayerDesc> &visibleLayerDescs
) {
    aon::Array<aon::ImageEncoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        cVisibleLayerDescs[v].size = aon::Int3(visibleLayerDescs[v].size[0], visibleLayerDescs[v].size[1], visibleLayerDescs[v].size[2]);
        cVisibleLayerDescs[v].radius = visibleLayerDescs[v].radius;
    }

    enc.initRandom(aon::Int3(hiddenSize[0], hiddenSize[1], hiddenSize[2]), cVisibleLayerDescs);

    alpha = enc.alpha;
    gamma = enc.gamma;
}

void PyImageEncoder::initFromFile(
    const std::string &name
) {
    PyStreamReader reader;
    reader.ins.open(name, std::ios::binary);

    enc.read(reader);

    alpha = enc.alpha;
    gamma = enc.gamma;
}

void PyImageEncoder::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    PyBufferReader reader;
    reader.buffer = &buffer;

    enc.read(reader);

    alpha = enc.alpha;
    gamma = enc.gamma;
}

void PyImageEncoder::saveToFile(
    const std::string &name
) {
    PyStreamWriter writer;
    writer.outs.open(name, std::ios::binary);

    enc.write(writer);
}

std::vector<unsigned char> PyImageEncoder::serializeToBuffer() {
    PyBufferWriter writer;

    enc.write(writer);

    return writer.buffer;
}

void PyImageEncoder::step(
    const std::vector<std::vector<float> > &inputs,
    bool learnEnabled
) {
    enc.alpha = alpha;
    enc.gamma = gamma;
    
    aon::Array<aon::FloatBuffer> cInputsBacking(inputs.size());
    aon::Array<const aon::Array<float>*> cInputs(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        cInputsBacking[i].resize(inputs[i].size());
        
        for (int j = 0; j < inputs[i].size(); j++)
            cInputsBacking[i][j] = inputs[i][j];

        cInputs[i] = &cInputsBacking[i];
    }

    enc.step(cInputs, learnEnabled);
}

void PyImageEncoder::reconstruct(
    const std::vector<int> &reconCIs
) {
    aon::IntBuffer cReconCIsBacking(reconCIs.size());

    for (int j = 0; j < reconCIs.size(); j++)
        cReconCIsBacking[j] = reconCIs[j];

    enc.reconstruct(&cReconCIsBacking);
}
