// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyaon;

void ImageEncoder::initRandom(
    const std::tuple<int, int, int> &hiddenSize,
    const std::vector<ImageEncoderVisibleLayerDesc> &visibleLayerDescs
) {
    aon::Array<aon::ImageEncoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        cVisibleLayerDescs[v].size = aon::Int3(std::get<0>(visibleLayerDescs[v].size), std::get<1>(visibleLayerDescs[v].size), std::get<2>(visibleLayerDescs[v].size));
        cVisibleLayerDescs[v].radius = visibleLayerDescs[v].radius;
    }

    enc.initRandom(aon::Int3(std::get<0>(hiddenSize), std::get<1>(hiddenSize), std::get<2>(hiddenSize)), cVisibleLayerDescs);
}

void ImageEncoder::initFromFile(
    const std::string &name
) {
    StreamReader reader;
    reader.ins.open(name, std::ios::binary);

    enc.read(reader);
}

void ImageEncoder::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    enc.read(reader);
}

void ImageEncoder::saveToFile(
    const std::string &name
) {
    StreamWriter writer;
    writer.outs.open(name, std::ios::binary);

    enc.write(writer);
}

std::vector<unsigned char> ImageEncoder::serializeToBuffer() {
    BufferWriter writer;

    enc.write(writer);

    return writer.buffer;
}

void ImageEncoder::step(
    const std::vector<std::vector<unsigned char> > &inputs,
    bool learnEnabled
) {
    aon::Array<aon::ByteBuffer> cInputsBacking(inputs.size());
    aon::Array<const aon::ByteBuffer*> cInputs(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
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
    aon::IntBuffer cReconCIsBacking(reconCIs.size());

    for (int j = 0; j < reconCIs.size(); j++)
        cReconCIsBacking[j] = reconCIs[j];

    enc.reconstruct(&cReconCIsBacking);
}
