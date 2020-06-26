// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyImageEncoder.h"

using namespace pyaon;

PyImageEncoder::PyImageEncoder(
    const PyInt3 &hiddenSize,
    const std::vector<PyImageEncoderVisibleLayerDesc> &visibleLayerDescs
) {
    aon::Array<aon::ImageEncoder::VisibleLayerDesc> cVisibleLayerDescs(visibleLayerDescs.size());

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        cVisibleLayerDescs[v].size = aon::Int3(visibleLayerDescs[v].size.x, visibleLayerDescs[v].size.y, visibleLayerDescs[v].size.z);
        cVisibleLayerDescs[v].radius = visibleLayerDescs[v].radius;
    }

    enc.initRandom(aon::Int3(hiddenSize.x, hiddenSize.y, hiddenSize.z), cVisibleLayerDescs);

    alpha = enc.alpha;
    gamma = enc.gamma;
}

PyImageEncoder::PyImageEncoder(
    const std::string &name
) {
    PyStreamReader reader;
    reader.ins.open(name, std::ios::binary);

    enc.read(reader);

    alpha = enc.alpha;
    gamma = enc.gamma;
}

void PyImageEncoder::save(
    const std::string &name
) {
    PyStreamWriter writer;
    writer.outs.open(name, std::ios::binary);

    enc.write(writer);
}

void PyImageEncoder::step(
    const std::vector<std::vector<unsigned char> > &inputs,
    bool learnEnabled
) {
    enc.alpha = alpha;
    enc.gamma = gamma;
    
    aon::Array<aon::ByteBuffer> cInputsBacking(inputs.size());
    aon::Array<const aon::Array<unsigned char>*> cInputs(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        cInputsBacking[i].resize(inputs[i].size());
        
        for (int j = 0; j < inputs[i].size(); j++)
            cInputsBacking[i][j] = inputs[i][j];

        cInputs[i] = &cInputsBacking[i];
    }

    enc.step(cInputs, learnEnabled);
}

void PyImageEncoder::reconstruct(
    const std::vector<unsigned char> &reconCs
) {
    aon::ByteBuffer cReconCsBacking(reconCs.size());

    for (int j = 0; j < reconCs.size(); j++)
        cReconCsBacking[j] = reconCs[j];

    enc.reconstruct(&cReconCsBacking);
}