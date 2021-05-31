// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyaon;

void Hierarchy::initRandom(
    const std::vector<IODesc> &ioDescs,
    const std::vector<LayerDesc> &layerDescs
) {
    aon::Array<aon::Hierarchy::IODesc> cIODescs(ioDescs.size());

    for (int i = 0; i < ioDescs.size(); i++) {
        cIODescs[i] = aon::Hierarchy::IODesc(
            aon::Int3(std::get<0>(ioDescs[i].size), std::get<1>(ioDescs[i].size), std::get<2>(ioDescs[i].size)),
            static_cast<aon::IOType>(ioDescs[i].type),
            ioDescs[i].eRadius,
            ioDescs[i].dRadius,
            ioDescs[i].historyCapacity
        );
    }
    
    aon::Array<aon::Hierarchy::LayerDesc> cLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        cLayerDescs[l] = aon::Hierarchy::LayerDesc(
            aon::Int3(std::get<0>(layerDescs[l].hiddenSize), std::get<1>(layerDescs[l].hiddenSize), std::get<2>(layerDescs[l].hiddenSize)),
            layerDescs[l].eRadius,
            layerDescs[l].dRadius,
            layerDescs[l].ticksPerUpdate,
            layerDescs[l].temporalHorizon
        );
    }

    h.initRandom(cIODescs, cLayerDescs);
}

void Hierarchy::initFromFile(
    const std::string &name
) {
    FileReader reader;
    reader.ins.open(name, std::ios::binary);

    h.read(reader);
}

void Hierarchy::initFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    h.read(reader);
}

void Hierarchy::saveToFile(
    const std::string &name
) {
    FileWriter writer;
    writer.outs.open(name, std::ios::binary);

    h.write(writer);
}

std::vector<unsigned char> Hierarchy::serializeToBuffer() {
    BufferWriter writer(h.size());

    h.write(writer);

    return writer.buffer;
}

void Hierarchy::setStateFromBuffer(
    const std::vector<unsigned char> &buffer
) {
    BufferReader reader;
    reader.buffer = &buffer;

    h.readState(reader);
}

std::vector<unsigned char> Hierarchy::serializeStateToBuffer() {
    BufferWriter writer(h.stateSize());

    h.writeState(writer);

    return writer.buffer;
}

void Hierarchy::step(
    const std::vector<std::vector<int> > &inputCIs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    assert(inputCIs.size() == h.getInputSizes().size());

    aon::Array<aon::IntBuffer> cInputCIsBacking(inputCIs.size());
    aon::Array<const aon::IntBuffer*> cInputCIs(inputCIs.size());

    for (int i = 0; i < inputCIs.size(); i++) {
        assert(inputCIs[i].size() == h.getInputSizes()[i].x * h.getInputSizes()[i].y);

        cInputCIsBacking[i].resize(inputCIs[i].size());

        for (int j = 0; j < inputCIs[i].size(); j++)
            cInputCIsBacking[i][j] = inputCIs[i][j];

        cInputCIs[i] = &cInputCIsBacking[i];
    }
    
    h.step(cInputCIs, learnEnabled, reward, mimic);
}
