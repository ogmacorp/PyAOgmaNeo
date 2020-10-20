// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyHierarchy.h"

using namespace pyaon;

PyHierarchy::PyHierarchy(
    const std::vector<PyIODesc> &ioDescs,
    const std::vector<PyLayerDesc> &layerDescs
) {
    aon::Array<aon::Hierarchy::IODesc> cIODescs(ioDescs.size());

    for (int i = 0; i < ioDescs.size(); i++)
        cIODescs[i] = aon::Hierarchy::IODesc(aon::Int3(ioDescs[i].size.x, ioDescs[i].size.y, ioDescs[i].size.z), static_cast<aon::IOType>(ioDescs[i].type));
    
    aon::Array<aon::Hierarchy::LayerDesc> cLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        cLayerDescs[l].hiddenSize = aon::Int3(layerDescs[l].hiddenSize.x, layerDescs[l].hiddenSize.y, layerDescs[l].hiddenSize.z);
        cLayerDescs[l].ffRadius = layerDescs[l].ffRadius;
        cLayerDescs[l].pRadius = layerDescs[l].pRadius;
        cLayerDescs[l].aRadius = layerDescs[l].aRadius;
        cLayerDescs[l].temporalHorizon = layerDescs[l].temporalHorizon;
        cLayerDescs[l].ticksPerUpdate = layerDescs[l].ticksPerUpdate;
        cLayerDescs[l].historyCapacity = layerDescs[l].historyCapacity;
    }

    h.initRandom(cIODescs, cLayerDescs);
}

PyHierarchy::PyHierarchy(
    const std::string &name
) {
    PyStreamReader reader;
    reader.ins.open(name, std::ios::binary);

    h.read(reader);
}

PyHierarchy::PyHierarchy(
    const std::vector<unsigned char> &buffer
) {
    PyBufferReader reader;
    reader.buffer = &buffer;

    h.read(reader);
}

void PyHierarchy::save(
    const std::string &name
) {
    PyStreamWriter writer;
    writer.outs.open(name, std::ios::binary);

    h.write(writer);
}

std::vector<unsigned char> PyHierarchy::save() {
    PyBufferWriter writer;

    h.write(writer);

    return writer.buffer;
}

void PyHierarchy::step(
    const std::vector<std::vector<int> > &inputCs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    assert(inputCs.size() == h.getInputSizes().size());

    aon::Array<aon::IntBuffer> cInputCsBacking(inputCs.size());
    aon::Array<const aon::IntBuffer*> cInputCs(inputCs.size());

    for (int i = 0; i < inputCs.size(); i++) {
        assert(inputCs[i].size() == h.getInputSizes()[i].x * h.getInputSizes()[i].y);

        cInputCsBacking[i].resize(inputCs[i].size());

        for (int j = 0; j < inputCs[i].size(); j++)
            cInputCsBacking[i][j] = inputCs[i][j];

        cInputCs[i] = &cInputCsBacking[i];
    }
    
    h.step(cInputCs, learnEnabled, reward, mimic);
}
