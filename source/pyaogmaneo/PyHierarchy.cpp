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
    const std::vector<PyInt3> &inputSizes,
    const std::vector<int> &inputTypes,
    const std::vector<PyLayerDesc> &layerDescs
) {
    aon::Array<aon::Int3> cInputSizes(inputSizes.size());

    for (int i = 0; i < inputSizes.size(); i++)
        cInputSizes[i] = aon::Int3(inputSizes[i].x, inputSizes[i].y, inputSizes[i].z);
    
    aon::Array<aon::InputType> cInputTypes(inputTypes.size());

    for (int i = 0; i < inputTypes.size(); i++) {
        switch(inputTypes[i]) {
        case inputTypeNone:
            cInputTypes[i] = aon::none;
            break;
        case inputTypePrediction:
            cInputTypes[i] = aon::prediction;
            break;
        case inputTypeAction:
            cInputTypes[i] = aon::action;
            break;
        }
    }

    aon::Array<aon::Hierarchy::LayerDesc> cLayerDescs(layerDescs.size());

    for (int l = 0; l < layerDescs.size(); l++) {
        cLayerDescs[l].hiddenSize = aon::Int3(layerDescs[l].hiddenSize.x, layerDescs[l].hiddenSize.y, layerDescs[l].hiddenSize.z);
        cLayerDescs[l].ffRadius = layerDescs[l].ffRadius;
        cLayerDescs[l].lRadius = layerDescs[l].lRadius;
        cLayerDescs[l].pRadius = layerDescs[l].pRadius;
        cLayerDescs[l].aRadius = layerDescs[l].aRadius;
        cLayerDescs[l].temporalHorizon = layerDescs[l].temporalHorizon;
        cLayerDescs[l].ticksPerUpdate = layerDescs[l].ticksPerUpdate;
        cLayerDescs[l].historyCapacity = layerDescs[l].historyCapacity;
    }

    h.initRandom(cInputSizes, cInputTypes, cLayerDescs);
}

void PyHierarchy::step(
    const std::vector<std::vector<unsigned char> > &inputCs,
    bool learnEnabled,
    float reward,
    bool mimic
) {
    assert(inputCs.size() == h.getInputSizes().size());

    aon::Array<aon::ByteBuffer> cInputCsBacking(inputCs.size());
    aon::Array<const aon::ByteBuffer*> cInputCs(inputCs.size());

    for (int i = 0; i < inputCs.size(); i++) {
        assert(inputCs[i].size() == h.getInputSizes()[i].x * h.getInputSizes()[i].y);

        cInputCsBacking[i].resize(inputCs[i].size());

        for (int j = 0; j < inputCs[i].size(); j++)
            cInputCsBacking[i][j] = inputCs[i][j];

        cInputCs[i] = &cInputCsBacking[i];
    }
    
    h.step(cInputCs, learnEnabled, reward, mimic);
}