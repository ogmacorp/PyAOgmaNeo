// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyConstructs.h"
#include <aogmaneo/Hierarchy.h>
#include <vector>
#include <fstream>

namespace pyaon {
const int inputTypeNone = 0;
const int inputTypePrediction = 1;
const int inputTypeAction = 2;

inline void setNumThreads(int numThreads) {
    aon::setNumThreads(numThreads);
}

inline int getNumThreads() {
    return aon::getNumThreads();
}

struct PyLayerDesc {
    PyInt3 hiddenSize;
    PyInt2 clumpSize;

    int ffRadius;
    int pRadius;
    int aRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    int historyCapacity;

    PyLayerDesc()
    :
    hiddenSize(4, 4, 16),
    clumpSize(2, 2),
    ffRadius(2),
    pRadius(2),
    aRadius(2),
    ticksPerUpdate(2),
    temporalHorizon(2),
    historyCapacity(32)
    {}

    PyLayerDesc(
        const PyInt3 &hiddenSize,
        const PyInt2 &clumpSize,
        int ffRadius,
        int pRadius,
        int aRadius,
        int ticksPerUpdate,
        int temporalHorizon,
        int historyCapacity
    )
    :
    hiddenSize(hiddenSize),
    clumpSize(clumpSize),
    ffRadius(ffRadius),
    pRadius(pRadius),
    aRadius(aRadius),
    ticksPerUpdate(ticksPerUpdate),
    temporalHorizon(temporalHorizon),
    historyCapacity(32)
    {}
};

class PyHierarchy {
private:
    aon::Hierarchy h;

public:
    PyHierarchy(
        const std::vector<PyInt3> &inputSizes,
        const std::vector<int> &inputTypes,
        const std::vector<PyLayerDesc> &layerDescs
    );

    PyHierarchy(
        const std::string &name
    );

    void save(
        const std::string &name
    );

    void step(
        const std::vector<std::vector<unsigned char> > &inputCs,
        bool learnEnabled = true,
        float reward = 0.0f
    );

    int getNumLayers() const {
        return h.getNumLayers();
    }

    std::vector<unsigned char> getPredictionCs(
        int i
    ) const {
        std::vector<unsigned char> predictions(h.getPredictionCs(i).size());

        for (int j = 0; j < predictions.size(); j++)
            predictions[j] = h.getPredictionCs(i)[j];

        return predictions;
    }

    bool getUpdate(
        int l
    ) const {
        return h.getUpdate(l);
    }

    std::vector<unsigned char> getHiddenCs(
        int l
    ) {
        std::vector<unsigned char> hiddenCs(h.getSCLayer(l).getHiddenCs().size());

        for (int j = 0; j < hiddenCs.size(); j++)
            hiddenCs[j] = h.getSCLayer(l).getHiddenCs()[j];

        return hiddenCs;
    }

    PyInt3 getHiddenSize(
        int l
    ) {
        aon::Int3 size = h.getSCLayer(l).getHiddenSize();

        return { size.x, size.y, size.z };
    }

    unsigned char getTicks(
        int l
    ) const {
        return h.getTicks(l);
    }

    unsigned char getTicksPerUpdate(
        int l
    ) const {
        return h.getTicksPerUpdate(l);
    }

    int getNumSCVisibleLayers(
        int l
    ) {
        return h.getSCLayer(l).getNumVisibleLayers();
    }

    int getNumInputs() const {
        return h.getInputSizes().size();
    }

    PyInt3 getInputSize(
        int i
    ) const {
        aon::Int3 size = h.getInputSizes()[i];

        return { size.x, size.y, size.z };
    }

    bool pLayerExists(
        int l,
        int v
    ) {
        return h.getPLayers(l)[v] != nullptr;
    }

    bool aLayerExists(
        int v
    ) {
        return h.getALayers()[v] != nullptr;
    }
    
    void setSCAlpha(
        int l,
        float alpha
    ) {
        h.getSCLayer(l).alpha = alpha;
    }

    float getSCAlpha(
        int l
    ) const {
        return h.getSCLayer(l).alpha;
    }

    void setSCBeta(
        int l,
        float beta
    ) {
        h.getSCLayer(l).beta = beta;
    }

    float getSCBeta(
        int l
    ) const {
        return h.getSCLayer(l).beta;
    }

    void setSCMinVigilance(
        int l,
        float minVigilance
    ) {
        h.getSCLayer(l).minVigilance = minVigilance;
    }

    float getSCMinVigilance(
        int l
    ) const {
        return h.getSCLayer(l).minVigilance;
    }

    void setAAlpha(
        int v,
        float alpha
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->alpha = alpha;
    }

    float getAAlpha(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->alpha;
    }

    void setAGamma(
        int v,
        float gamma
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->gamma = gamma;
    }

    float getAGamma(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->gamma;
    }

    friend class PyVisualizer;
};
} // namespace pyaon