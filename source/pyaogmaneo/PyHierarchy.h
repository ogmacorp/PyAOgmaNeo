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

    int ffRadius;
    int pRadius;
    int aRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    int historyCapacity;

    PyLayerDesc()
    :
    hiddenSize(4, 4, 16),
    ffRadius(2),
    pRadius(2),
    aRadius(2),
    ticksPerUpdate(2),
    temporalHorizon(2),
    historyCapacity(32)
    {}

    PyLayerDesc(
        const PyInt3 &hiddenSize,
        int ffRadius,
        int pRadius,
        int aRadius,
        int ticksPerUpdate,
        int temporalHorizon,
        int historyCapacity
    )
    :
    hiddenSize(hiddenSize),
    ffRadius(ffRadius),
    pRadius(pRadius),
    aRadius(aRadius),
    ticksPerUpdate(ticksPerUpdate),
    temporalHorizon(temporalHorizon),
    historyCapacity(historyCapacity)
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

    PyHierarchy(
        const std::vector<unsigned char> &buffer
    );

    void save(
        const std::string &name
    );

    std::vector<unsigned char> save();

    void step(
        const std::vector<std::vector<unsigned char> > &inputCs,
        bool learnEnabled = true,
        float reward = 0.0f,
        bool mimic = false
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

    void setPAlpha(
        int l,
        int v,
        float alpha
    ) {
        assert(h.getPLayers(l)[v] != nullptr);
        
        h.getPLayers(l)[v]->alpha = alpha;
    }

    float getPAlpha(
        int l,
        int v
    ) const {
        assert(h.getPLayers(l)[v] != nullptr);
        
        return h.getPLayers(l)[v]->alpha;
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

    void setABeta(
        int v,
        float beta
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->beta = beta;
    }

    float getABeta(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->beta;
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

    void setAMinSteps(
        int v,
        int minSteps
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->minSteps = minSteps;
    }

    int getAMinSteps(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->minSteps;
    }

    void setAHistoryIters(
        int v,
        int historyIters
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->historyIters = historyIters;
    }

    int getAHistoryIters(
        int v
    ) const {
        assert(h.getALayers()[v] != nullptr);
        
        return h.getALayers()[v]->historyIters;
    }
};
} // namespace pyaon