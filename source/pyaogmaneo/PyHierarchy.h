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
const int typeNone = 0;
const int typePrediction = 1;
const int typeAction = 2;

inline void setNumThreads(int numThreads) {
    aon::setNumThreads(numThreads);
}

inline int getNumThreads() {
    return aon::getNumThreads();
}

struct PyIODesc {
    PyInt3 size;

    int type;

    PyIODesc()
    :
    size(4, 4, 16),
    type(0)
    {}

    PyIODesc(
        const PyInt3 &size,
        int type
    )
    :
    size(size),
    type(type)
    {}
};

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
    PyHierarchy() {}

    void initRandom(
        const std::vector<PyIODesc> &ioDescs,
        const std::vector<PyLayerDesc> &layerDescs
    );

    void initFromFile(
        const std::string &name
    );

    void initFromBuffer(
        const std::vector<unsigned char> &buffer
    );

    void saveToFile(
        const std::string &name
    );

    std::vector<unsigned char> serializeToBuffer();

    void step(
        const std::vector<std::vector<int> > &inputCIs,
        bool learnEnabled = true,
        float reward = 0.0f,
        bool mimic = false
    );

    int getNumLayers() const {
        return h.getNumLayers();
    }

    std::vector<int> getPredictionCIs(
        int i
    ) const {
        std::vector<int> predictions(h.getPredictionCIs(i).size());

        for (int j = 0; j < predictions.size(); j++)
            predictions[j] = h.getPredictionCIs(i)[j];

        return predictions;
    }

    bool getUpdate(
        int l
    ) const {
        return h.getUpdate(l);
    }

    std::vector<int> getHiddenCIs(
        int l
    ) {
        std::vector<int> hiddenCIs(h.getSCLayer(l).getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getSCLayer(l).getHiddenCIs()[j];

        return hiddenCIs;
    }

    PyInt3 getHiddenSize(
        int l
    ) {
        aon::Int3 size = h.getSCLayer(l).getHiddenSize();

        return PyInt3(size.x, size.y, size.z);
    }

    int getTicks(
        int l
    ) const {
        return h.getTicks(l);
    }

    int getTicksPerUpdate(
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

        return PyInt3(size.x, size.y, size.z);
    }

    bool pLayerExists(
        int l,
        int v
    ) const {
        return h.getPLayers(l)[v] != nullptr;
    }

    bool aLayerExists(
        int v
    ) const {
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

    // Retrieve additional parameters on the SPH's structure
    int getFFRadius(
        int l
    ) const {
        return h.getSCLayer(l).getVisibleLayerDesc(0).radius;
    }

    int getPRadius(
        int l,
        int v
    ) const {
        return h.getPLayers(l)[v]->getVisibleLayerDesc(0).radius;
    }

    int getARadius(
        int v
    ) const {
        return h.getALayers()[v]->getVisibleLayerDesc(0).radius;
    }

    int getAHistoryCapacity(
        int v
    ) const {
        return h.getALayers()[v]->getHistoryCapacity();
    }
};
} // namespace pyaon
