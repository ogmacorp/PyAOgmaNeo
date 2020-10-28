// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/Hierarchy.h>

namespace pyaon {
const int typePrediction = 0;
const int typeAction = 1;

inline void setNumThreads(int numThreads) {
    aon::setNumThreads(numThreads);
}

inline int getNumThreads() {
    return aon::getNumThreads();
}

struct PyIODesc {
    Arr3i size;

    int type;

    int ffRadius;
    int pRadius;
    int aRadius;

    int historyCapacity;

    PyIODesc(
        Arr3i size = Arr3i({ 4, 4, 16 }),
        int type = typePrediction,
        int ffRadius  = 2,
        int pRadius = 2,
        int aRadius = 2,
        int historyCapacity = 32
    )
    :
    size(size),
    type(type),
    ffRadius(ffRadius),
    pRadius(pRadius),
    aRadius(aRadius),
    historyCapacity(historyCapacity)
    {}
};

struct PyLayerDesc {
    Arr3i hiddenSize;

    int ffRadius;
    int pRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    PyLayerDesc(
        Arr3i hiddenSize = Arr3i({ 4, 4, 16 }),
        int ffRadius = 2,
        int pRadius = 2,
        int ticksPerUpdate = 2,
        int temporalHorizon = 2
    )
    :
    hiddenSize(hiddenSize),
    ffRadius(ffRadius),
    pRadius(pRadius),
    ticksPerUpdate(ticksPerUpdate),
    temporalHorizon(temporalHorizon)
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

    Arr3i getHiddenSize(
        int l
    ) {
        aon::Int3 size = h.getSCLayer(l).getHiddenSize();

        return { size.x, size.y, size.z };
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

    Arr3i getInputSize(
        int i
    ) const {
        aon::Int3 size = h.getInputSizes()[i];

        return { size.x, size.y, size.z };
    }

    bool aLayerExists(
        int i
    ) const {
        return h.getALayers()[i] != nullptr;
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
        int i,
        int t,
        float alpha
    ) {
        h.getPLayers(l)[i][t].alpha = alpha;
    }

    float getPAlpha(
        int l,
        int i,
        int t
    ) const {
        return h.getPLayers(l)[i][t].alpha;
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
        int i,
        int t
    ) const {
        return h.getPLayers(l)[i][t].getVisibleLayerDesc(0).radius;
    }

    int getARadius(
        int i
    ) const {
        return h.getALayers()[i]->getVisibleLayerDesc(0).radius;
    }

    int getAHistoryCapacity(
        int i
    ) const {
        return h.getALayers()[i]->getHistoryCapacity();
    }
};
} // namespace pyaon
