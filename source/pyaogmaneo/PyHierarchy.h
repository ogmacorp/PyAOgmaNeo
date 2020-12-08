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
enum IOType {
    none = 0,
    prediction = 1,
    action = 2
};

struct IODesc {
    std::tuple<int, int, int> size;

    IOType type;

    int ffRadius;
    int pRadius;
    int aRadius;

    int historyCapacity;

    IODesc(
        const std::tuple<int, int, int> &size,
        IOType type,
        int ffRadius,
        int pRadius,
        int aRadius,
        int historyCapacity
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

struct LayerDesc {
    std::tuple<int, int, int> hiddenSize;
    int numPriorities;

    int ffRadius;
    int pRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    LayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        int numPriorities,
        int ffRadius,
        int pRadius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    hiddenSize(hiddenSize),
    numPriorities(numPriorities),
    ffRadius(ffRadius),
    pRadius(pRadius),
    ticksPerUpdate(ticksPerUpdate),
    temporalHorizon(temporalHorizon)
    {}
};

class Hierarchy {
private:
    aon::Hierarchy h;

public:
    Hierarchy() {}

    void initRandom(
        const std::vector<IODesc> &ioDescs,
        const std::vector<LayerDesc> &layerDescs
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
        bool learnEnabled,
        float reward,
        bool mimic
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

    std::tuple<int, int, int> getHiddenSize(
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

    std::tuple<int, int, int> getInputSize(
        int i
    ) const {
        aon::Int3 size = h.getInputSizes()[i];

        return { size.x, size.y, size.z };
    }

    bool pLayerExists(
        int l,
        int v
    ) const {
        return h.getPLayers(l)[v] != nullptr;
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

    void setPTargetRange(
        int l,
        int v,
        float targetRange
    ) {
        assert(h.getPLayers(l)[v] != nullptr);

        h.getPLayers(l)[v]->targetRange = targetRange;
    }

    float getPTargetRange(
        int l,
        int v
    ) const {
        assert(h.getPLayers(l)[v] != nullptr);

        return h.getPLayers(l)[v]->targetRange;
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
