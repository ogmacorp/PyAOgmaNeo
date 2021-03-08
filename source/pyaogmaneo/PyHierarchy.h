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
    prediction = 0,
    action = 1
};

struct IODesc {
    std::tuple<int, int, int> size;

    IOType type;

    int hRadius;
    int eRadius;
    int pRadius;
    int fbRadius;

    int historyCapacity;

    IODesc(
        const std::tuple<int, int, int> &size,
        IOType type,
        int hRadius,
        int eRadius,
        int pRadius,
        int fbRadius,
        int historyCapacity
    )
    :
    size(size),
    type(type),
    hRadius(hRadius),
    eRadius(eRadius),
    pRadius(pRadius),
    fbRadius(fbRadius),
    historyCapacity(historyCapacity)
    {}
};

struct LayerDesc {
    std::tuple<int, int, int> hiddenSize;
    std::tuple<int, int, int> errorSize;

    int hRadius;
    int eRadius;
    int pRadius;
    int fbRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    LayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        const std::tuple<int, int, int> &errorSize,
        int hRadius,
        int eRadius,
        int pRadius,
        int fbRadius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    hiddenSize(hiddenSize),
    errorSize(errorSize),
    hRadius(hRadius),
    eRadius(eRadius),
    pRadius(pRadius),
    fbRadius(fbRadius),
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

    void setStateFromBuffer(
        const std::vector<unsigned char> &buffer
    );

    std::vector<unsigned char> serializeStateToBuffer();

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
        std::vector<int> hiddenCIs(h.getSCLayer(l).hidden.getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getSCLayer(l).hidden.getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::vector<int> getErrorCIs(
        int l
    ) {
        std::vector<int> errorCIs(h.getSCLayer(l).error.getHiddenCIs().size());

        for (int j = 0; j < errorCIs.size(); j++)
            errorCIs[j] = h.getSCLayer(l).error.getHiddenCIs()[j];

        return errorCIs;
    }

    std::tuple<int, int, int> getHiddenSize(
        int l
    ) {
        aon::Int3 size = h.getSCLayer(l).hidden.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    std::tuple<int, int, int> getErrorSize(
        int l
    ) {
        aon::Int3 size = h.getSCLayer(l).error.getHiddenSize();

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
        return h.getSCLayer(l).hidden.getNumVisibleLayers();
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

    bool aLayerExists(
        int i
    ) const {
        return h.getALayers()[i] != nullptr;
    }

    void setHAlpha(
        int l,
        float alpha
    ) {
        h.getSCLayer(l).hidden.alpha = alpha;
    }

    float getHAlpha(
        int l
    ) {
        return h.getSCLayer(l).hidden.alpha;
    }

    void setEAlpha(
        int l,
        float alpha
    ) {
        h.getSCLayer(l).error.alpha = alpha;
    }

    float getEAlpha(
        int l
    ) {
        return h.getSCLayer(l).error.alpha;
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
        int i,
        float alpha
    ) {
        assert(h.getALayers()[i] != nullptr);
        
        h.getALayers()[i]->alpha = alpha;
    }

    float getAAlpha(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->alpha;
    }

    void setABeta(
        int i,
        float beta
    ) {
        assert(h.getALayers()[i] != nullptr);
        
        h.getALayers()[i]->beta = beta;
    }

    float getABeta(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->beta;
    }

    void setAGamma(
        int v,
        float gamma
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->gamma = gamma;
    }

    float getAGamma(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->gamma;
    }

    void setAMinSteps(
        int i,
        int minSteps
    ) {
        assert(h.getALayers()[i] != nullptr);

        h.getALayers()[i]->minSteps = minSteps;
    }

    int getAMinSteps(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->minSteps;
    }

    void setAHistoryIters(
        int i,
        int historyIters
    ) {
        assert(h.getALayers()[i] != nullptr);

        h.getALayers()[i]->historyIters = historyIters;
    }

    int getAHistoryIters(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->historyIters;
    }

    // Retrieve additional parameters on the SPH's structure
    int getHRadius(
        int l
    ) const {
        return h.getSCLayer(l).hidden.getVisibleLayerDesc(0).radius;
    }

    int getERadius(
        int l
    ) const {
        return h.getSCLayer(l).error.getVisibleLayerDesc(0).radius;
    }

    int getPRadius(
        int l,
        int i
    ) const {
        return h.getPLayers(l)[i][0].getVisibleLayerDesc(0).radius;
    }

    int getFBRadius(
        int l,
        int i
    ) const {
        return h.getPLayers(l)[i][0].getVisibleLayerDesc(1).radius;
    }

    int getAHistoryCapacity(
        int i
    ) const {
        return h.getALayers()[i]->getHistoryCapacity();
    }
};
} // namespace pyaon
