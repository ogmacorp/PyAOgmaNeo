// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
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
    int dRadius;

    int historyCapacity;

    IODesc(
        const std::tuple<int, int, int> &size,
        IOType type,
        int hRadius,
        int eRadius,
        int dRadius,
        int historyCapacity
    )
    :
    size(size),
    type(type),
    hRadius(hRadius),
    eRadius(eRadius),
    dRadius(dRadius),
    historyCapacity(historyCapacity)
    {}
};

struct LayerDesc {
    std::tuple<int, int, int> hiddenSize;
    std::tuple<int, int, int> errorSize;

    int hRadius;
    int eRadius;
    int dRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    LayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        const std::tuple<int, int, int> &errorSize,
        int hRadius,
        int eRadius,
        int dRadius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    hiddenSize(hiddenSize),
    errorSize(errorSize),
    hRadius(hRadius),
    eRadius(eRadius),
    dRadius(dRadius),
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
        std::vector<int> hiddenCIs(h.getEncLayer(l).hidden.getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getEncLayer(l).hidden.getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::vector<int> getErrorCIs(
        int l
    ) {
        std::vector<int> errorCIs(h.getEncLayer(l).error.getHiddenCIs().size());

        for (int j = 0; j < errorCIs.size(); j++)
            errorCIs[j] = h.getEncLayer(l).error.getHiddenCIs()[j];

        return errorCIs;
    }

    std::tuple<int, int, int> getHiddenSize(
        int l
    ) {
        aon::Int3 size = h.getEncLayer(l).hidden.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    std::tuple<int, int, int> getErrorSize(
        int l
    ) {
        aon::Int3 size = h.getEncLayer(l).error.getHiddenSize();

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

    int getNumEncVisibleLayers(
        int l
    ) {
        return h.getEncLayer(l).hidden.getNumVisibleLayers();
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

    void setHLR(
        int l,
        float lr
    ) {
        h.getEncLayer(l).hidden.lr = lr;
    }

    float getHLR(
        int l
    ) {
        return h.getEncLayer(l).hidden.lr;
    }

    void setELR(
        int l,
        float lr
    ) {
        h.getEncLayer(l).error.lr = lr;
    }

    float getELR(
        int l
    ) {
        return h.getEncLayer(l).error.lr;
    }

    void setDLR(
        int l,
        int i,
        int t,
        float lr
    ) {
        h.getDLayers(l)[i][t].lr = lr;
    }

    float getDLR(
        int l,
        int i,
        int t
    ) const {
        return h.getDLayers(l)[i][t].lr;
    }

    void setAVLR(
        int i,
        float vlr
    ) {
        assert(h.getALayers()[i] != nullptr);
        
        h.getALayers()[i]->vlr = vlr;
    }

    float getAVLR(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->vlr;
    }

    void setAALR(
        int i,
        float alr
    ) {
        assert(h.getALayers()[i] != nullptr);
        
        h.getALayers()[i]->alr = alr;
    }

    float getAALR(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->alr;
    }

    void setADiscount(
        int v,
        float discount
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->discount = discount;
    }

    float getADiscount(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->discount;
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
        return h.getEncLayer(l).hidden.getVisibleLayerDesc(0).radius;
    }

    int getERadius(
        int l
    ) const {
        return h.getEncLayer(l).error.getVisibleLayerDesc(0).radius;
    }

    int getDRadius(
        int l,
        int i
    ) const {
        return h.getDLayers(l)[i][0].getVisibleLayerDesc(0).radius;
    }

    int getAHistoryCapacity(
        int i
    ) const {
        return h.getALayers()[i]->getHistoryCapacity();
    }
};
} // namespace pyaon
