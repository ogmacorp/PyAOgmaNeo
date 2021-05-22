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
    none = 0,
    prediction = 1,
    action = 2
};

struct IODesc {
    std::tuple<int, int, int> size;

    IOType type;

    int eRadius;
    int dRadius;

    IODesc(
        const std::tuple<int, int, int> &size,
        IOType type,
        int eRadius,
        int dRadius
    )
    :
    size(size),
    type(type),
    eRadius(eRadius),
    dRadius(dRadius)
    {}
};

struct LayerDesc {
    std::tuple<int, int, int> hiddenSize;

    int eRadius;
    int dRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    LayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        int eRadius,
        int dRadius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    hiddenSize(hiddenSize),
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
        float reward
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
        std::vector<int> hiddenCIs(h.getELayer(l).getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getELayer(l).getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getHiddenSize(
        int l
    ) {
        aon::Int3 size = h.getELayer(l).getHiddenSize();

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
        return h.getELayer(l).getNumVisibleLayers();
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

    void setELR(
        int l,
        float lr
    ) {
        h.getELayer(l).lr = lr;
    }

    float getELR(
        int l
    ) {
        return h.getELayer(l).lr;
    }

    void setEGroupRadius(
        int l,
        int groupRadius
    ) {
        h.getELayer(l).groupRadius = groupRadius;
    }

    int getEGroupRadius(
        int l
    ) {
        return h.getELayer(l).groupRadius;
    }

    void setDLR(
        int l,
        int v,
        float lr
    ) {
        assert(h.getDLayers(l)[v] != nullptr);

        h.getDLayers(l)[v]->lr = lr;
    }

    float getDLR(
        int l,
        int v
    ) const {
        assert(h.getDLayers(l)[v] != nullptr);

        return h.getDLayers(l)[v]->lr;
    }

    void setDTemperature(
        int l,
        int v,
        float temperature
    ) {
        assert(h.getDLayers(l)[v] != nullptr);

        h.getDLayers(l)[v]->temperature = temperature;
    }

    float getDTemperature(
        int l,
        int v
    ) const {
        return h.getDLayers(l)[v]->temperature;
    }

    void setALR(
        int i,
        float lr
    ) {
        assert(h.getALayers()[i] != nullptr);
        
        h.getALayers()[i]->lr = lr;
    }

    float getALR(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->lr;
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

    void setATraceDecay(
        int v,
        float traceDecay
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->traceDecay = traceDecay;
    }

    float getATraceDecay(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->traceDecay;
    }

    void setAEpsilon(
        int v,
        float epsilon
    ) {
        assert(h.getALayers()[v] != nullptr);
        
        h.getALayers()[v]->epsilon = epsilon;
    }

    float getAEpsilon(
        int i
    ) const {
        assert(h.getALayers()[i] != nullptr);
        
        return h.getALayers()[i]->epsilon;
    }

    // Retrieve additional parameters on the SPH's structure
    int getERadius(
        int l
    ) const {
        return h.getELayer(l).getVisibleLayerDesc(0).radius;
    }

    int getDRadius(
        int l,
        int v
    ) const {
        if (h.getALayers()[v] == nullptr) {
            assert(h.getDLayers(l)[v] != nullptr);

            return h.getDLayers(l)[v]->getVisibleLayerDesc(0).radius;
        }

        return h.getALayers()[v]->getVisibleLayerDesc(0).radius;
    }
};
} // namespace pyaon
