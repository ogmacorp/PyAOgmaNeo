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
struct IODesc {
    std::tuple<int, int, int> size;

    int eRadius;
    int dRadius;

    IODesc(
        const std::tuple<int, int, int> &size,
        int eRadius,
        int dRadius
    )
    :
    size(size),
    eRadius(eRadius),
    dRadius(dRadius)
    {}
};

struct LayerDesc {
    std::tuple<int, int, int> hiddenSize;
    std::tuple<int, int, int> concatSize;

    int eRadius;
    int cRadius;
    int dRadius;

    int ticksPerUpdate;
    int temporalHorizon;

    LayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        const std::tuple<int, int, int> &concatSize,
        int eRadius,
        int cRadius,
        int dRadius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    hiddenSize(hiddenSize),
    concatSize(concatSize),
    eRadius(eRadius),
    cRadius(cRadius),
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
        const std::vector<int> &topGoalCIs,
        bool learnEnabled
    );

    int getNumLayers() const {
        return h.getNumLayers();
    }

    std::vector<int> getTopHiddenCIs() const {
        std::vector<int> hiddenCIs(h.getTopHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getTopHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getTopHiddenSize() const {
        aon::Int3 size = h.getTopHiddenSize();

        return { size.x, size.y, size.z };
    }
    
    bool getTopUpdate() const {
        return h.getTopUpdate();
    }

    void setImportance(
        int i,
        float importance
    ) {
        h.setImportance(i, importance);
    }

    float getImportance(
        int i
    ) const {
        return h.getImportance(i);
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

    int getNumEVisibleLayers(
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

    void setCLR(
        int l,
        float lr
    ) {
        h.getCLayer(l).lr = lr;
    }

    float getCLR(
        int l
    ) {
        return h.getCLayer(l).lr;
    }

    void setDLR(
        int l,
        int i,
        float lr
    ) {
        h.getDLayers(l)[i].lr = lr;
    }

    float getDLR(
        int l,
        int i
    ) const {
        return h.getDLayers(l)[i].lr;
    }

    // Retrieve additional parameters on the SPH's structure
    int getERadius(
        int l
    ) const {
        return h.getELayer(l).getVisibleLayerDesc(0).radius;
    }

    int getDRadius(
        int l,
        int i
    ) const {
        return h.getDLayers(l)[i].getVisibleLayerDesc(0).radius;
    }
};
} // namespace pyaon
