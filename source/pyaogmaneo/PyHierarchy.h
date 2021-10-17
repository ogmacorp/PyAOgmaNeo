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
    prediction = 1
};

struct IODesc {
    std::tuple<int, int, int> size;

    IOType type;

    int ffRadius;
    int fbRadius;

    int historyCapacity;

    IODesc(
        const std::tuple<int, int, int> &size,
        IOType type,
        int ffRadius,
        int fbRadius,
        int historyCapacity
    )
    :
    size(size),
    type(type),
    ffRadius(ffRadius),
    fbRadius(fbRadius),
    historyCapacity(historyCapacity)
    {}
};

struct LayerDesc {
    std::tuple<int, int, int> hiddenSize;

    int ffRadius;
    int rRadius;
    int fbRadius;

    int historyCapacity;

    LayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        int ffRadius,
        int rRadius,
        int fbRadius,
        int historyCapacity
    )
    :
    hiddenSize(hiddenSize),
    ffRadius(ffRadius),
    rRadius(rRadius),
    fbRadius(fbRadius),
    historyCapacity(historyCapacity)
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

    void setRecurrence(
        int l,
        float recurrence
    ) {
        h.setRecurrence(l, recurrence);
    }

    float getRecurrence(
        int l
    ) const {
        return h.getRecurrence(l);
    }

    std::vector<int> getPredictionCIs(
        int i
    ) const {
        std::vector<int> predictions(h.getPredictionCIs(i).size());

        for (int j = 0; j < predictions.size(); j++)
            predictions[j] = h.getPredictionCIs(i)[j];

        return predictions;
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

    std::vector<int> getTopHiddenCIs() {
        std::vector<int> hiddenCIs(h.getTopHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getTopHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getTopHiddenSize() {
        aon::Int3 size = h.getTopHiddenSize();

        return { size.x, size.y, size.z };
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

    void setDDiscount(
        int l,
        int v,
        float discount
    ) {
        assert(h.getDLayers(l)[v] != nullptr);

        h.getDLayers(l)[v]->discount = discount;
    }

    float getDDiscount(
        int l,
        int v
    ) const {
        assert(h.getDLayers(l)[v] != nullptr);

        return h.getDLayers(l)[v]->discount;
    }

    void setDGenGoalNoise(
        int l,
        int v,
        float genGoalNoise
    ) {
        assert(h.getDLayers(l)[v] != nullptr);

        h.getDLayers(l)[v]->genGoalNoise = genGoalNoise;
    }

    float getDGenGoalNoise(
        int l,
        int v
    ) const {
        assert(h.getDLayers(l)[v] != nullptr);

        return h.getDLayers(l)[v]->genGoalNoise;
    }

    void setDQSteps(
        int l,
        int v,
        int qSteps
    ) {
        assert(h.getDLayers(l)[v] != nullptr);

        h.getDLayers(l)[v]->qSteps = qSteps;
    }

    int getDQSteps(
        int l,
        int v
    ) const {
        assert(h.getDLayers(l)[v] != nullptr);

        return h.getDLayers(l)[v]->qSteps;
    }

    void setDHistoryIters(
        int l,
        int v,
        int historyIters
    ) {
        assert(h.getDLayers(l)[v] != nullptr);

        h.getDLayers(l)[v]->historyIters = historyIters;
    }

    int getDHistoryIters(
        int l,
        int v
    ) const {
        assert(h.getDLayers(l)[v] != nullptr);

        return h.getDLayers(l)[v]->historyIters;
    }

    // Retrieve additional parameters on the SPH's structure
    int getFFRadius(
        int l
    ) const {
        return h.getELayer(l).getVisibleLayerDesc(0).radius;
    }

    int getRRadius(
        int l
    ) const {
        return h.getELayer(l).getVisibleLayerDesc(h.getELayer(l).getNumVisibleLayers() - 1).radius;
    }

    int getFBRadius(
        int l
    ) const {
        return h.getDLayers(l)[0]->getVisibleLayerDesc().radius;
    }
};
} // namespace pyaon
