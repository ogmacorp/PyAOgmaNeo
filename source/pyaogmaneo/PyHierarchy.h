// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/Hierarchy.h>

namespace pyaon {
const int hierarchyMagic = 54318325;

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

    int historyCapacity;

    IODesc(
        const std::tuple<int, int, int> &size,
        IOType type,
        int eRadius,
        int dRadius,
        int historyCapacity
    )
    :
    size(size),
    type(type),
    eRadius(eRadius),
    dRadius(dRadius),
    historyCapacity(historyCapacity)
    {}

    void checkInRange() const;
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

    void checkInRange() const;
};

class Hierarchy {
private:
    aon::Hierarchy h;

    void encGetSetIndexCheck(
        int l
    ) const;

    void decGetSetIndexCheck(
        int l,
        int i
    ) const;

    void actGetSetIndexCheck(
        int i
    ) const;

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

public:
    Hierarchy(
        const std::vector<IODesc> &ioDescs,
        const std::vector<LayerDesc> &layerDescs,
        const std::string &name,
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
        const std::vector<std::vector<int>> &inputCIs,
        bool learnEnabled,
        float reward,
        float mimic
    );

    void clearState() {
        h.clearState();
    }

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

    void setInputImportance(
        int i,
        float importance
    ) {
        if (i < 0 || i >= h.getNumIO())
            throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");

        h.setInputImportance(i, importance);
    }

    float getInputImportance(
        int i
    ) const {
        if (i < 0 || i >= h.getNumIO())
            throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");

        return h.getInputImportance(i);
    }

    std::vector<int> getPredictionCIs(
        int i
    ) const;

    std::vector<int> getHiddenCIs(
        int l
    ) {
        if (l < 0 || l >= h.getNumLayers())
            throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");

        std::vector<int> hiddenCIs(h.getELayer(l).getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getELayer(l).getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getHiddenSize(
        int l
    ) {
        if (l < 0 || l >= h.getNumLayers())
            throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");

        aon::Int3 size = h.getELayer(l).getHiddenSize();

        return { size.x, size.y, size.z };
    }

    int getNumEVisibleLayers(
        int l
    ) {
        if (l < 0 || l >= h.getNumLayers())
            throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");

        return h.getELayer(l).getNumVisibleLayers();
    }

    int getTicks(
        int l
    ) const {
        if (l < 0 || l >= h.getNumLayers())
            throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");

        return h.getTicks(l);
    }

    int getTicksPerUpdate(
        int l
    ) const {
        if (l < 0 || l >= h.getNumLayers())
            throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");

        return h.getTicksPerUpdate(l);
    }

    int getNumIO() const {
        return h.getNumIO();
    }

    std::tuple<int, int, int> getIOSize(
        int i
    ) const {
        if (i < 0 || i >= h.getNumIO())
            throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");

        aon::Int3 size = h.getIOSize(i);

        return { size.x, size.y, size.z };
    }

    IOType getIOType(
        int i
    ) const {
        if (i < 0 || i >= h.getNumIO())
            throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");

        return static_cast<IOType>(h.getIOType(i));
    }

    void setEGap(
        int l,
        float gap
    ) {
        encGetSetIndexCheck(l);

        if (gap <= 0.0f)
            throw std::runtime_error("Error: EGap must be > 0.0");

        h.getELayer(l).gap = gap;
    }

    float getEGap(
        int l
    ) const {
        encGetSetIndexCheck(l);

        return h.getELayer(l).gap;
    }

    void setEVigilance(
        int l,
        float vigilance
    ) {
        encGetSetIndexCheck(l);

        if (vigilance < 0.0f || vigilance > 1.0f)
            throw std::runtime_error("Error: EVigilance must be >= 0.0 and <= 1.0");

        h.getELayer(l).vigilance = vigilance;
    }

    float getEVigilance(
        int l
    ) const {
        encGetSetIndexCheck(l);

        return h.getELayer(l).vigilance;
    }

    void setELR(
        int l,
        float lr
    ) {
        encGetSetIndexCheck(l);

        if (lr < 0.0f)
            throw std::runtime_error("Error: ELR must be >= 0.0");

        h.getELayer(l).lr = lr;
    }

    float getELR(
        int l
    ) const {
        encGetSetIndexCheck(l);

        return h.getELayer(l).lr;
    }

    void setELRadius(
        int l,
        int lRadius
    ) {
        encGetSetIndexCheck(l);

        if (lRadius < 0)
            throw std::runtime_error("Error: ELRadius must be >= 0");

        h.getELayer(l).lRadius = lRadius;
    }

    int getELRadius(
        int l
    ) const {
        encGetSetIndexCheck(l);

        return h.getELayer(l).lRadius;
    }

    void setDScale(
        int l,
        int i,
        float scale
    ) {
        decGetSetIndexCheck(l, i);

        if (scale <= 0.0f)
            throw std::runtime_error("Error: DScale must be > 0.0");

        h.getDLayer(l, i).scale = scale;
    }

    float getDScale(
        int l,
        int i
    ) const {
        decGetSetIndexCheck(l, i);

        return h.getDLayer(l, i).scale;
    }

    void setDLR(
        int l,
        int i,
        float lr
    ) {
        decGetSetIndexCheck(l, i);

        if (lr < 0.0f)
            throw std::runtime_error("Error: DLR must be >= 0.0");

        h.getDLayer(l, i).lr = lr;
    }

    float getDLR(
        int l,
        int i
    ) const {
        decGetSetIndexCheck(l, i);

        return h.getDLayer(l, i).lr;
    }

    void setDStability(
        int l,
        int i,
        float stability
    ) {
        decGetSetIndexCheck(l, i);

        if (stability < 0.0f)
            throw std::runtime_error("Error: DStability must be >= 0.0");

        h.getDLayer(l, i).stability = stability;
    }

    float getDStability(
        int l,
        int i
    ) const {
        decGetSetIndexCheck(l, i);

        return h.getDLayer(l, i).stability;
    }

    void setAVLR(
        int i,
        float vlr
    ) {
        actGetSetIndexCheck(i);

        if (vlr < 0.0f)
            throw std::runtime_error("Error: AVLR must be >= 0.0");

        h.getALayer(i).vlr = vlr;
    }

    float getAVLR(
        int i
    ) const {
        actGetSetIndexCheck(i);

        return h.getALayer(i).vlr;
    }

    void setAALR(
        int i,
        float alr
    ) {
        actGetSetIndexCheck(i);

        if (alr < 0.0f)
            throw std::runtime_error("Error: AALR must be >= 0.0");

        h.getALayer(i).alr = alr;
    }

    float getAALR(
        int i
    ) const {
        actGetSetIndexCheck(i);

        return h.getALayer(i).alr;
    }

    void setABias(
        int i,
        float bias
    ) {
        actGetSetIndexCheck(i);

        if (bias < 0.0f || bias > 1.0f)
            throw std::runtime_error("Error: ABias must be >= 0.0 and <= 1.0");

        h.getALayer(i).bias = bias;
    }

    float getABias(
        int i
    ) const {
        actGetSetIndexCheck(i);

        return h.getALayer(i).bias;
    }

    void setADiscount(
        int i,
        float discount
    ) {
        actGetSetIndexCheck(i);

        if (discount < 0.0f || discount >= 1.0f)
            throw std::runtime_error("Error: ADiscount must be >= 0.0 and < 1.0");

        h.getALayer(i).discount = discount;
    }

    float getADiscount(
        int i
    ) const {
        actGetSetIndexCheck(i);

        return h.getALayer(i).discount;
    }

    void setATemperature(
        int i,
        float temperature
    ) {
        actGetSetIndexCheck(i);

        if (temperature < 0.0f)
            throw std::runtime_error("Error: ATemperature must be >= 0.0");

        h.getALayer(i).temperature = temperature;
    }

    float getATemperature(
        int i
    ) const {
        actGetSetIndexCheck(i);

        return h.getALayer(i).temperature;
    }

    void setAMinSteps(
        int i,
        int minSteps
    ) {
        actGetSetIndexCheck(i);

        if (minSteps < 1)
            throw std::runtime_error("Error: AMinSteps must be >= 1");

        h.getALayer(i).minSteps = minSteps;
    }

    int getAMinSteps(
        int i
    ) const {
        actGetSetIndexCheck(i);

        return h.getALayer(i).minSteps;
    }

    void setAHistoryIters(
        int i,
        int historyIters
    ) {
        actGetSetIndexCheck(i);

        if (historyIters < 0)
            throw std::runtime_error("Error: AHistoryIters must be >= 0");

        h.getALayer(i).historyIters = historyIters;
    }

    int getAHistoryIters(
        int i
    ) const {
        actGetSetIndexCheck(i);

        return h.getALayer(i).historyIters;
    }

    // Retrieve additional parameters on the SPH's structure
    int getERadius(
        int l
    ) const {
        if (l < 0 || l >= h.getNumLayers())
            throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");

        return h.getELayer(l).getVisibleLayerDesc(0).radius;
    }

    int getDRadius(
        int l,
        int i
    ) const {
        if (l < 0 || l >= h.getNumLayers())
            throw std::runtime_error("Error: " + std::to_string(l) + " is not a valid layer index!");

        if (l == 0 && i < 0 || i >= h.getNumIO() || h.getIOType(i) != aon::prediction)
            throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");

        return h.getDLayer(l, i).getVisibleLayerDesc(0).radius;
    }

    int getARadius(
        int i
    ) const {
        if (i < 0 || i >= h.getNumIO() || h.getIOType(i) != aon::action)
            throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");

        return h.getALayer(i).getVisibleLayerDesc(0).radius;
    }

    int getAHistoryCapacity(
        int i
    ) const {
        if (i < 0 || i >= h.getNumIO() || h.getIOType(i) != aon::action)
            throw std::runtime_error("Error: " + std::to_string(i) + " is not a valid input index!");

        return h.getALayer(i).getHistoryCapacity();
    }
};
} // namespace pyaon
