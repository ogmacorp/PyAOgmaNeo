// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/Hierarchy.h>

namespace pyaon {
const int hierarchyMagic = 54398723;

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

    bool checkInRange() const;
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

    bool checkInRange() const;
};

class Hierarchy {
private:
    bool initialized;

    void initCheck() const;

    aon::Hierarchy h;

public:
    Hierarchy() 
    :
    initialized(false)
    {}

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
        const std::vector<std::vector<int>> &inputCIs,
        bool learnEnabled,
        float reward,
        bool mimic
    );

    void clearState() {
        h.clearState();
    }

    int getNumLayers() const {
        initCheck();

        return h.getNumLayers();
    }

    std::vector<int> getTopHiddenCIs() const {
        initCheck();

        std::vector<int> hiddenCIs(h.getTopHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getTopHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getTopHiddenSize() const {
        initCheck();

        aon::Int3 size = h.getTopHiddenSize();

        return { size.x, size.y, size.z };
    }
    
    bool getTopUpdate() const {
        initCheck();

        return h.getTopUpdate();
    }

    void setInputImportance(
        int i,
        float importance
    ) {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        h.setInputImportance(i, importance);
    }

    float getInputImportance(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getInputImportance(i);
    }

    std::vector<int> getPredictionCIs(
        int i
    ) const;

    bool getUpdate(
        int l
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getUpdate(l);
    }

    std::vector<int> getHiddenCIs(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        std::vector<int> hiddenCIs(h.getELayer(l).getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getELayer(l).getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getHiddenSize(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        aon::Int3 size = h.getELayer(l).getHiddenSize();

        return { size.x, size.y, size.z };
    }

    int getTicks(
        int l
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getTicks(l);
    }

    int getTicksPerUpdate(
        int l
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getTicksPerUpdate(l);
    }

    int getNumEVisibleLayers(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getELayer(l).getNumVisibleLayers();
    }

    int getNumIO() const {
        initCheck();

        return h.getNumIO();
    }

    std::tuple<int, int, int> getIOSize(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        aon::Int3 size = h.getIOSize(i);

        return { size.x, size.y, size.z };
    }

    IOType getIOType(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return static_cast<IOType>(h.getIOType(i));
    }

    void setEGroupRadius(
        int l,
        int groupRadius
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (groupRadius < 0) {
            std::cerr << "Error: EGroupRadius must be >= 0" << std::endl;
            abort();
        }

        h.getELayer(l).groupRadius = groupRadius;
    }

    int getEGroupRadius(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getELayer(l).groupRadius;
    }

    void setELR(
        int l,
        float lr
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (lr < 0.0f) {
            std::cerr << "Error: ELR must be >= 0.0" << std::endl;
            abort();
        }

        h.getELayer(l).lr = lr;
    }

    float getELR(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getELayer(l).lr;
    }

    void setDLR(
        int l,
        int i,
        float lr
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (l == 0 && !h.ioLayerExists(i) || h.getIOType(i) != aon::prediction) {
            std::cerr << "Error: index " << i << " does not have a decoder!" << std::endl;
            abort();
        }

        if (lr < 0.0f) {
            std::cerr << "Error: DLR must be >= 0.0" << std::endl;
            abort();
        }

        h.getDLayer(l, i).lr = lr;
    }

    float getDLR(
        int l,
        int i
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (l == 0 && !h.ioLayerExists(i) || h.getIOType(i) != aon::prediction) {
            std::cerr << "Error: index " << i << " does not have a decoder!" << std::endl;
            abort();
        }

        return h.getDLayer(l, i).lr;
    }

    void setAVLR(
        int i,
        float vlr
    ) {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        if (vlr < 0.0f) {
            std::cerr << "Error: AVLR must be >= 0.0" << std::endl;
            abort();
        }

        h.getALayer(i).vlr = vlr;
    }

    float getAVLR(
        int i
    ) const {
        initCheck();
        
        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        return h.getALayer(i).vlr;
    }

    void setAALR(
        int i,
        float alr
    ) {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        if (alr < 0.0f) {
            std::cerr << "Error: AALR must be >= 0.0" << std::endl;
            abort();
        }

        h.getALayer(i).alr = alr;
    }

    float getAALR(
        int i
    ) const {
        initCheck();
        
        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        return h.getALayer(i).alr;
    }

    void setABias(
        int i,
        float bias
    ) {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        if (bias < 0.0f) {
            std::cerr << "Error: ABias must be >= 0.0" << std::endl;
            abort();
        }

        h.getALayer(i).bias = bias;
    }

    float getABias(
        int i
    ) const {
        initCheck();
        
        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        return h.getALayer(i).bias;
    }

    void setADiscount(
        int i,
        float discount
    ) {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        if (discount < 0.0f || discount >= 1.0f) {
            std::cerr << "Error: ADiscount must be >= 0.0 and < 1.0" << std::endl;
            abort();
        }

        h.getALayer(i).discount = discount;
    }

    float getADiscount(
        int i
    ) const {
        initCheck();
        
        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        return h.getALayer(i).discount;
    }

    void setATemperature(
        int i,
        float temperature
    ) {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        if (temperature < 0.0f) {
            std::cerr << "Error: ATemperature must be >= 0.0" << std::endl;
            abort();
        }

        h.getALayer(i).temperature = temperature;
    }

    float getATemperature(
        int i
    ) const {
        initCheck();
        
        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        return h.getALayer(i).temperature;
    }

    void setAMinSteps(
        int i,
        int minSteps
    ) {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        if (minSteps < 1) {
            std::cerr << "Error: AMinSteps must be >= 1" << std::endl;
            abort();
        }

        h.getALayer(i).minSteps = minSteps;
    }

    int getAMinSteps(
        int i
    ) const {
        initCheck();
        
        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i)) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        return h.getALayer(i).minSteps;
    }

    void setAHistoryIters(
        int i,
        int historyIters
    ) {
        initCheck();

        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i) || h.getIOType(i) != aon::action) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        if (historyIters < 0) {
            std::cerr << "Error: AHistoryIters must be >= 0" << std::endl;
            abort();
        }

        h.getALayer(i).historyIters = historyIters;
    }

    int getAHistoryIters(
        int i
    ) const {
        initCheck();
        
        if (i < 0 || i >= h.getNumIO()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (!h.ioLayerExists(i)) {
            std::cerr << "Error: index " << i << " does not have an actor!" << std::endl;
            abort();
        }

        return h.getALayer(i).historyIters;
    }

    // Retrieve additional parameters on the SPH's structure
    int getERadius(
        int l
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getELayer(l).getVisibleLayerDesc(0).radius;
    }

    int getDRadius(
        int l,
        int i
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getNumIO() || h.getIOType(i) != aon::prediction) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getDLayer(l, i).getVisibleLayerDesc(0).radius;
    }

    int getARadius(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= h.getNumIO() || h.getIOType(i) != aon::action) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getALayer(i).getVisibleLayerDesc(0).radius;
    }

    int getAHistoryCapacity(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= h.getNumIO() || h.getIOType(i) != aon::action) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getALayer(i).getHistoryCapacity();
    }
};
} // namespace pyaon
