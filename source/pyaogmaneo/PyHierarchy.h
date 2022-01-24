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
const int hierarchyMagic = 54398714;

enum IOType {
    none = 0,
    prediction = 1
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
    int gHiddenSizeZ;

    int eRadius;
    int dRadius;

    int historyCapacity;

    int ticksPerUpdate;
    int temporalHorizon;

    LayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        int gHiddenSizeZ,
        int eRadius,
        int dRadius,
        int historyCapacity,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    hiddenSize(hiddenSize),
    gHiddenSizeZ(gHiddenSizeZ),
    eRadius(eRadius),
    dRadius(dRadius),
    historyCapacity(historyCapacity),
    ticksPerUpdate(ticksPerUpdate),
    temporalHorizon(temporalHorizon)
    {}

    bool checkInRange() const;
};

struct GDesc {
    std::tuple<int, int, int> size;

    int radius;

    GDesc(
        const std::tuple<int, int, int> &size,
        int radius
    )
    :
    size(size),
    radius(radius)
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
        const std::vector<GDesc> &gDescs,
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
        const std::vector<std::vector<int>> &goalCIs,
        const std::vector<std::vector<int>> &actualCIs,
        bool learnEnabled
    );

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

    void setImportance(
        int i,
        float importance
    ) {
        initCheck();

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid layer index!" << std::endl;
            abort();
        }

        h.setImportance(i, importance);
    }

    float getImportance(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getImportance(i);
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

        return h.getIOSizes().size();
    }

    int getNumGVisibleLayers(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getGLayer(l).getNumVisibleLayers();
    }

    std::tuple<int, int, int> getIOSize(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        aon::Int3 size = h.getIOSizes()[i];

        return { size.x, size.y, size.z };
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

    void setEDecay(
        int l,
        float decay
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (decay < 0.0f || decay >= 1.0f) {
            std::cerr << "Error: EDecay must be >= 0.0 and < 1.0" << std::endl;
            abort();
        }

        h.getELayer(l).decay = decay;
    }

    float getEDecay(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getELayer(l).decay;
    }

    void setEBoost(
        int l,
        float boost
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (boost < 0.0f) {
            std::cerr << "Error: EBoost must be >= 0.0" << std::endl;
            abort();
        }

        h.getELayer(l).boost = boost;
    }

    float getEBoost(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getELayer(l).boost;
    }

    void setGLR(
        int l,
        float lr
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (lr < 0.0f) {
            std::cerr << "Error: GLR must be >= 0.0" << std::endl;
            abort();
        }

        h.getGLayer(l).lr = lr;
    }

    float getGLR(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getGLayer(l).lr;
    }

    void setGDecay(
        int l,
        float decay
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (decay < 0.0f || decay >= 1.0f) {
            std::cerr << "Error: GDecay must be >= 0.0 and < 1.0" << std::endl;
            abort();
        }

        h.getGLayer(l).decay = decay;
    }

    float getGDecay(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getGLayer(l).decay;
    }

    void setGBoost(
        int l,
        float boost
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (boost < 0.0f) {
            std::cerr << "Error: GBoost must be >= 0.0" << std::endl;
            abort();
        }

        h.getGLayer(l).boost = boost;
    }

    float getGBoost(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getGLayer(l).boost;
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

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
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

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getDLayer(l, i).lr;
    }

    void setDDecay(
        int l,
        int i,
        float decay
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (decay < 0.0f) {
            std::cerr << "Error: DDecay must be >= 0.0" << std::endl;
            abort();
        }

        h.getDLayer(l, i).decay = decay;
    }

    float getDDecay(
        int l,
        int i
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getDLayer(l, i).decay;
    }

    void setDDiscount(
        int l,
        int i,
        float discount
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (discount < 0.0f || discount >= 1.0f) {
            std::cerr << "Error: DDiscount must be >= 0.0 and < 1.0" << std::endl;
            abort();
        }

        h.getDLayer(l, i).discount = discount;
    }

    float getDDiscount(
        int l,
        int i
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getDLayer(l, i).discount;
    }

    void setDHistoryIters(
        int l,
        int i,
        int historyIters
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (historyIters < 0) {
            std::cerr << "Error: DHistoryIters must be >= 0" << std::endl;
            abort();
        }

        h.getDLayer(l, i).historyIters = historyIters;
    }

    int getDHistoryIters(
        int l,
        int i
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getDLayer(l, i).historyIters;
    }

    void setDMaxSteps(
        int l,
        int i,
        int maxSteps
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (maxSteps < 0) {
            std::cerr << "Error: DMaxSteps must be >= 0" << std::endl;
            abort();
        }

        h.getDLayer(l, i).historyIters = maxSteps;
    }

    int getDMaxSteps(
        int l,
        int i
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getDLayer(l, i).maxSteps;
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

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        return h.getDLayer(l, i).getVisibleLayerDesc().radius;
    }
};
} // namespace pyaon
