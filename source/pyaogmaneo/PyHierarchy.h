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

    bool checkInRange() const;
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
        const std::vector<int> &topProgCIs,
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

        return h.getUpdate(l);
    }

    std::vector<int> getHiddenCIs(
        int l
    ) {
        initCheck();

        std::vector<int> hiddenCIs(h.getELayer(l).getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getELayer(l).getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getHiddenSize(
        int l
    ) {
        initCheck();

        aon::Int3 size = h.getELayer(l).getHiddenSize();

        return { size.x, size.y, size.z };
    }

    int getTicks(
        int l
    ) const {
        initCheck();

        return h.getTicks(l);
    }

    int getTicksPerUpdate(
        int l
    ) const {
        initCheck();

        return h.getTicksPerUpdate(l);
    }

    int getNumEVisibleLayers(
        int l
    ) {
        initCheck();

        return h.getELayer(l).getNumVisibleLayers();
    }

    int getNumIO() const {
        initCheck();

        return h.getIOSizes().size();
    }

    std::tuple<int, int, int> getIOSize(
        int i
    ) const {
        initCheck();

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

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid input index!" << std::endl;
            abort();
        }

        if (lr < 0.0f) {
            std::cerr << "Error: DLR must be >= 0.0" << std::endl;
            abort();
        }

        h.getDLayers(l)[i].lr = lr;
    }

    float getDLR(
        int l,
        int i
    ) const {
        initCheck();

        return h.getDLayers(l)[i].lr;
    }

    // Retrieve additional parameters on the SPH's structure
    int getERadius(
        int l
    ) const {
        initCheck();

        return h.getELayer(l).getVisibleLayerDesc(0).radius;
    }

    int getDRadius(
        int l,
        int i
    ) const {
        initCheck();

        return h.getDLayers(l)[i].getVisibleLayerDesc(0).radius;
    }
};
} // namespace pyaon
