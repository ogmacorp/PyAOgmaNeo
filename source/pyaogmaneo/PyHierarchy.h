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
const int hierarchyMagic = 54398714;

enum IOType {
    none = 0,
    prediction = 1
};

struct IODesc {
    std::tuple<int, int, int> size;

    int radius;

    IODesc(
        const std::tuple<int, int, int> &size,
        int radius
    )
    :
    size(size),
    radius(radius)
    {}

    bool checkInRange() const;
};

struct LayerDesc {
    std::tuple<int, int, int> hiddenSize;

    int radius;

    int ticksPerUpdate;
    int temporalHorizon;

    LayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        int radius,
        int ticksPerUpdate,
        int temporalHorizon
    )
    :
    hiddenSize(hiddenSize),
    radius(radius),
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
        const std::vector<int> &topGoalCIs,
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

    void setInputImportance(
        int i,
        float importance
    ) {
        initCheck();

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid layer index!" << std::endl;
            abort();
        }

        h.setInputImportance(i, importance);
    }

    float getInputImportance(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= h.getIOSizes().size()) {
            std::cerr << "Error: " << i << " is not a valid layer index!" << std::endl;
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

        std::vector<int> hiddenCIs(h.getLayer(l).getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = h.getLayer(l).getHiddenCIs()[j];

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

        aon::Int3 size = h.getLayer(l).getHiddenSize();

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

        return h.getLayer(l).getNumVisibleLayers();
    }

    int getNumIO() const {
        initCheck();

        return h.getIOSizes().size();
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

    void setRLR(
        int l,
        float rlr
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (rlr < 0.0f) {
            std::cerr << "Error: RLR must be >= 0.0" << std::endl;
            abort();
        }

        h.getLayer(l).rlr = rlr;
    }

    float getRLR(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getLayer(l).rlr;
    }

    void setTLR(
        int l,
        float tlr
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        if (tlr < 0.0f) {
            std::cerr << "Error: TLR must be >= 0.0" << std::endl;
            abort();
        }

        h.getLayer(l).tlr = tlr;
    }

    float getTLR(
        int l
    ) {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getLayer(l).tlr;
    }

    // Retrieve additional parameters on the SPH's structure
    int getRadius(
        int l
    ) const {
        initCheck();

        if (l < 0 || l >= h.getNumLayers()) {
            std::cerr << "Error: " << l << " is not a valid layer index!" << std::endl;
            abort();
        }

        return h.getLayer(l).getVisibleLayerDesc(0).radius;
    }
};
} // namespace pyaon
