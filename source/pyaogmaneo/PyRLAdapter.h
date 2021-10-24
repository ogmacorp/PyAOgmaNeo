// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/RLAdapter.h>

namespace pyaon {
const int rlAdapterMagic = 834903;

class RLAdapter {
private:
    bool initialized;

    void initCheck() const;

    aon::RLAdapter adapter;

public:
    RLAdapter()
    :
    initialized(false)
    {}

    void initRandom(
        const std::tuple<int, int, int> &hiddenSize,
        int numGoals
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

    void step(
        const std::vector<int> &hiddenCIs,
        float reward,
        bool learnEnabled,
        bool stateUpdate
    );

    std::vector<int> getGoalCIs() const {
        initCheck();

        std::vector<int> goalCIs(adapter.getGoalCIs().size());

        for (int j = 0; j < goalCIs.size(); j++)
            goalCIs[j] = adapter.getGoalCIs()[j];

        return goalCIs;
    }

    std::tuple<int, int, int> getHiddenSize() const {
        initCheck();

        aon::Int3 size = adapter.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    // Params
    void setGLR(
        float glr
    ) {
        initCheck();

        if (glr < 0.0f) {
            std::cerr << "Error: RLAdapter GLR must be >= 0.0" << std::endl;
            abort();
        }

        adapter.glr = glr;
    }

    float getGLR() const {
        initCheck();

        return adapter.glr;
    }

    void setVLR(
        float vlr
    ) {
        initCheck();

        if (vlr < 0.0f) {
            std::cerr << "Error: RLAdapter VLR must be >= 0.0" << std::endl;
            abort();
        }

        adapter.vlr = vlr;
    }

    float getVLR() const {
        initCheck();

        return adapter.vlr;
    }

    void setFalloff(
        float falloff
    ) {
        initCheck();

        if (falloff < 0.0f) {
            std::cerr << "Error: RLAdapter Falloff must be >= 0.0" << std::endl;
            abort();
        }

        adapter.falloff = falloff;
    }

    float getFalloff() const {
        initCheck();

        return adapter.falloff;
    }

    void setDiscount(
        float discount
    ) {
        initCheck();

        if (discount < 0.0f || discount >= 1.0f) {
            std::cerr << "Error: RLAdapter discount must be in [0.0, 1.0)" << std::endl;
            abort();
        }

        adapter.discount = discount;
    }

    float getDiscount() const {
        initCheck();

        return adapter.discount;
    }

    void setTraceDecay(
        float traceDecay
    ) {
        initCheck();

        if (traceDecay < 0.0f || traceDecay >= 1.0f) {
            std::cerr << "Error: RLAdapter traceDecay must be in [0.0, 1.0)" << std::endl;
            abort();
        }

        adapter.traceDecay = traceDecay;
    }

    float getTraceDecay() const {
        initCheck();

        return adapter.traceDecay;
    }
};
} // namespace pyaon
