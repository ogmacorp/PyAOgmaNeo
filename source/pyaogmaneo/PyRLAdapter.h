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
class RLAdapter {
private:
    aon::RLAdapter adapter;

public:
    RLAdapter() {}

    void initRandom(
        const std::tuple<int, int, int> &hiddenSize
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
        bool learnEnabled
    );

    std::vector<int> getGoalCIs() const {
        std::vector<int> goalCIs(adapter.getGoalCIs().size());

        for (int j = 0; j < goalCIs.size(); j++)
            goalCIs[j] = adapter.getGoalCIs()[j];

        return goalCIs;
    }

    std::tuple<int, int, int> getHiddenSize() const {
        aon::Int3 size = adapter.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    // Params
    void setLR(
        float lr
    ) {
        adapter.lr = lr;
    }

    float getLR() const {
        return adapter.lr;
    }

    void setDiscount(
        float discount
    ) {
        adapter.discount = discount;
    }

    float getDiscount() const {
        return adapter.discount;
    }

    void setTraceDecay(
        float traceDecay
    ) {
        adapter.traceDecay = traceDecay;
    }

    float getTraceDecay() const {
        return adapter.traceDecay;
    }
};
} // namespace pyaon
