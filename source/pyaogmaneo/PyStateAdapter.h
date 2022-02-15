// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/StateAdapter.h>

namespace pyaon {
const int stateAdapterMagic = 91233;

class StateAdapter {
private:
    bool initialized;

    void initCheck() const;

    aon::StateAdapter adapter;

public:
    StateAdapter()
    :
    initialized(false)
    {}

    void initRandom(
        const std::tuple<int, int, int> &hiddenSize,
        int radius,
        int historyCapacity
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
        const std::vector<int> &goalCIs,
        const std::vector<int> &hiddenCIs,
        bool learnEnabled
    );

    std::vector<int> getProgCIs() const {
        initCheck();

        std::vector<int> progCIs(adapter.getProgCIs().size());

        for (int j = 0; j < progCIs.size(); j++)
            progCIs[j] = adapter.getProgCIs()[j];

        return progCIs;
    }

    std::tuple<int, int, int> getHiddenSize() const {
        initCheck();

        aon::Int3 size = adapter.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    // Params
    void setLR(
        float lr
    ) {
        initCheck();

        if (lr < 0.0f) {
            std::cerr << "Error: StateAdapter LR must be >= 0.0" << std::endl;
            abort();
        }

        adapter.lr = lr;
    }

    float getLR() const {
        initCheck();

        return adapter.lr;
    }

    void setDiscount(
        float discount
    ) {
        initCheck();

        if (discount < 0.0f || discount >= 1.0f) {
            std::cerr << "Error: StateAdapter Discount must be >= 0.0 and < 1.0" << std::endl;
            abort();
        }

        adapter.discount = discount;
    }

    float getDiscount() const {
        initCheck();

        return adapter.discount;
    }

    void setHistoryIters(
        int historyIters
    ) {
        initCheck();

        if (historyIters < 1) {
            std::cerr << "Error: StateAdapter HistoryIters must be >= 1" << std::endl;
            abort();
        }

        adapter.historyIters = historyIters;
    }

    int getHistoryIters() const {
        initCheck();

        return adapter.historyIters;
    }
};
} // namespace pyaon
