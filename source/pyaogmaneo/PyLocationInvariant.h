// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/LocationInvariant.h>

namespace pyaon {
const int locationInvariantMagic = 111534;

class LocationInvariant {
private:
    bool initialized;

    void initCheck() const;

    aon::LocationInvariant li;

public:
    LocationInvariant() 
    :
    initialized(false)
    {}

    void initRandom(
        const std::tuple<int, int, int> &hiddenSize,
        const std::tuple<int, int, int> &sensorSize,
        const std::tuple<int, int, int> &whereSize,
        int radius
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
        const std::vector<int> &sensorCIs,
        const std::vector<int> &whereCIs,
        bool learnEnabled
    );

    void clearState() {
        li.clearState();
    }

    std::vector<int> getHiddenCIs() const {
        initCheck();

        std::vector<int> hiddenCIs(li.getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = li.getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::vector<unsigned char> getMemory() const {
        initCheck();

        std::vector<unsigned char> memory(li.getMemory().size());

        for (int j = 0; j < memory.size(); j++)
            memory[j] = li.getMemory()[j];

        return memory;
    }

    std::tuple<int, int, int> getHiddenSize() const {
        initCheck();

        aon::Int3 size = li.getHiddenSize();

        return { size.x, size.y, size.z };
    }
    
    std::tuple<int, int, int> getSensorSize() const {
        initCheck();

        aon::Int3 size = li.getSensorSize();

        return { size.x, size.y, size.z };
    }

    std::tuple<int, int, int> getWhereSize() const {
        initCheck();

        aon::Int3 size = li.getWhereSize();

        return { size.x, size.y, size.z };
    }

    // Params
    void setMR(
        int mr
    ) {
        initCheck();

        if (mr < 0) {
            std::cerr << "Error: LocationInvariant MR must be >= 0" << std::endl;
            abort();
        }

        li.mr = mr;
    }

    int getMR() const {
        initCheck();

        return li.mr;
    }

    void setLR(
        float lr
    ) {
        initCheck();

        if (lr < 0.0f) {
            std::cerr << "Error: LocationInvariant LR must be >= 0.0" << std::endl;
            abort();
        }

        li.getEnc().lr = lr;
    }

    float getLR() const {
        initCheck();

        return li.getEnc().lr;
    }
};
} // namespace pyaon
