// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/Encoder.h>

namespace pyaon {
const int encoderMagic = 98332;

struct EncoderVisibleLayerDesc {
    std::tuple<int, int, int> size;

    int radius;

    EncoderVisibleLayerDesc(
        const std::tuple<int, int, int> &size,
        int radius
    )
    : 
    size(size),
    radius(radius)
    {}

    bool checkInRange() const;
};

class Encoder {
private:
    bool initialized;

    void initCheck() const;

    aon::Encoder enc;

public:
    Encoder() 
    :
    initialized(false)
    {}

    void initRandom(
        const std::tuple<int, int, int> &hiddenSize,
        const std::vector<EncoderVisibleLayerDesc> &visibleLayerDescs
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
        const std::vector<std::vector<int>> &inputCIs,
        bool learnEnabled
    );

    int getNumVisibleLayers() const {
        initCheck();

        return enc.getNumVisibleLayers();
    }

    std::vector<int> getHiddenCIs() const {
        initCheck();

        std::vector<int> hiddenCIs(enc.getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = enc.getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getHiddenSize() const {
        initCheck();

        aon::Int3 size = enc.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    std::tuple<int, int, int> getVisibleSize(
        int i
    ) const {
        initCheck();

        aon::Int3 size = enc.getVisibleLayerDesc(i).size;

        return { size.x, size.y, size.z };
    }

    // Params
    void setLR(
        float lr
    ) {
        initCheck();

        if (lr < 0.0f) {
            std::cerr << "Error: Encoder LR must be >= 0.0" << std::endl;
            abort();
        }

        enc.lr = lr;
    }

    float getLR() const {
        initCheck();

        return enc.lr;
    }
};
} // namespace pyaon
