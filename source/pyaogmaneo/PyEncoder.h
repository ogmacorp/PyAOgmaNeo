// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/Encoder.h>

namespace pyaon {
const int encoderMagic = 953342;

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

    void checkInRange() const;
};

class Encoder {
private:
    aon::Encoder enc;

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

public:
    Encoder(
        const std::tuple<int, int, int> &hiddenSize,
        const std::vector<EncoderVisibleLayerDesc> &visibleLayerDescs,
        const std::string &name,
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

    std::vector<int> reconstruct(
        const std::vector<int> &hiddenCIs,
        int vli
    );

    int getNumVisibleLayers() const {
        return enc.getNumVisibleLayers();
    }

    std::vector<int> getHiddenCIs() const {
        std::vector<int> hiddenCIs(enc.getHiddenCIs().size());

        for (int j = 0; j < hiddenCIs.size(); j++)
            hiddenCIs[j] = enc.getHiddenCIs()[j];

        return hiddenCIs;
    }

    std::tuple<int, int, int> getHiddenSize() const {
        aon::Int3 size = enc.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    std::tuple<int, int, int> getVisibleSize(
        int i
    ) const {
        aon::Int3 size = enc.getVisibleLayerDesc(i).size;

        return { size.x, size.y, size.z };
    }

    void setLR(
        float lr
    ) {
        if (lr < 0.0f)
            throw std::runtime_error("Error: Encoder LR must be >= 0.0");

        enc.lr = lr;
    }

    float getLR() const {
        return enc.lr;
    }
};
} // namespace pyaon
