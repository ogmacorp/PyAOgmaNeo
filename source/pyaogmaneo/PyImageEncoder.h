// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/ImageEncoder.h>

namespace pyaon {
const int imageEncoderMagic = 128847;

struct ImageEncoderVisibleLayerDesc {
    std::tuple<int, int, int> size;

    int radius;

    ImageEncoderVisibleLayerDesc(
        const std::tuple<int, int, int> &size,
        int radius
    )
    : 
    size(size),
    radius(radius)
    {}

    bool checkInRange() const;
};

class ImageEncoder {
private:
    bool initialized;

    void initCheck() const;

    aon::ImageEncoder enc;

public:
    ImageEncoder() 
    :
    initialized(false)
    {}

    void initRandom(
        const std::tuple<int, int, int> &hiddenSize,
        const std::vector<ImageEncoderVisibleLayerDesc> &visibleLayerDescs
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
        const std::vector<std::vector<unsigned char>> &inputs,
        bool learnEnabled
    );

    void reconstruct(
        const std::vector<int> &reconCIs
    );

    int getNumVisibleLayers() const {
        initCheck();

        return enc.getNumVisibleLayers();
    }

    std::vector<unsigned char> getReconstruction(
        int i
    ) const {
        initCheck();

        if (i < 0 || i >= enc.getNumVisibleLayers()) {
            std::cerr << "Cannot get reconstruction at index " << i << " - out of bounds [0, " << enc.getNumVisibleLayers() << "]" << std::endl;
            abort();
        }

        std::vector<unsigned char> reconstruction(enc.getReconstruction(i).size());

        for (int j = 0; j < reconstruction.size(); j++)
            reconstruction[j] = enc.getReconstruction(i)[j];

        return reconstruction;
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

    void setLR(
        float lr
    ) {
        initCheck();

        if (lr < 0.0f) {
            std::cerr << "Error: ImageEncoder LR must be >= 0.0" << std::endl;
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
