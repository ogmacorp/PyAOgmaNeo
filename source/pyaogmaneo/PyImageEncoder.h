// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/ImageEncoder.h>

namespace pyaon {
const int imageEncoderMagic = 128833;

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

struct ImageEncoderHigherLayerDesc {
    std::tuple<int, int, int> hiddenSize;

    int radius;

    ImageEncoderHigherLayerDesc(
        const std::tuple<int, int, int> &hiddenSize,
        int radius
    )
    : 
    hiddenSize(hiddenSize),
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
        const std::vector<ImageEncoderVisibleLayerDesc> &visibleLayerDescs,
        const std::vector<ImageEncoderHigherLayerDesc> &higherLayerDescs
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

    int getNumHigherLayers() const {
        initCheck();

        return enc.getNumHigherLayers();
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

    std::vector<int> getOutputCIs() const {
        initCheck();

        std::vector<int> outputCIs(enc.getOutputCIs().size());

        for (int j = 0; j < outputCIs.size(); j++)
            outputCIs[j] = enc.getOutputCIs()[j];

        return outputCIs;
    }

    std::tuple<int, int, int> getHiddenSize() const {
        initCheck();

        aon::Int3 size = enc.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    std::tuple<int, int, int> getOutputSize() const {
        initCheck();

        aon::Int3 size = enc.getOutputSize();

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
            std::cerr << "Error: ImageEncoder LR must be >= 0.0" << std::endl;
            abort();
        }

        enc.lr = lr;
    }

    float getLR() const {
        initCheck();

        return enc.lr;
    }

    void setFalloff(
        float falloff
    ) {
        initCheck();

        if (falloff < 0.0f) {
            std::cerr << "Error: ImageEncoder falloff must be >= 0.0" << std::endl;
            abort();
        }

        enc.falloff = falloff;
    }

    float getFalloff() const {
        initCheck();

        return enc.falloff;
    }

    void setHigherLR(
        int l,
        float lr
    ) {
        initCheck();

        if (l < 0 || l > enc.getNumHigherLayers()) {
            std::cerr << "Error: " << l << " is not a valid higher layer index!" << std::endl;
            abort();
        }

        if (lr < 0.0f) {
            std::cerr << "Error: ImageEncoder higher LR must be >= 0.0" << std::endl;
            abort();
        }

        enc.getHigherLayer(l).lr = lr;
    }

    float getHigherLR(
        int l
    ) const {
        initCheck();

        if (l < 0 || l > enc.getNumHigherLayers()) {
            std::cerr << "Error: " << l << " is not a valid higher layer index!" << std::endl;
            abort();
        }

        return enc.getHigherLayer(l).lr;
    }
};
} // namespace pyaon
