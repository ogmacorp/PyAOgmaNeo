// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHelpers.h"
#include <aogmaneo/ImageEncoder.h>

namespace pyaon {
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
};

class ImageEncoder {
private:
    aon::ImageEncoder enc;

public:
    ImageEncoder() {}

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
        const std::vector<std::vector<float> > &inputs,
        bool learnEnabled
    );

    void reconstruct(
        const std::vector<int> &reconCIs
    );

    int getNumVisibleLayers() const {
        return enc.getNumVisibleLayers();
    }

    std::vector<float> getReconstruction(
        int i
    ) const {
        std::vector<float> reconstruction(enc.getReconstruction(i).size());

        for (int j = 0; j < reconstruction.size(); j++)
            reconstruction[j] = enc.getReconstruction(i)[j];

        return reconstruction;
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

    // Params
    void setAlpha(
        float alpha
    ) {
        enc.alpha = alpha;
    }

    float getAlpha() const {
        return enc.alpha;
    }

    void setGamma(
        float gamma
    ) {
        enc.gamma = gamma;
    }

    float getGamma() const {
        return enc.gamma;
    }
};
} // namespace pyaon
