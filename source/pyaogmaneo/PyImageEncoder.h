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
#include <vector>
#include <array>

namespace pyaon {
struct PyImageEncoderVisibleLayerDesc {
    std::array<int, 3> size;

    int radius;

    PyImageEncoderVisibleLayerDesc(
        std::array<int, 3> size = std::array<int, 3>{ 32, 32, 3 },
        int radius = 4
    )
    : 
    size(size),
    radius(radius)
    {}
};

class PyImageEncoder {
private:
    aon::ImageEncoder enc;

public:
    PyImageEncoder() {}

    void initRandom(
        std::array<int, 3> hiddenSize,
        const std::vector<PyImageEncoderVisibleLayerDesc> &visibleLayerDescs
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
        bool learnEnabled = true
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

    std::array<int, 3> getHiddenSize() const {
        aon::Int3 size = enc.getHiddenSize();

        return { size.x, size.y, size.z };
    }

    std::array<int, 3> getVisibleSize(
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
