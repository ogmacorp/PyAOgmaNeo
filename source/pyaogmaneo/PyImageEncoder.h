// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyConstructs.h"
#include <aogmaneo/ImageEncoder.h>
#include <vector>

namespace pyaon {
struct PyImageEncoderVisibleLayerDesc {
    PyInt3 size;

    int radius;

    PyImageEncoderVisibleLayerDesc()
    :
    size(8, 8, 16),
    radius(2)
    {}

    PyImageEncoderVisibleLayerDesc(
        const PyInt3 &size,
        int radius)
    : 
    size(size),
    radius(radius)
    {}
};

class PyImageEncoder {
private:
    aon::ImageEncoder enc;

public:
    float alpha;
    float gamma;
    
    PyImageEncoder(
        const PyInt3 &hiddenSize,
        const std::vector<PyImageEncoderVisibleLayerDesc> &visibleLayerDescs
    );

    PyImageEncoder(
        const std::string &name
    );

    void save(
        const std::string &name
    );

    void step(
        const std::vector<std::vector<float> > &inputs,
        bool learnEnabled = true
    );

    void reconstruct(
        const std::vector<int> &reconCs
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

    std::vector<int> getHiddenCs() const {
        std::vector<int> hiddenCs(enc.getHiddenCs().size());

        for (int j = 0; j < hiddenCs.size(); j++)
            hiddenCs[j] = enc.getHiddenCs()[j];

        return hiddenCs;
    }

    PyInt3 getHiddenSize() const {
        aon::Int3 size = enc.getHiddenSize();

        return PyInt3(size.x, size.y, size.z);
    }

    PyInt3 getVisibleSize(
        int i
    ) const {
        aon::Int3 size = enc.getVisibleLayerDesc(i).size;

        return PyInt3(size.x, size.y, size.z);
    }
};
} // namespace pyaon
