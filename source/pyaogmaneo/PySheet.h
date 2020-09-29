// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyConstructs.h"
#include <aogmaneo/Sheet.h>
#include <vector>
#include <fstream>

namespace pyaon {
inline void setNumThreads(int numThreads) {
    aon::setNumThreads(numThreads);
}

inline int getNumThreads() {
    return aon::getNumThreads();
}

struct PySheetInputDesc {
    PyInt3 size;

    int radius;

    bool recurrent;

    PySheetInputDesc()
    :
    size(4, 4, 16),
    radius(2),
    recurrent(false)
    {}

    PySheetInputDesc(
        const PyInt3 &size,
        int radius,
        bool recurrent
    )
    :
    size(size),
    radius(radius),
    recurrent(recurrent)
    {}
};

struct PySheetOutputDesc {
    PyInt3 size;

    int radius;

    PySheetOutputDesc()
    :
    size(4, 4, 16),
    radius(2)
    {}

    PySheetOutputDesc(
        const PyInt3 &size,
        int radius
    )
    :
    size(size),
    radius(radius)
    {}
};

class PySheet {
private:
    aon::Sheet s;

public:
    PySheet(
        const std::vector<PySheetInputDesc> &inputDescs,
        int recurrentRadius,
        const std::vector<PySheetOutputDesc> &outputDescs,
        const PyInt3 &actorSize
    );

    PySheet(
        const std::string &name
    );

    PySheet(
        const std::vector<unsigned char> &buffer
    );

    void save(
        const std::string &name
    );

    std::vector<unsigned char> save();

    void step(
        const std::vector<std::vector<int> > &inputCs,
        const std::vector<std::vector<int> > &targetCs,
        int subSteps,
        bool learnEnabled = true
    );

    std::vector<int> getPredictionCs(
        int i
    ) const {
        std::vector<int> predictions(s.getPredictionCs(i).size());

        for (int j = 0; j < predictions.size(); j++)
            predictions[j] = s.getPredictionCs(i)[j];

        return predictions;
    }
};
} // namespace pyaon
