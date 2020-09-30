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

    PySheetInputDesc()
    :
    size(4, 4, 16),
    radius(2)
    {}

    PySheetInputDesc(
        const PyInt3 &size,
        int radius
    )
    :
    size(size),
    radius(radius)
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

    std::vector<std::vector<int> > step(
        const std::vector<std::vector<int> > &inputCs,
        const std::vector<std::vector<int> > &targetCs,
        int subSteps,
        bool learnEnabled = true
    );

    std::vector<int> getActorHiddenCs() const {
        std::vector<int> actorHiddenCs(s.actor.getHiddenCs().size());

        for (int j = 0; j < actorHiddenCs.size(); j++)
            actorHiddenCs[j] = s.actor.getHiddenCs()[j];

        return actorHiddenCs;
    }

    std::vector<int> getPredictionCs(
        int i
    ) const {
        std::vector<int> predictions(s.getPredictionCs(i).size());

        for (int j = 0; j < predictions.size(); j++)
            predictions[j] = s.getPredictionCs(i)[j];

        return predictions;
    }

    void setATemperature(
        float value
    ) {
        s.actor.temperature = value;
    }

    float getATemperature() const {
        return s.actor.temperature;
    }

    void setAAlpha(
        float value
    ) {
        s.actor.alpha = value;
    }

    float getAAlpha() const {
        return s.actor.alpha;
    }

    void setAGamma(
        float value
    ) {
        s.actor.gamma = value;
    }

    float getAGamma() const {
        return s.actor.gamma;
    }

    void setATraceDecay(
        float value
    ) {
        s.actor.traceDecay = value;
    }

    float getATraceDecay() const {
        return s.actor.traceDecay;
    }

    void setPAlpha(
        int i,
        float value
    ) {
        s.predictors[i].alpha = value;
    }

    float getPAlpha(
        int i
    ) const {
        return s.predictors[i].alpha;
    }
};
} // namespace pyaon
