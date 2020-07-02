// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "PyHierarchy.h"

#include <tuple>

#include <raylib.h>

namespace pyaon {
class PyVisualizer {
private:
    Camera camera;

    // Storing generated model
    std::vector<std::tuple<Vector3, Vector3, Color>> columns;
    std::vector<std::tuple<Vector3, Color>> cells;
    std::vector<std::tuple<Vector3, Vector3, Color>> lines;

public:
    PyVisualizer(
        int width,
        int height,
        const std::string &title
    );

    ~PyVisualizer();

    void update(
        const PyHierarchy &h
    );

    void render();
};
} // namespace pyaon