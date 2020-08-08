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

#define RAYGUI_SUPPORT_ICONS

#include "../raygui/raygui.h"

namespace pyaon {
class PyVisualizer {
private:
    int winWidth;
    int winHeight;

    Camera camera;

    // Storing generated model
    std::vector<std::tuple<Vector3, Vector3, Color>> columns;
    std::vector<std::tuple<Vector3, Color>> cells;
    std::vector<std::tuple<Vector3, Vector3, Color>> lines;

    // Images being encoded (optional)
    Texture2D imEncTexture;
    bool hasImEncImg;
    Model imEncPlane;

    // Selection
    int selectLayer;
    int selectInput;
    int selectX;
    int selectY;
    int selectZ;

    int selectLayerPrev;
    int selectInputPrev;
    int selectXPrev;
    int selectYPrev;
    int selectZPrev;

    bool showTextures;
    bool refreshTextures;

    // FF
    int ffVli;
    int ffVliRange;
    int ffZ;
    int ffZRange;

    Texture2D ffTexture;
    int ffWidth;
    int ffHeight;

    // FB

public:
    PyVisualizer(
        int winWidth,
        int winHeight,
        const std::string &title
    );

    ~PyVisualizer();

    void update(
        const PyHierarchy &h
    ) {
        update(h, {}, -1, 0, 0, true);
    }

    void update(
        const PyHierarchy &h,
        const std::vector<unsigned char> &imEncImg,
        int imEncIndex,
        int imWidth,
        int imHeight,
        bool grayscale
    );

    void render();
};
} // namespace pyaon