// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyVisualizer.h"

using namespace pyaon;

const Color cellActiveColor = (Color){ 255, 64, 64, 255 };
const Color cellPredictedColor = (Color){ 64, 255, 64, 255 };
const Color cellOffColor = (Color){ 192, 192, 192, 255 };

const float cellRadius = 0.3f;
const float columnRadius = 0.4f;
const float layerDelta = 10.0f;

PyVisualizer::PyVisualizer(
    int width,
    int height,
    const std::string &title
) {
    InitWindow(width, height, title.c_str());

    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 70.0f;
    camera.type = CAMERA_PERSPECTIVE;

    SetCameraMode(camera, CAMERA_ORBITAL);
}

PyVisualizer::~PyVisualizer() {
    CloseWindow();
}

void PyVisualizer::update(
    const PyHierarchy &h
) {
    int oldColumnSize = columns.size();
    int oldCellsSize = cells.size();
    int oldLinesSize = lines.size();

    columns.clear();
    cells.clear();
    lines.clear();

    columns.reserve(oldColumnSize);
    cells.reserve(oldCellsSize);
    lines.reserve(oldLinesSize);

    // Generate necessary geometry

    const aon::Hierarchy &ah = h.h;

    for (int l = 0; l < ah.getNumLayers(); l++) {
        aon::ByteBuffer csdr = ah.getSCLayer(l).getHiddenCs();
        aon::ByteBuffer pcsdr;
        
        if (l < ah.getNumLayers() - 1)
            pcsdr = ah.getPLayers(l + 1)[ah.getTicksPerUpdate(l + 1) - ah.getTicks(l + 1)]->getHiddenCs();

        Vector3 offset = (Vector3){-ah.getSCLayer(l).getHiddenSize().x * 0.5f, -ah.getSCLayer(l).getHiddenSize().y * 0.5f, (l + 1) * layerDelta };

        // Construct columns
        for (int cx = 0; cx < ah.getSCLayer(l).getHiddenSize().x; cx++)
            for (int cy = 0; cy < ah.getSCLayer(l).getHiddenSize().y; cy++) {
                int columnIndex = aon::address2(aon::Int2(cx, cy), aon::Int2(ah.getSCLayer(l).getHiddenSize().x, ah.getSCLayer(l).getHiddenSize().y));

                unsigned char c = csdr[columnIndex];
                
                for (int cz = 0; cz < ah.getSCLayer(l).getHiddenSize().z; cz++)
                    cells.push_back(std::tuple<Vector3, Color>((Vector3){ cx + offset.x, cy + offset.y, cz + offset.z }, cz == c ? cellActiveColor : (pcsdr.size() != 0 && cz == pcsdr[columnIndex] ? cellPredictedColor : cellOffColor)));

                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x, cy + offset.y, offset.z }, (Vector3){ columnRadius * 2.0f, ah.getSCLayer(l).getHiddenSize().z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){0, 0, 0, 255}));
            }
    }
}

void PyVisualizer::render() {
    UpdateCamera(&camera);

    BeginDrawing();

        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

            for (int i = 0; i < cells.size(); i++) {
                DrawSphere(std::get<0>(cells[i]), cellRadius, std::get<1>(cells[i]));
            }

            for (int i = 0; i < columns.size(); i++) {
                DrawCubeWiresV(std::get<0>(columns[i]), std::get<1>(columns[i]), std::get<2>(columns[i]));
            }

        EndMode3D();

        DrawRectangle( 10, 10, 220, 70, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines( 10, 10, 220, 70, BLUE);

        DrawText("Mouse move to look around", 40, 60, 10, DARKGRAY);

    EndDrawing();
}