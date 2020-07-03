// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyVisualizer.h"

#include <unordered_map>

using namespace pyaon;

const Color cellActiveColor = (Color){ 255, 64, 64, 255 };
const Color cellPredictedColor = (Color){ 64, 255, 64, 255 };
const Color cellOffColor = (Color){ 192, 192, 192, 255 };

const float cellRadius = 0.25f;
const float columnRadius = 0.3f;
const float layerDelta = 6.0f;
const float weightScaling = 1.0f;

PyVisualizer::PyVisualizer(
    int width,
    int height,
    const std::string &title
) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);

    InitWindow(width, height, title.c_str());

    camera.position = (Vector3){ 10.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 70.0f;
    camera.type = CAMERA_PERSPECTIVE;

    SetCameraMode(camera, CAMERA_FREE);

    SetCameraAltControl(KEY_LEFT_SHIFT);
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

    if (oldColumnSize > 0)
        columns.reserve(oldColumnSize);

    if (oldCellsSize > 0)
        cells.reserve(oldCellsSize);

    if (oldLinesSize > 0)
        lines.reserve(oldLinesSize);

    // Generate necessary geometry
    const aon::Hierarchy &ah = h.h;

    // Calculate full size
    float hierarchyHeight = 0.0f;

    for (int l = 0; l < ah.getNumLayers(); l++)
        hierarchyHeight += (l < ah.getNumLayers() - 1 ? layerDelta : 0) + ah.getSCLayer(l).getHiddenSize().z;

    // Find total input layer width
    float inputWidthTotal = 0.0f;
    float maxInputHeight = 0.0f;

    for (int i = 0; i < ah.getInputSizes().size(); i++) {
        inputWidthTotal += (i < ah.getInputSizes().size() - 1 ? layerDelta : 0) + ah.getInputSizes()[i].x;

        maxInputHeight = std::max<float>(maxInputHeight, ah.getInputSizes()[i].z);
    }

    float zOffset = -hierarchyHeight * 0.5f;

    // Render input layers
    float xOffset = -inputWidthTotal * 0.5f;

    for (int i = 0; i < ah.getInputSizes().size(); i++) {
        const aon::CircleBuffer<aon::ByteBuffer> &hist = ah.getHistories(0)[i];

        aon::ByteBuffer csdr = hist[0];
        aon::ByteBuffer pcsdr;
        
        if (ah.getPLayers(0)[i] != nullptr || ah.getALayers()[i] != nullptr)
            pcsdr = ah.getPredictionCs(i);
        
        Vector3 offset = (Vector3){ -ah.getInputSizes()[i].x * 0.5f + xOffset, -ah.getInputSizes()[i].y * 0.5f, -ah.getInputSizes()[i].z * 0.5f + zOffset - layerDelta - maxInputHeight * 0.5f};

        // Construct columns
        for (int cx = 0; cx < ah.getInputSizes()[i].x; cx++)
            for (int cy = 0; cy < ah.getInputSizes()[i].y; cy++) {
                int columnIndex = aon::address2(aon::Int2(cx, cy), aon::Int2(ah.getInputSizes()[i].x, ah.getInputSizes()[i].y));

                unsigned char c = csdr[columnIndex];
                
                for (int cz = 0; cz < ah.getInputSizes()[i].z; cz++)
                    cells.push_back(std::tuple<Vector3, Color>((Vector3){ cx + offset.x + 0.5f, cz + offset.z, cy + offset.y + 0.5f }, cz == c ? cellActiveColor : (pcsdr.size() != 0 && cz == pcsdr[columnIndex] ? cellPredictedColor : cellOffColor)));

                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x + 0.5f, offset.z + ah.getInputSizes()[i].z * 0.5f - columnRadius, cy + offset.y + 0.5f }, (Vector3){ columnRadius * 2.0f, ah.getInputSizes()[i].z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){0, 0, 0, 255}));
            }

        // Line to next layer
        lines.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ xOffset, zOffset - layerDelta - (maxInputHeight - ah.getInputSizes()[i].z) * 0.5f, 0.0f }, (Vector3){ 0.0f, zOffset, 0.0f }, (Color){ 0, 0, 0, 128 }));

        xOffset += layerDelta + ah.getInputSizes()[i].x;
    }

    for (int l = 0; l < ah.getNumLayers(); l++) {
        aon::ByteBuffer csdr = ah.getSCLayer(l).getHiddenCs();
        aon::ByteBuffer pcsdr;
        
        if (l < ah.getNumLayers() - 1)
            pcsdr = ah.getPLayers(l + 1)[ah.getTicksPerUpdate(l + 1) - 1 - ah.getTicks(l + 1)]->getHiddenCs();

        Vector3 offset = (Vector3){ -ah.getSCLayer(l).getHiddenSize().x * 0.5f, -ah.getSCLayer(l).getHiddenSize().y * 0.5f, zOffset };

        // Construct columns
        for (int cx = 0; cx < ah.getSCLayer(l).getHiddenSize().x; cx++)
            for (int cy = 0; cy < ah.getSCLayer(l).getHiddenSize().y; cy++) {
                int columnIndex = aon::address2(aon::Int2(cx, cy), aon::Int2(ah.getSCLayer(l).getHiddenSize().x, ah.getSCLayer(l).getHiddenSize().y));

                unsigned char c = csdr[columnIndex];
                
                for (int cz = 0; cz < ah.getSCLayer(l).getHiddenSize().z; cz++)
                    cells.push_back(std::tuple<Vector3, Color>((Vector3){ cx + offset.x + 0.5f, cz + offset.z, cy + offset.y + 0.5f }, cz == c ? cellActiveColor : (pcsdr.size() != 0 && cz == pcsdr[columnIndex] ? cellPredictedColor : cellOffColor)));

                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x + 0.5f, offset.z + ah.getSCLayer(l).getHiddenSize().z * 0.5f - columnRadius, cy + offset.y + 0.5f }, (Vector3){ columnRadius * 2.0f, ah.getSCLayer(l).getHiddenSize().z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){0, 0, 0, 255}));
            }

        if (l < ah.getNumLayers() - 1)
            lines.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ 0.0f, zOffset + ah.getSCLayer(l).getHiddenSize().z, 0.0f }, (Vector3){ 0.0f, zOffset + ah.getSCLayer(l).getHiddenSize().z + layerDelta, 0.0f }, (Color){ 0, 0, 0, 128 }));

        zOffset += layerDelta + ah.getSCLayer(l).getHiddenSize().z;
    }
}

void PyVisualizer::render() {
    UpdateCamera(&camera);

    BeginDrawing();

        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

            for (int i = 0; i < cells.size(); i++)
                DrawSphere(std::get<0>(cells[i]), cellRadius, std::get<1>(cells[i]));

            for (int i = 0; i < columns.size(); i++)
                DrawCubeWiresV(std::get<0>(columns[i]), std::get<1>(columns[i]), std::get<2>(columns[i]));

            for (int i = 0; i < lines.size(); i++)
                DrawLine3D(std::get<0>(lines[i]), std::get<1>(lines[i]), std::get<2>(lines[i]));

        EndMode3D();

        DrawRectangle( 10, 10, 220, 50, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines( 10, 10, 220, 50, BLUE);

        DrawText("Mouse move to look around", 20, 20, 10, DARKGRAY);

    EndDrawing();
}