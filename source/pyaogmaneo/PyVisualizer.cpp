// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyVisualizer.h"

#define RAYGUI_IMPLEMENTATION

#include "../raygui/raygui.h"

#undef RAYGUI_IMPLEMENTATION

#include <unordered_map>
#include <iostream>

using namespace pyaon;

const Color cellActiveColor = (Color){ 255, 64, 64, 255 };
const Color cellPredictedColor = (Color){ 64, 255, 64, 255 };
const Color cellOffColor = (Color){ 192, 192, 192, 255 };
const Color cellSelectColor = (Color){ 64, 64, 255, 255 };

const float cellRadius = 0.25f;
const float columnRadius = 0.3f;
const float layerDelta = 6.0f;
const float weightScaling = 1.0f;
const float textureScaling = 8.0f;

PyVisualizer::PyVisualizer(
    int winWidth,
    int winHeight,
    const std::string &title
) {
    this->winWidth = winWidth;
    this->winHeight = winHeight;

    SetConfigFlags(FLAG_MSAA_4X_HINT);

    InitWindow(winWidth, winHeight, title.c_str());

    camera.position = (Vector3){ 20.0f, 20.0f, 20.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 70.0f;
    camera.type = CAMERA_PERSPECTIVE;

    SetCameraMode(camera, CAMERA_FREE);

    SetCameraAltControl(KEY_LEFT_SHIFT);

    SetTargetFPS(0);

    selectLayer = -1;
    selectInput = -1;
    selectX = -1;
    selectY = -1;
    selectZ = -1;

    selectLayerPrev = selectLayer;
    selectInputPrev = selectInput;
    selectXPrev = selectX;
    selectYPrev = selectY;
    selectZPrev = selectZ;

    showTextures = false;
    refreshTextures = false;

    ffVli = 0;
    ffVliRange = 0;
    ffZ = 0;
    ffZRange = 0;
}

PyVisualizer::~PyVisualizer() {
    if (showTextures) {
        UnloadTexture(ffTexture);
    }

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

    bool select = IsMouseButtonPressed(MOUSE_RIGHT_BUTTON);

    Ray ray = { 0 };

    float minDistance = -1.0f;
    Vector3 minPosition = (Vector3){ 0.0f, 0.0f, 0.0f };

    if (select) {
        ray = GetMouseRay(GetMousePosition(), camera);

        selectLayer = -1;
        selectInput = -1;
        selectX = -1;
        selectY = -1;
        selectZ = -1;
    }

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
                
                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x + 0.5f, offset.z + ah.getInputSizes()[i].z * 0.5f - columnRadius, cy + offset.y + 0.5f }, (Vector3){ columnRadius * 2.0f, ah.getInputSizes()[i].z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){0, 0, 0, 255}));
                
                Vector3 lowerBound = (Vector3){ std::get<0>(columns.back()).x - std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y - std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z - std::get<1>(columns.back()).z * 0.5f };
                Vector3 upperBound = (Vector3){ std::get<0>(columns.back()).x + std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y + std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z + std::get<1>(columns.back()).z * 0.5f };
                
                bool columnCollision = select ? CheckCollisionRayBox(ray, (BoundingBox){ lowerBound, upperBound }) : false;
                
                for (int cz = 0; cz < ah.getInputSizes()[i].z; cz++) {
                    Vector3 position = (Vector3){ cx + offset.x + 0.5f, cz + offset.z, cy + offset.y + 0.5f };

                    bool cellCollision = columnCollision ? CheckCollisionRaySphere(ray, position, cellRadius) : false;

                    if (cellCollision) {
                        // If already found one, compare distance
                        if (selectX != -1 && minDistance > 0.0f) {
                            float dx = position.x - minPosition.x;
                            float dy = position.y - minPosition.y;
                            float dz = position.z - minPosition.z;

                            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                            if (dist < minDistance) {
                                minDistance = dist;

                                selectLayer = -1;
                                selectInput = i;
                                selectX = cx;
                                selectY = cy;
                                selectZ = cz;
                            }
                        }
                        else {
                            selectLayer = -1;
                            selectInput = i;
                            selectX = cx;
                            selectY = cy;
                            selectZ = cz;
                        }
                    }

                    bool isSelected = cellCollision || (selectLayer == -1 && selectInput == i && selectX == cx && selectY == cy && selectZ == cz);

                    cells.push_back(std::tuple<Vector3, Color>(position, (isSelected ? cellSelectColor : (cz == c ? cellActiveColor : (pcsdr.size() != 0 && cz == pcsdr[columnIndex] ? cellPredictedColor : cellOffColor)))));
                }
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

                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x + 0.5f, offset.z + ah.getSCLayer(l).getHiddenSize().z * 0.5f - columnRadius, cy + offset.y + 0.5f }, (Vector3){ columnRadius * 2.0f, ah.getSCLayer(l).getHiddenSize().z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){0, 0, 0, 255}));
                
                Vector3 lowerBound = (Vector3){ std::get<0>(columns.back()).x - std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y - std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z - std::get<1>(columns.back()).z * 0.5f };
                Vector3 upperBound = (Vector3){ std::get<0>(columns.back()).x + std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y + std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z + std::get<1>(columns.back()).z * 0.5f };
                
                bool columnCollision = select ? CheckCollisionRayBox(ray, (BoundingBox){ lowerBound, upperBound }) : false;
                
                for (int cz = 0; cz < ah.getSCLayer(l).getHiddenSize().z; cz++) {
                    Vector3 position = (Vector3){ cx + offset.x + 0.5f, cz + offset.z, cy + offset.y + 0.5f };

                    bool cellCollision = columnCollision ? CheckCollisionRaySphere(ray, position, cellRadius) : false;

                    if (cellCollision) {
                        // If already found one, compare distance
                        if (selectX != -1 && minDistance > 0.0f) {
                            float dx = position.x - minPosition.x;
                            float dy = position.y - minPosition.y;
                            float dz = position.z - minPosition.z;

                            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                            if (dist < minDistance) {
                                minDistance = dist;
                                
                                selectLayer = l;
                                selectInput = 0;
                                selectX = cx;
                                selectY = cy;
                                selectZ = cz;
                            }
                        }
                        else {
                            selectLayer = l;
                            selectInput = 0;
                            selectX = cx;
                            selectY = cy;
                            selectZ = cz;
                        }
                    }

                    bool isSelected = cellCollision || (selectLayer == l && selectInput == 0 && selectX == cx && selectY == cy && selectZ == cz);

                    cells.push_back(std::tuple<Vector3, Color>(position, (isSelected ? cellSelectColor : (cz == c ? cellActiveColor : (pcsdr.size() != 0 && cz == pcsdr[columnIndex] ? cellPredictedColor : cellOffColor)))));
                }
            }

        if (l < ah.getNumLayers() - 1)
            lines.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ 0.0f, zOffset + ah.getSCLayer(l).getHiddenSize().z, 0.0f }, (Vector3){ 0.0f, zOffset + ah.getSCLayer(l).getHiddenSize().z + layerDelta, 0.0f }, (Color){ 0, 0, 0, 128 }));

        zOffset += layerDelta + ah.getSCLayer(l).getHiddenSize().z;
    }

    // Display active cell receptive fields
    bool changed = selectLayer != selectLayerPrev || selectInput != selectInputPrev || selectX != selectXPrev || selectY != selectYPrev || selectZ != selectZPrev;

    if (selectX != -1 && changed)
        refreshTextures = true;
    else if (selectX == -1)
        showTextures = false;

    if (refreshTextures) {
        if (showTextures) {
            // Unload old
            UnloadTexture(ffTexture);
        }

        showTextures = true;

        // FF
        if (selectLayer >= 0) {
            ffVliRange = ah.getSCLayer(selectLayer).getNumVisibleLayers();

            // Clamp
            ffVli = aon::min(ffVli, ffVliRange - 1);

            const aon::SparseCoder::VisibleLayer &vl = ah.getSCLayer(selectLayer).getVisibleLayer(ffVli);
            const aon::SparseCoder::VisibleLayerDesc &vld = ah.getSCLayer(selectLayer).getVisibleLayerDesc(ffVli);

            aon::Int3 hiddenSize = ah.getSCLayer(selectLayer).getHiddenSize();
            int hiddenIndex = aon::address3(aon::Int3(selectX, selectY, selectZ), hiddenSize);

            ffZRange = vld.size.z;

            // Clamp
            ffZ = aon::min(ffZ, ffZRange - 1);

            int diam = vld.radius * 2 + 1;

            // Projection
            aon::Float2 hToV = aon::Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

            aon::Int2 visibleCenter = project(aon::Int2(selectX, selectY), hToV);

            // Lower corner
            aon::Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

            // Bounds of receptive field, clamped to input size
            aon::Int2 iterLowerBound(aon::max(0, fieldLowerBound.x), aon::max(0, fieldLowerBound.y));
            aon::Int2 iterUpperBound(aon::min(vld.size.x - 1, visibleCenter.x + vld.radius), aon::min(vld.size.y - 1, visibleCenter.y + vld.radius));

            int width = diam;
            int height = diam;

            aon::Array<Color> colors(width * height, (Color){ 0, 0, 0, 255 });

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = aon::address2(aon::Int2(ix, iy), aon::Int2(vld.size.x,  vld.size.y));

                    aon::Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    float weight = vl.weights[ffZ + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];

                    // Rescale
                    unsigned char c = aon::min(1.0f, aon::expf(weight * weightScaling)) * 255;

                    colors[offset.y + offset.x * diam] = (Color){ c, c, c, 255 };
                }

            // Load image
            Image image = LoadImageEx(&colors[0], width, height);

            // Load texture
            ffTexture = LoadTextureFromImage(image);

            ffWidth = width;
            ffHeight = height;

            // Unload image
            UnloadImage(image);
        }
    }

    selectLayerPrev = selectLayer;
    selectInputPrev = selectInput;
    selectXPrev = selectX;
    selectYPrev = selectY;
    selectZPrev = selectZ;
}

void PyVisualizer::render() {
    UpdateCamera(&camera);

    BeginDrawing();

        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

            for (int i = 0; i < cells.size(); i++)
                DrawSphereEx(std::get<0>(cells[i]), cellRadius, 6, 6, std::get<1>(cells[i]));

            for (int i = 0; i < columns.size(); i++)
                DrawCubeWiresV(std::get<0>(columns[i]), std::get<1>(columns[i]), std::get<2>(columns[i]));

            for (int i = 0; i < lines.size(); i++)
                DrawLine3D(std::get<0>(lines[i]), std::get<1>(lines[i]), std::get<2>(lines[i]));

        EndMode3D();

        DrawRectangle( 10, 10, 290, 60, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines( 10, 10, 290, 60, BLUE);

        DrawText("Middle mouse button + move mouse -> pan", 20, 20, 8, DARKGRAY);
        DrawText("Shift + middle mouse button + move mouse -> rotate", 20, 30, 8, DARKGRAY);
        DrawText("Scroll wheel -> zoom", 20, 40, 8, DARKGRAY);
        DrawText("Right click on cell -> select", 20, 50, 8, DARKGRAY);

        GuiEnable();

        if (showTextures) {
            DrawTextureEx(ffTexture, (Vector2){ 10, winHeight - 80 - ffHeight * textureScaling}, 0.0f, textureScaling, (Color){ 255, 255, 255, 255 });

            int oldffVli = ffVli;
            int oldffZ = ffZ;

            ffVli = GuiSlider((Rectangle){ 20, winHeight - 30, 200, 20 }, "Vli", TextFormat("%i", ffVli), ffVli, 0, ffVliRange);
            ffZ = GuiSlider((Rectangle){ 20, winHeight - 60, 200, 20 }, "Z", TextFormat("%i", ffZ), ffZ, 0, ffZRange);

            refreshTextures = ffVli != oldffVli || ffZ != oldffZ;
        }

    EndDrawing();
}