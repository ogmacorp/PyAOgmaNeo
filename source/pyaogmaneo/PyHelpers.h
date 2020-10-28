// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <aogmaneo/Helpers.h>
#include <array>
#include <vector>
#include <fstream>

namespace pyaon {
typedef std::array<int, 3> Arr3i;

class PyStreamReader : public aon::StreamReader {
public:
    std::ifstream ins;

    void read(
        void* data,
        int len
    ) override;
};

class PyStreamWriter : public aon::StreamWriter {
public:
    std::ofstream outs;

    void write(
        const void* data,
        int len
    ) override;
};

class PyBufferReader : public aon::StreamReader {
public:
    int start;
    const std::vector<unsigned char>* buffer;

    PyBufferReader()
    :
    start(0),
    buffer(nullptr)
    {}

    void read(
        void* data,
        int len
    ) override;
};

class PyBufferWriter : public aon::StreamWriter {
public:
    std::vector<unsigned char> buffer;

    void write(
        const void* data,
        int len
    ) override;
};
} // namespace pyaon
