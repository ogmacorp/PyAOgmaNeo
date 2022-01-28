// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <aogmaneo/Helpers.h>
#include <tuple>
#include <vector>
#include <fstream>
#include <iostream>

namespace pyaon {
inline void setNumThreads(
    int numThreads
) {
    aon::setNumThreads(numThreads);
}

inline int getNumThreads() {
    return aon::getNumThreads();
}

inline void setGlobalState(
    unsigned int state
) {
    aon::globalState = state;
}

inline unsigned int getGlobalState() {
    return aon::globalState;
}

class FileReader : public aon::StreamReader {
public:
    std::ifstream ins;

    void read(
        void* data,
        int len
    ) override;
};

class FileWriter : public aon::StreamWriter {
public:
    std::ofstream outs;

    void write(
        const void* data,
        int len
    ) override;
};

class BufferReader : public aon::StreamReader {
public:
    int start;
    const std::vector<unsigned char>* buffer;

    BufferReader()
    :
    start(0),
    buffer(nullptr)
    {}

    void read(
        void* data,
        int len
    ) override;
};

class BufferWriter : public aon::StreamWriter {
public:
    int start;
    std::vector<unsigned char> buffer;

    BufferWriter(
        int bufferSize
    )
    :
    start(0)
    {
        buffer.resize(bufferSize);
    }

    void write(
        const void* data,
        int len
    ) override;
};
} // namespace pyaon
