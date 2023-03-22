// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <aogmaneo/helpers.h>
#include <tuple>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <exception>

namespace pyaon {
inline void set_num_threads(
    int num_threads
) {
    aon::set_num_threads(num_threads);
}

inline int get_num_threads() {
    return aon::get_num_threads();
}

inline void set_global_state(
    unsigned int state
) {
    aon::global_state = state;
}

inline unsigned int get_global_state() {
    return aon::global_state;
}

class File_Reader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        int len
    ) override;
};

class File_Writer : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        int len
    ) override;
};

class Buffer_Reader : public aon::Stream_Reader {
public:
    int start;
    const std::vector<unsigned char>* buffer;

    Buffer_Reader()
    :
    start(0),
    buffer(nullptr)
    {}

    void read(
        void* data,
        int len
    ) override;
};

class Buffer_Writer : public aon::Stream_Writer {
public:
    int start;
    std::vector<unsigned char> buffer;

    Buffer_Writer(
        int buffer_size
    )
    :
    start(0)
    {
        buffer.resize(buffer_size);
    }

    void write(
        const void* data,
        int len
    ) override;
};
}
