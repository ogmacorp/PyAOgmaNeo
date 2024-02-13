// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

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
    unsigned long state
) {
    aon::global_state = state;
}

inline unsigned long get_global_state() {
    return aon::global_state;
}

class File_Reader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        long len
    ) override;
};

class File_Writer : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        long len
    ) override;
};

class Buffer_Reader : public aon::Stream_Reader {
public:
    int start;
    const py::array_t<unsigned char>* buffer;

    Buffer_Reader()
    :
    start(0),
    buffer(nullptr)
    {}

    void read(
        void* data,
        long len
    ) override;
};

class Buffer_Writer : public aon::Stream_Writer {
public:
    long start;
    py::array_t<unsigned char> buffer;

    Buffer_Writer(
        long buffer_size
    )
    :
    start(0),
    buffer(buffer_size)
    {}

    void write(
        const void* data,
        long len
    ) override;
};

enum Merge_Mode {
    merge_random = 0,
    merge_average = 1
};
}
