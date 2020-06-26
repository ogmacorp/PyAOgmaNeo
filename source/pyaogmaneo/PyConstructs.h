// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <aogmaneo/Helpers.h>
#include <fstream>

namespace pyaon {
class PyInt3 {
private:
public:
    int x, y, z;

    PyInt3() 
    :
    x(0),
    y(0),
    z(0)
    {}

    PyInt3(
        int x,
        int y,
        int z
    )
    :
    x(x),
    y(y),
    z(z)
    {}
};

class PyStreamReader : public aon::StreamReader {
public:
    std::ifstream is;

    void read(void* data, int len) override {
        is.read(static_cast<char*>(data), len);
    }
};

class PyStreamWriter : public aon::StreamWriter {
public:
    std::ofstream os;

    void write(const void* data, int len) override {
        os.write(static_cast<const char*>(data), len);
    }
};
} // namespace pyaon