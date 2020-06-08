// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

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
} // namespace pyaon