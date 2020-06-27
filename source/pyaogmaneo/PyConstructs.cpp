// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "PyConstructs.h"

using namespace pyaon;

void PyStreamReader::read(void* data, int len) {
    ins.read(static_cast<char*>(data), len);
}

void PyStreamWriter::write(const void* data, int len) {
    outs.write(static_cast<const char*>(data), len);
}