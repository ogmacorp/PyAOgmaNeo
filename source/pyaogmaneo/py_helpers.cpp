// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "py_helpers.h"

#include <assert.h>

using namespace pyaon;

void File_Reader::read(void* data, long len) {
    ins.read(static_cast<char*>(data), len);
}

void File_Writer::write(const void* data, long len) {
    outs.write(static_cast<const char*>(data), len);
}

void Buffer_Reader::read(void* data, long len) {
    auto view = buffer->unchecked();

    for (long i = 0; i < len; i++)
        static_cast<unsigned char*>(data)[i] = view(start + i);

    start += len;
}

void Buffer_Writer::write(const void* data, long len) {
    assert(buffer.size() >= start + len);

    auto view = buffer.mutable_unchecked();

    for (long i = 0; i < len; i++)
        view[start + i] = static_cast<const unsigned char*>(data)[i];

    start += len;
}
