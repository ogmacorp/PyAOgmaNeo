// ----------------------------------------------------------------------------
//  py_aogma_neo
//  copyright(c) 2020-2023 ogma intelligent systems corp. all rights reserved.
//
//  this copy of py_aogma_neo is licensed to you under the terms described
//  in the pyaogmaneo_license.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "py_helpers.h"

#include <assert.h>

using namespace pyaon;

void File_Reader::read(void* data, int len) {
    ins.read(static_cast<char*>(data), len);
}

void File_Writer::write(const void* data, int len) {
    outs.write(static_cast<const char*>(data), len);
}

void Buffer_Reader::read(void* data, int len) {
    for (int i = 0; i < len; i++)
        static_cast<unsigned char*>(data)[i] = (*buffer)[start + i];

    start += len;
}

void Buffer_Writer::write(const void* data, int len) {
    assert(buffer.size() >= start + len);

    for (int i = 0; i < len; i++)
        buffer[start + i] = static_cast<const unsigned char*>(data)[i];

    start += len;
}
