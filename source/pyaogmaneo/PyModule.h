// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Hierarchy.h"
#include "ImageEncoder.h"

PYBIND11_MODULE(pyaogmaneo, m) {
    mod_init_hierarchy(m);
    mod_init_imageencoder(m);
}
