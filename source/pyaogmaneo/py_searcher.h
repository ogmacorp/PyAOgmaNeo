// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "py_helpers.h"
#include <aogmaneo/searcher.h>

namespace py = pybind11;

namespace pyaon {
const int searcher_magic = 381717;

class Searcher {
private:
    aon::Searcher searcher;

    void init_random(
        const std::tuple<int, int, int> &config_size,
        int num_dendrites
    );

    void init_from_file(
        const std::string &file_name
    );

    void init_from_buffer(
        const py::array_t<unsigned char> &buffer
    );

public:
    aon::Searcher::Params params;

    Searcher(
        const std::tuple<int, int, int> &config_size,
        int num_dendrites,
        const std::string &file_name,
        const py::array_t<unsigned char> &buffer
    );

    void save_to_file(
        const std::string &file_name
    );

    void set_state_from_buffer(
        const py::array_t<unsigned char> &buffer
    );

    void set_weights_from_buffer(
        const py::array_t<unsigned char> &buffer
    );

    py::array_t<unsigned char> serialize_to_buffer();

    py::array_t<unsigned char> serialize_state_to_buffer();

    py::array_t<unsigned char> serialize_weights_to_buffer();

    long get_size() const {
        return searcher.size();
    }

    long get_state_size() const {
        return searcher.state_size();
    }

    long get_weights_size() const {
        return searcher.weights_size();
    }

    void step(
        float reward,
        bool learn_enabled
    );

    py::array_t<int> get_config_cis() const;

    std::tuple<int, int, int> get_config_size() const {
        aon::Int3 size = searcher.get_config_size();

        return { size.x, size.y, size.z };
    }

    void merge(
        const std::vector<Searcher*> &searchers,
        Merge_Mode mode
    );
};
}
