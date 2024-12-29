// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "py_helpers.h"
#include <aogmaneo/hierarchy.h>

namespace py = pybind11;

namespace pyaon {
const int hierarchy_magic = 5523058;

enum IO_Type {
    none = 0,
    prediction = 1,
    action = 2
};

struct IO_Desc {
    std::tuple<int, int, int> size;
    IO_Type type;

    int num_dendrites_per_cell;
    int value_num_dendrites_per_cell;

    int up_radius;
    int down_radius;

    int history_capacity;

    IO_Desc(
        const std::tuple<int, int, int> &size,
        IO_Type type,
        int num_dendrites_per_cell,
        int value_num_dendrites_per_cell,
        int up_radius,
        int down_radius,
        int history_capacity
    )
    :
    size(size),
    type(type),
    num_dendrites_per_cell(num_dendrites_per_cell),
    value_num_dendrites_per_cell(value_num_dendrites_per_cell),
    up_radius(up_radius),
    down_radius(down_radius),
    history_capacity(history_capacity)
    {}

    void check_in_range() const;
};

struct Layer_Desc {
    std::tuple<int, int, int> hidden_size;

    int num_dendrites_per_cell;

    int up_radius;
    int recurrent_radius;
    int down_radius;

    Layer_Desc(
        const std::tuple<int, int, int> &hidden_size,
        int num_dendrites_per_cell,
        int up_radius,
        int recurrent_radius,
        int down_radius
    )
    :
    hidden_size(hidden_size),
    num_dendrites_per_cell(num_dendrites_per_cell),
    up_radius(up_radius),
    recurrent_radius(recurrent_radius),
    down_radius(down_radius)
    {}

    void check_in_range() const;
};

struct Params {
    std::vector<aon::Hierarchy::Layer_Params> layers;
    std::vector<aon::Hierarchy::IO_Params> ios;

    bool anticipation;
};

class Hierarchy {
private:
    aon::Hierarchy h;

    aon::Array<aon::Int_Buffer> c_input_cis_backing;
    aon::Array<aon::Int_Buffer_View> c_input_cis;

    void init_random(
        const std::vector<IO_Desc> &io_descs,
        const std::vector<Layer_Desc> &layer_descs
    );

    void init_from_file(
        const std::string &file_name
    );

    void init_from_buffer(
        const py::array_t<unsigned char> &buffer
    );

    void copy_params_to_h();

public:
    Params params;

    Hierarchy(
        const std::vector<IO_Desc> &io_descs,
        const std::vector<Layer_Desc> &layer_descs,
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
        return h.size();
    }

    long get_state_size() const {
        return h.state_size();
    }

    long get_weights_size() const {
        return h.weights_size();
    }

    void step(
        const std::vector<py::array_t<int, py::array::c_style | py::array::forcecast>> &input_cis,
        bool learn_enabled,
        float reward,
        float mimic
    );

    void clear_state() {
        h.clear_state();
    }

    int get_num_layers() const {
        return h.get_num_layers();
    }

    py::array_t<int> get_prediction_cis(
        int i
    ) const;

    py::array_t<int> get_layer_prediction_cis(
        int l
    ) const;

    py::array_t<float> get_prediction_acts(
        int i
    ) const;

    py::array_t<int> sample_prediction(
        int i,
        float temperature
    ) const;

    py::array_t<int> get_hidden_cis(
        int l
    );

    std::tuple<int, int, int> get_hidden_size(
        int l
    ) {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        aon::Int3 size = h.get_encoder(l).get_hidden_size();

        return { size.x, size.y, size.z };
    }

    int get_num_encoder_visible_layers(
        int l
    ) {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        return h.get_num_encoder_visible_layers(l);
    }

    int get_num_io() const {
        return h.get_num_io();
    }

    std::tuple<int, int, int> get_io_size(
        int i
    ) const {
        if (i < 0 || i >= h.get_num_io())
            throw std::runtime_error("error: " + std::to_string(i) + " is not a valid input index!");

        aon::Int3 size = h.get_io_size(i);

        return { size.x, size.y, size.z };
    }

    IO_Type get_io_type(
        int i
    ) const {
        if (i < 0 || i >= h.get_num_io())
            throw std::runtime_error("error: " + std::to_string(i) + " is not a valid input index!");

        return static_cast<IO_Type>(h.get_io_type(i));
    }

    // retrieve additional parameters on the sph's structure
    int get_up_radius(
        int l
    ) const {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        return h.get_encoder(l).get_visible_layer_desc(0).radius;
    }

    int get_down_radius(
        int l,
        int i
    ) const {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        if (l == 0 && i < 0 || i >= h.get_num_io())
            throw std::runtime_error("error: " + std::to_string(i) + " is not a valid input index!");

        if (h.get_io_type(i) == aon::action)
            return h.get_actor(i).get_visible_layer_desc(0).radius;
        
        return h.get_decoder(l, i).get_visible_layer_desc(0).radius;
    }

    void merge(
        const std::vector<Hierarchy*> &hierarchies,
        Merge_Mode mode
    );
};
}
