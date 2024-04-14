// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "py_helpers.h"
#include <aogmaneo/image_encoder.h>

namespace py = pybind11;

namespace pyaon {
const int image_encoder_magic = 4153116;

struct Image_Visible_Layer_Desc {
    std::tuple<int, int, int> size;

    int radius;

    Image_Visible_Layer_Desc(
        const std::tuple<int, int, int> &size,
        int radius
    )
    : 
    size(size),
    radius(radius)
    {}

    void check_in_range() const;
};

class Image_Encoder {
private:
    aon::Image_Encoder enc;

    aon::Array<aon::Byte_Buffer> c_inputs_backing;
    aon::Array<aon::Byte_Buffer_View> c_inputs;

    void init_random(
        const std::tuple<int, int, int> &hidden_size,
        const std::vector<Image_Visible_Layer_Desc> &visible_layer_descs
    );

    void init_from_file(
        const std::string &file_name
    );

    void init_from_buffer(
        const py::array_t<unsigned char> &buffer
    );

public:
    aon::Image_Encoder::Params params;

    Image_Encoder(
        const std::tuple<int, int, int> &hidden_size,
        const std::vector<Image_Visible_Layer_Desc> &visible_layer_descs,
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
        return enc.size();
    }

    long get_state_size() const {
        return enc.state_size();
    }

    long get_weights_size() const {
        return enc.weights_size();
    }

    void step(
        const std::vector<py::array_t<unsigned char, py::array::c_style | py::array::forcecast>> &inputs,
        bool learn_enabled,
        bool learn_recon
    );

    void reconstruct(
        const py::array_t<int, py::array::c_style | py::array::forcecast> &recon_cis
    );

    int get_num_visible_layers() const {
        return enc.get_num_visible_layers();
    }

    py::array_t<unsigned char> get_reconstruction(
        int i
    ) const;

    py::array_t<int> get_hidden_cis() const;

    std::tuple<int, int, int> get_hidden_size() const {
        aon::Int3 size = enc.get_hidden_size();

        return { size.x, size.y, size.z };
    }

    std::tuple<int, int, int> get_visible_size(
        int i
    ) const {
        aon::Int3 size = enc.get_visible_layer_desc(i).size;

        return { size.x, size.y, size.z };
    }

    void merge(
        const std::vector<Image_Encoder*> &image_encoders,
        Merge_Mode mode
    );
};
}
