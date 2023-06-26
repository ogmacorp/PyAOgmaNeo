// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "py_helpers.h"
#include <aogmaneo/image_encoder.h>

namespace pyaon {
const int image_encoder_magic = 6221138;

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

    void init_random(
        const std::tuple<int, int, int> &hidden_size,
        const std::vector<Image_Visible_Layer_Desc> &visible_layer_descs
    );

    void init_from_file(
        const std::string &file_name
    );

    void init_from_buffer(
        const std::vector<unsigned char> &buffer
    );

public:
    aon::Image_Encoder::Params params;

    Image_Encoder(
        const std::tuple<int, int, int> &hidden_size,
        const std::vector<Image_Visible_Layer_Desc> &visible_layer_descs,
        const std::string &file_name,
        const std::vector<unsigned char> &buffer
    );

    void save_to_file(
        const std::string &file_name
    );

    std::vector<unsigned char> serialize_to_buffer();

    void step(
        const std::vector<std::vector<unsigned char>> &inputs,
        bool learn_enabled
    );

    void reconstruct(
        const std::vector<int> &recon_cis
    );

    int get_num_visible_layers() const {
        return enc.get_num_visible_layers();
    }

    std::vector<unsigned char> get_reconstruction(
        int i
    ) const {
        if (i < 0 || i >= enc.get_num_visible_layers())
            throw std::runtime_error("cannot get reconstruction at index " + std::to_string(i) + " - out of bounds [0, " + std::to_string(enc.get_num_visible_layers()) + "]");

        std::vector<unsigned char> reconstruction(enc.get_reconstruction(i).size());

        for (int j = 0; j < reconstruction.size(); j++)
            reconstruction[j] = enc.get_reconstruction(i)[j];

        return reconstruction;
    }

    std::vector<int> get_hidden_cis() const {
        std::vector<int> hidden_cis(enc.get_hidden_cis().size());

        for (int j = 0; j < hidden_cis.size(); j++)
            hidden_cis[j] = enc.get_hidden_cis()[j];

        return hidden_cis;
    }

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

    // for visualization mostly
    std::tuple<std::vector<float>, std::tuple<int, int, int>> get_receptive_field(
        int i,
        const std::tuple<int, int, int> &cell_pos
    );
};
}
