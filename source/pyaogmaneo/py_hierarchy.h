// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "py_helpers.h"
#include <aogmaneo/hierarchy.h>

namespace pyaon {
const int hierarchy_magic = 3133285;

enum IO_Type {
    none = 0,
    prediction = 1,
    action = 2
};

struct IO_Desc {
    std::tuple<int, int, int> size;
    IO_Type type;

    int up_radius;
    int down_radius;

    int history_capacity;

    IO_Desc(
        const std::tuple<int, int, int> &size,
        IO_Type type,
        int up_radius,
        int down_radius,
        int history_capacity
    )
    :
    size(size),
    type(type),
    up_radius(up_radius),
    down_radius(down_radius),
    history_capacity(history_capacity)
    {}

    void check_in_range() const;
};

struct Layer_Desc {
    std::tuple<int, int, int> hidden_size;

    int up_radius;
    int down_radius;

    int ticks_per_update;
    int temporal_horizon;

    Layer_Desc(
        const std::tuple<int, int, int> &hidden_size,
        int up_radius,
        int down_radius,
        int ticks_per_update,
        int temporal_horizon
    )
    :
    hidden_size(hidden_size),
    up_radius(up_radius),
    down_radius(down_radius),
    ticks_per_update(ticks_per_update),
    temporal_horizon(temporal_horizon)
    {}

    void check_in_range() const;
};

struct Params {
    std::vector<aon::Hierarchy::Layer_Params> layers;
    std::vector<aon::Hierarchy::IO_Params> ios;
};

class Hierarchy {
private:
    aon::Hierarchy h;

    void init_random(
        const std::vector<IO_Desc> &io_descs,
        const std::vector<Layer_Desc> &layer_descs
    );

    void init_from_file(
        const std::string &file_name
    );

    void init_from_buffer(
        const std::vector<unsigned char> &buffer
    );

    void copy_params_to_h();

public:
    Params params;

    Hierarchy(
        const std::vector<IO_Desc> &io_descs,
        const std::vector<Layer_Desc> &layer_descs,
        const std::string &file_name,
        const std::vector<unsigned char> &buffer
    );

    void save_to_file(
        const std::string &file_name
    );

    std::vector<unsigned char> serialize_to_buffer();

    void set_state_from_buffer(
        const std::vector<unsigned char> &buffer
    );

    std::vector<unsigned char> serialize_state_to_buffer();

    void step(
        const std::vector<std::vector<int>> &input_cis,
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

    std::vector<int> get_prediction_cis(
        int i
    ) const;

    std::vector<int> get_layer_prediction_cis(
        int l
    ) const;

    std::vector<float> get_prediction_probs(
        int i
    ) const;

    std::vector<int> sample_prediction(
        int i,
        float temperature
    ) const;

    std::vector<int> get_hidden_cis(
        int l
    ) {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        std::vector<int> hidden_cis(h.get_encoder(l).get_hidden_cis().size());

        for (int j = 0; j < hidden_cis.size(); j++)
            hidden_cis[j] = h.get_encoder(l).get_hidden_cis()[j];

        return hidden_cis;
    }

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

    int get_ticks(
        int l
    ) const {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        return h.get_ticks(l);
    }

    int get_ticks_per_update(
        int l
    ) const {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        return h.get_ticks_per_update(l);
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

    int get_actor_history_capacity(
        int i
    ) const {
        if (i < 0 || i >= h.get_num_io() || h.get_io_type(i) != aon::action)
            throw std::runtime_error("error: " + std::to_string(i) + " is not a valid input index!");

        return h.get_actor(i).get_history_capacity();
    }

    // for visualization mostly
    std::tuple<std::vector<float>, std::tuple<int, int, int>> get_encoder_receptive_field(
        int l,
        int i,
        const std::tuple<int, int, int> &cell_pos
    );

    std::tuple<std::vector<float>, std::tuple<int, int, int>> get_decoder_receptive_field(
        int l,
        int i,
        bool feedback,
        const std::tuple<int, int, int> &cell_pos
    );
};
}
