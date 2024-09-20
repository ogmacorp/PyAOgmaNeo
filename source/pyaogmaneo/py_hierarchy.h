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
const int hierarchy_magic = 2840321;
const int hierarchy_magic_stride = 64;

enum IO_Type {
    none = 0,
    prediction = 1
};

struct IO_Desc {
    std::tuple<int, int> size;
    IO_Type type;

    int radius;

    float positional_scale;

    IO_Desc(
        const std::tuple<int, int> &size,
        IO_Type type,
        int radius,
        float positional_scale
    )
    :
    size(size),
    type(type),
    radius(radius),
    positional_scale(positional_scale)
    {}

    void check_in_range() const;
};

struct Layer_Desc {
    std::tuple<int, int> hidden_size;

    int radius;

    float positional_scale;

    Layer_Desc(
        const std::tuple<int, int> &hidden_size,
        int radius,
        float positional_scale
    )
    :
    hidden_size(hidden_size),
    radius(radius),
    positional_scale(positional_scale)
    {}

    void check_in_range() const;
};

struct Params {
    std::vector<aon::Layer_Params> layers;
    std::vector<aon::IO_Params> ios;
};

template<int S, int L>
class Hierarchy {
private:
    aon::Hierarchy<S, L> h;

    aon::Array<aon::Array<aon::Vec<S, L>>> c_input_vecs_backing;
    aon::Array<aon::Array_View<aon::Vec<S, L>>> c_input_vecs;

    void init_random(
        const std::vector<IO_Desc> &io_descs,
        const std::vector<Layer_Desc> &layer_descs
    ) {
        aon::Array<aon::IO_Desc> c_io_descs(io_descs.size());

        for (int i = 0; i < io_descs.size(); i++) {
            io_descs[i].check_in_range();

            c_io_descs[i] = aon::IO_Desc(
                aon::Int2(std::get<0>(io_descs[i].size), std::get<1>(io_descs[i].size)),
                static_cast<aon::IO_Type>(io_descs[i].type),
                io_descs[i].radius,
                io_descs[i].positional_scale
            );
        }
        
        aon::Array<aon::Layer_Desc> c_layer_descs(layer_descs.size());

        for (int l = 0; l < layer_descs.size(); l++) {
            layer_descs[l].check_in_range();

            c_layer_descs[l] = aon::Layer_Desc(
                aon::Int2(std::get<0>(layer_descs[l].hidden_size), std::get<1>(layer_descs[l].hidden_size)),
                layer_descs[l].radius,
                layer_descs[l].positional_scale
            );
        }

        h.init_random(c_io_descs, c_layer_descs);
    }

    void init_from_file(
        const std::string &file_name
    ) {
        File_Reader reader;
        reader.ins.open(file_name, std::ios::binary);

        int magic;
        reader.read(&magic, sizeof(int));

        if (magic != hierarchy_magic)
            throw std::runtime_error("attempted to initialize Hierarchy from incompatible file - " + file_name);

        h.read(reader);
    }

    void init_from_buffer(
        const py::array_t<unsigned char> &buffer
    ) {
        Buffer_Reader reader;
        reader.buffer = &buffer;

        int magic;
        reader.read(&magic, sizeof(int));

        if (magic != hierarchy_magic)
            throw std::runtime_error("attempted to initialize Hierarchy from incompatible buffer!");

        h.read(reader);
    }

    void copy_params_to_h() {
        if (params.ios.size() != h.params.ios.size())
            throw std::runtime_error("ios parameter size mismatch - did you modify the length of params.ios?");

        if (params.layers.size() != h.params.layers.size())
            throw std::runtime_error("layers parameter size mismatch - did you modify the length of params.layers?");
        
        // copy params
        for (int i = 0; i < params.ios.size(); i++)
            h.params.ios[i] = params.ios[i];

        // copy params
        for (int l = 0; l < params.layers.size(); l++)
            h.params.layers[l] = params.layers[l];
    }

public:
    Params params;

    Hierarchy(
        const std::vector<IO_Desc> &io_descs,
        const std::vector<Layer_Desc> &layer_descs,
        const std::string &file_name,
        const py::array_t<unsigned char> &buffer
    ) {
        if (buffer.unchecked().size() > 0)
            init_from_buffer(buffer);
        else if (!file_name.empty())
            init_from_file(file_name);
        else {
            if (io_descs.empty() || layer_descs.empty())
                throw std::runtime_error("error: Hierarchy constructor requires some non-empty arguments!");

            init_random(io_descs, layer_descs);
        }

        // copy params
        params.ios.resize(h.get_num_io());

        for (int i = 0; i < h.get_num_io(); i++)
            params.ios[i] = h.params.ios[i];

        // copy params
        params.layers.resize(h.get_num_layers());

        for (int l = 0; l < h.get_num_layers(); l++)
            params.layers[l] = h.params.layers[l];

        c_input_vecs_backing.resize(h.get_num_io());
        c_input_vecs.resize(h.get_num_io());

        for (int i = 0; i < c_input_vecs_backing.size(); i++)
            c_input_vecs_backing[i].resize(h.get_io_size(i).x * h.get_io_size(i).y);
    }

    void save_to_file(
        const std::string &file_name
    ) {
        File_Writer writer;
        writer.outs.open(file_name, std::ios::binary);

        int magic = hierarchy_magic + L + hierarchy_magic * S;

        writer.write(&magic, sizeof(int));

        h.write(writer);
    }

    void set_state_from_buffer(
        const py::array_t<unsigned char> &buffer
    ) {
        Buffer_Reader reader;
        reader.buffer = &buffer;

        h.read_state(reader);
    }

    void set_weights_from_buffer(
        const py::array_t<unsigned char> &buffer
    ) {
        Buffer_Reader reader;
        reader.buffer = &buffer;

        h.read_weights(reader);
    }

    py::array_t<unsigned char> serialize_to_buffer() {
        Buffer_Writer writer(h.size() + sizeof(int));

        int magic = hierarchy_magic + L + hierarchy_magic * S;

        writer.write(&magic, sizeof(int));

        h.write(writer);

        return writer.buffer;
    }

    py::array_t<unsigned char> serialize_state_to_buffer() {
        Buffer_Writer writer(h.state_size());

        h.write_state(writer);

        return writer.buffer;
    }

    py::array_t<unsigned char> serialize_weights_to_buffer() {
        Buffer_Writer writer(h.weights_size());

        h.write_weights(writer);

        return writer.buffer;
    }

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
        const std::vector<std::vector<aon::Vec<S, L>>> &input_vecs,
        bool learn_enabled
    ) {
        if (input_vecs.size() != h.get_num_io())
            throw std::runtime_error("incorrect number of input vecs passed to step! received " + std::to_string(input_vecs.size()) + ", need " + std::to_string(h.get_num_io()));

        copy_params_to_h();

        for (int i = 0; i < input_vecs.size(); i++) {
            int num_columns = h.get_io_size(i).x * h.get_io_size(i).y;

            if (input_vecs[i].size() != num_columns)
                throw std::runtime_error("incorrect csdr size at index " + std::to_string(i) + " - expected " + std::to_string(num_columns) + " columns, got " + std::to_string(input_vecs[i].size()));

            for (int j = 0; j < input_vecs[i].size(); j++)
                c_input_vecs_backing[i][j] = input_vecs[i][j];

            c_input_vecs[i] = c_input_vecs_backing[i];
        }
        
        h.step(c_input_vecs, learn_enabled);

    }

    void clear_state() {
        h.clear_state();
    }

    int get_num_layers() const {
        return h.get_num_layers();
    }

    std::vector<aon::Vec<S, L>> get_prediction_vecs(
        int i
    ) {
        if (i < 0 || i >= h.get_num_io())
            throw std::runtime_error("prediction index " + std::to_string(i) + " out of range [0, " + std::to_string(h.get_num_io() - 1) + "]!");

        if (h.get_io_type(i) == aon::none)
            throw std::runtime_error("no decoder exists at index " + std::to_string(i) + " - did you set it to the correct type?");

        std::vector<aon::Vec<S, L>> prediction_vecs(h.get_prediction_vecs(i).size());

        for (int j = 0; j < prediction_vecs.size(); j++)
            prediction_vecs[j] = h.get_prediction_vecs(i)[j];

        return prediction_vecs;
    }

    std::vector<aon::Vec<S, L>> get_hidden_vecs(
        int l
    ) {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        std::vector<aon::Vec<S, L>> hidden_vecs(h.get_layer(l).get_hidden_vecs().size());

        for (int j = 0; j < hidden_vecs.size(); j++)
            hidden_vecs[j] = h.get_layer(l).get_hidden_vecs()[j];

        return hidden_vecs;
    }

    std::tuple<int, int> get_hidden_size(
        int l
    ) {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        aon::Int2 size = h.get_layer(l).get_hidden_size();

        return { size.x, size.y };
    }

    int get_num_visible_layers(
        int l
    ) {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        return h.get_layer(l).get_num_visible_layers();
    }

    int get_num_io() const {
        return h.get_num_io();
    }

    std::tuple<int, int> get_io_size(
        int i
    ) const {
        if (i < 0 || i >= h.get_num_io())
            throw std::runtime_error("error: " + std::to_string(i) + " is not a valid input index!");

        aon::Int2 size = h.get_io_size(i);

        return { size.x, size.y };
    }

    IO_Type get_io_type(
        int i
    ) const {
        if (i < 0 || i >= h.get_num_io())
            throw std::runtime_error("error: " + std::to_string(i) + " is not a valid input index!");

        return static_cast<IO_Type>(h.get_io_type(i));
    }

    // retrieve additional parameters on the sph's structure
    int get_radius(
        int l
    ) const {
        if (l < 0 || l >= h.get_num_layers())
            throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

        return h.get_layer(l).get_visible_layer_desc(0).radius;
    }
};
}
