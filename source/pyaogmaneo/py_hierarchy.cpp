// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "py_hierarchy.h"

using namespace pyaon;

void IO_Desc::check_in_range() const {
    if (std::get<0>(size) < 1)
        throw std::runtime_error("error: size[0] < 1 is not allowed!");

    if (std::get<1>(size) < 1)
        throw std::runtime_error("error: size[1] < 1 is not allowed!");

    if (std::get<2>(size) < 1)
        throw std::runtime_error("error: size[2] < 1 is not allowed!");

    if (up_radius < 0)
        throw std::runtime_error("error: up_radius < 0 is not allowed!");

    if (down_radius < 0)
        throw std::runtime_error("error: down_radius < 0 is not allowed!");

    if (history_capacity < 2)
        throw std::runtime_error("error: history_capacity < 2 is not allowed!");
}

void Layer_Desc::check_in_range() const {
    if (std::get<0>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[0] < 1 is not allowed!");

    if (std::get<1>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[1] < 1 is not allowed!");

    if (std::get<2>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[2] < 1 is not allowed!");

    if (up_radius < 0)
        throw std::runtime_error("error: up_radius < 0 is not allowed!");

    if (down_radius < 0)
        throw std::runtime_error("error: down_radius < 0 is not allowed!");

    if (ticks_per_update < 1)
        throw std::runtime_error("error: ticks_per_update < 1 is not allowed!");

    if (temporal_horizon < 1)
        throw std::runtime_error("error: temporal_horizon < 1 is not allowed!");

    if (ticks_per_update > temporal_horizon)
        throw std::runtime_error("error: ticks_per_update > temporal_horizon is not allowed!");
}

void Hierarchy::enc_get_set_index_check(
    int l
) const {
    if (l < 0 || l >= h.get_num_layers())
        throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");
}

void Hierarchy::dec_get_set_index_check(
    int l, int i
) const {
    if (l < 0 || l >= h.get_num_layers())
        throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

    if (l == 0 && (i < 0 || i >= h.get_num_io()))
        throw std::runtime_error("error: " + std::to_string(i) + " is not a valid input index!");

    if (l == 0 && (!h.io_layer_exists(i) || h.get_io_type(i) != aon::prediction))
        throw std::runtime_error("error: index " + std::to_string(i) + " does not have a decoder!");
}

void Hierarchy::act_get_set_index_check(
    int i
) const {
    if (i < 0 || i >= h.get_num_io())
        throw std::runtime_error("error: " + std::to_string(i) + " is not a valid input index!");

    if (!h.io_layer_exists(i) || h.get_io_type(i) != aon::action)
        throw std::runtime_error("error: index " + std::to_string(i) + " does not have an actor!");
}

Hierarchy::Hierarchy(
    const std::vector<IO_Desc> &io_descs,
    const std::vector<Layer_Desc> &layer_descs,
    const std::string &name,
    const std::vector<unsigned char> &buffer
) {
    if (!buffer.empty())
        init_from_buffer(buffer);
    else if (!name.empty())
        init_from_file(name);
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
}

void Hierarchy::init_random(
    const std::vector<IO_Desc> &io_descs,
    const std::vector<Layer_Desc> &layer_descs
) {
    aon::Array<aon::Hierarchy::IO_Desc> c_io_descs(io_descs.size());

    for (int i = 0; i < io_descs.size(); i++) {
        io_descs[i].check_in_range();

        c_io_descs[i] = aon::Hierarchy::IO_Desc(
            aon::Int3(std::get<0>(io_descs[i].size), std::get<1>(io_descs[i].size), std::get<2>(io_descs[i].size)),
            static_cast<aon::IO_Type>(io_descs[i].type),
            io_descs[i].up_radius,
            io_descs[i].down_radius,
            io_descs[i].history_capacity
        );
    }
    
    aon::Array<aon::Hierarchy::Layer_Desc> c_layer_descs(layer_descs.size());

    for (int l = 0; l < layer_descs.size(); l++) {
        layer_descs[l].check_in_range();

        c_layer_descs[l] = aon::Hierarchy::Layer_Desc(
            aon::Int3(std::get<0>(layer_descs[l].hidden_size), std::get<1>(layer_descs[l].hidden_size), std::get<2>(layer_descs[l].hidden_size)),
            layer_descs[l].up_radius,
            layer_descs[l].down_radius,
            layer_descs[l].ticks_per_update,
            layer_descs[l].temporal_horizon
        );
    }

    h.init_random(c_io_descs, c_layer_descs);
}

void Hierarchy::init_from_file(
    const std::string &name
) {
    File_Reader reader;
    reader.ins.open(name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != hierarchy_magic)
        throw std::runtime_error("attempted to initialize Hierarchy from incompatible file - " + name);

    h.read(reader);
}

void Hierarchy::init_from_buffer(
    const std::vector<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != hierarchy_magic)
        throw std::runtime_error("attempted to initialize Hierarchy from incompatible buffer!");

    h.read(reader);
}

void Hierarchy::save_to_file(
    const std::string &name
) {
    File_Writer writer;
    writer.outs.open(name, std::ios::binary);

    writer.write(&hierarchy_magic, sizeof(int));

    h.write(writer);
}

std::vector<unsigned char> Hierarchy::serialize_to_buffer() {
    Buffer_Writer writer(h.size() + sizeof(int));

    writer.write(&hierarchy_magic, sizeof(int));

    h.write(writer);

    return writer.buffer;
}

void Hierarchy::set_state_from_buffer(
    const std::vector<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != hierarchy_magic)
        throw std::runtime_error("attempted to set Hierarchy state from incompatible buffer!");

    h.read_state(reader);
}

std::vector<unsigned char> Hierarchy::serialize_state_to_buffer() {
    Buffer_Writer writer(h.state_size() + sizeof(int));

    writer.write(&hierarchy_magic, sizeof(int));

    h.write_state(writer);

    return writer.buffer;
}

void Hierarchy::step(
    const std::vector<std::vector<int>> &input_cis,
    bool learn_enabled,
    float reward,
    float mimic
) {
    if (input_cis.size() != h.get_num_io())
        throw std::runtime_error("incorrect number of input_cis passed to step! received " + std::to_string(input_cis.size()) + ", need " + std::to_string(h.get_num_io()));

    copy_params_to_h();

    aon::Array<aon::Int_Buffer> c_input_cis_backing(input_cis.size());
    aon::Array<const aon::Int_Buffer*> c_input_cis(input_cis.size());

    for (int i = 0; i < input_cis.size(); i++) {
        int num_columns = h.get_io_size(i).x * h.get_io_size(i).y;

        if (input_cis[i].size() != num_columns)
            throw std::runtime_error("incorrect csdr size at index " + std::to_string(i) + " - expected " + std::to_string(num_columns) + " columns, got " + std::to_string(input_cis[i].size()));

        c_input_cis_backing[i].resize(input_cis[i].size());

        for (int j = 0; j < input_cis[i].size(); j++) {
            if (input_cis[i][j] < 0 || input_cis[i][j] >= h.get_io_size(i).z)
                throw std::runtime_error("input csdr at input index " + std::to_string(i) + " has an out-of-bounds column index (" + std::to_string(input_cis[i][j]) + ") at column index " + std::to_string(j) + ". it must be in the range [0, " + std::to_string(h.get_io_size(i).z - 1) + "]");

            c_input_cis_backing[i][j] = input_cis[i][j];
        }

        c_input_cis[i] = &c_input_cis_backing[i];
    }
    
    h.step(c_input_cis, learn_enabled, reward, mimic);
}

std::vector<int> Hierarchy::get_prediction_cis(
    int i
) const {
    if (i < 0 || i >= h.get_num_io())
        throw std::runtime_error("prediction index " + std::to_string(i) + " out of range [0, " + std::to_string(h.get_num_io() - 1) + "]!");

    if (!h.io_layer_exists(i) || h.get_io_type(i) == aon::none)
        throw std::runtime_error("no decoder exists at index " + std::to_string(i) + " - did you set it to the correct type?");

    std::vector<int> predictions(h.get_prediction_cis(i).size());

    for (int j = 0; j < predictions.size(); j++)
        predictions[j] = h.get_prediction_cis(i)[j];

    return predictions;
}

void Hierarchy::copy_params_to_h() {
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
