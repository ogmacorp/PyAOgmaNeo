// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
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

    if (num_dendrites_per_cell < 1)
        throw std::runtime_error("error: num_dendrites_per_cell < 1 is not allowed!");

    if (value_num_dendrites_per_cell < 2)
        throw std::runtime_error("error: value_num_dendrites_per_cell < 2 is not allowed!");

    if (up_radius < 0)
        throw std::runtime_error("error: up_radius < 0 is not allowed!");

    if (down_radius < 0)
        throw std::runtime_error("error: down_radius < 0 is not allowed!");
}

void Layer_Desc::check_in_range() const {
    if (std::get<0>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[0] < 1 is not allowed!");

    if (std::get<1>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[1] < 1 is not allowed!");

    if (std::get<2>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[2] < 1 is not allowed!");

    if (num_dendrites_per_cell < 1)
        throw std::runtime_error("error: num_dendrites_per_cell < 1 is not allowed!");

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

Hierarchy::Hierarchy(
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

    params.anticipation = h.params.anticipation;

    c_input_cis_backing.resize(h.get_num_io());
    c_input_cis.resize(h.get_num_io());

    for (int i = 0; i < c_input_cis_backing.size(); i++)
        c_input_cis_backing[i].resize(h.get_io_size(i).x * h.get_io_size(i).y);
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
            io_descs[i].num_dendrites_per_cell,
            io_descs[i].value_num_dendrites_per_cell,
            io_descs[i].up_radius,
            io_descs[i].down_radius
        );
    }
    
    aon::Array<aon::Hierarchy::Layer_Desc> c_layer_descs(layer_descs.size());

    for (int l = 0; l < layer_descs.size(); l++) {
        layer_descs[l].check_in_range();

        c_layer_descs[l] = aon::Hierarchy::Layer_Desc(
            aon::Int3(std::get<0>(layer_descs[l].hidden_size), std::get<1>(layer_descs[l].hidden_size), std::get<2>(layer_descs[l].hidden_size)),
            layer_descs[l].num_dendrites_per_cell,
            layer_descs[l].up_radius,
            layer_descs[l].down_radius,
            layer_descs[l].ticks_per_update,
            layer_descs[l].temporal_horizon
        );
    }

    h.init_random(c_io_descs, c_layer_descs);
}

void Hierarchy::init_from_file(
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

void Hierarchy::init_from_buffer(
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

void Hierarchy::save_to_file(
    const std::string &file_name
) {
    File_Writer writer;
    writer.outs.open(file_name, std::ios::binary);

    writer.write(&hierarchy_magic, sizeof(int));

    h.write(writer);
}

void Hierarchy::set_state_from_buffer(
    const py::array_t<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    h.read_state(reader);
}

void Hierarchy::set_weights_from_buffer(
    const py::array_t<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    h.read_weights(reader);
}

py::array_t<unsigned char> Hierarchy::serialize_to_buffer() {
    Buffer_Writer writer(h.size() + sizeof(int));

    writer.write(&hierarchy_magic, sizeof(int));

    h.write(writer);

    return writer.buffer;
}

py::array_t<unsigned char> Hierarchy::serialize_state_to_buffer() {
    Buffer_Writer writer(h.state_size());

    h.write_state(writer);

    return writer.buffer;
}

py::array_t<unsigned char> Hierarchy::serialize_weights_to_buffer() {
    Buffer_Writer writer(h.weights_size());

    h.write_weights(writer);

    return writer.buffer;
}

void Hierarchy::step(
    const std::vector<py::array_t<int, py::array::c_style | py::array::forcecast>> &input_cis,
    bool learn_enabled,
    float reward,
    float mimic
) {
    if (input_cis.size() != h.get_num_io())
        throw std::runtime_error("incorrect number of input_cis passed to step! received " + std::to_string(input_cis.size()) + ", need " + std::to_string(h.get_num_io()));

    copy_params_to_h();

    for (int i = 0; i < input_cis.size(); i++) {
        auto view = input_cis[i].unchecked();

        int num_columns = h.get_io_size(i).x * h.get_io_size(i).y;

        if (view.size() != num_columns)
            throw std::runtime_error("incorrect csdr size at index " + std::to_string(i) + " - expected " + std::to_string(num_columns) + " columns, got " + std::to_string(view.size()));

        for (int j = 0; j < view.size(); j++) {
            if (view(j) < 0 || view(j) >= h.get_io_size(i).z)
                throw std::runtime_error("input csdr at input index " + std::to_string(i) + " has an out-of-bounds column index (" + std::to_string(view(j)) + ") at column index " + std::to_string(j) + ". it must be in the range [0, " + std::to_string(h.get_io_size(i).z - 1) + "]");

            c_input_cis_backing[i][j] = view(j);
        }

        c_input_cis[i] = c_input_cis_backing[i];
    }
    
    h.step(c_input_cis, learn_enabled, reward, mimic);
}

py::array_t<int> Hierarchy::get_prediction_cis(
    int i
) const {
    if (i < 0 || i >= h.get_num_io())
        throw std::runtime_error("prediction index " + std::to_string(i) + " out of range [0, " + std::to_string(h.get_num_io() - 1) + "]!");

    if (!h.io_layer_exists(i) || h.get_io_type(i) == aon::none)
        throw std::runtime_error("no decoder exists at index " + std::to_string(i) + " - did you set it to the correct type?");

    py::array_t<int> predictions(h.get_prediction_cis(i).size());

    auto view = predictions.mutable_unchecked();

    for (int j = 0; j < view.size(); j++)
        view(j) = h.get_prediction_cis(i)[j];

    return predictions;
}

py::array_t<int> Hierarchy::get_layer_prediction_cis(
    int l
) const {
    if (l < 1 || l >= h.get_num_layers())
        throw std::runtime_error("layer index " + std::to_string(l) + " out of range [1, " + std::to_string(h.get_num_layers() - 1) + "]!");

    const aon::Int_Buffer &cis = h.get_decoder(l, h.get_ticks_per_update(l) - 1 - h.get_ticks(l)).get_hidden_cis();

    py::array_t<int> predictions(cis.size());

    auto view = predictions.mutable_unchecked();

    for (int j = 0; j < view.size(); j++)
        view(j) = cis[j];

    return predictions;
}

py::array_t<float> Hierarchy::get_prediction_acts(
    int i
) const {
    if (i < 0 || i >= h.get_num_io())
        throw std::runtime_error("prediction index " + std::to_string(i) + " out of range [0, " + std::to_string(h.get_num_io() - 1) + "]!");

    if (!h.io_layer_exists(i) || h.get_io_type(i) == aon::none)
        throw std::runtime_error("no decoder or actor exists at index " + std::to_string(i) + " - did you set it to the correct type?");

    py::array_t<float> predictions(h.get_prediction_acts(i).size());

    auto view = predictions.mutable_unchecked();

    for (int j = 0; j < view.size(); j++)
        view(j) = h.get_prediction_acts(i)[j];

    return predictions;
}

py::array_t<int> Hierarchy::sample_prediction(
    int i,
    float temperature
) const {
    if (temperature == 0.0f)
        return get_prediction_cis(i);

    if (i < 0 || i >= h.get_num_io())
        throw std::runtime_error("prediction index " + std::to_string(i) + " out of range [0, " + std::to_string(h.get_num_io() - 1) + "]!");

    if (!h.io_layer_exists(i) || h.get_io_type(i) == aon::none)
        throw std::runtime_error("no decoder or actor exists at index " + std::to_string(i) + " - did you set it to the correct type?");

    py::array_t<int> sample(h.get_prediction_cis(i).size());

    auto view = sample.mutable_unchecked();

    int size_z = h.get_io_size(i).z;

    float temperature_inv = 1.0f / temperature;

    for (int j = 0; j < view.size(); j++) {
        float total = 0.0f;

        for (int k = 0; k < size_z; k++)
            total += aon::powf(h.get_prediction_acts(i)[k + j * size_z], temperature_inv);

        float cusp = aon::randf() * total;

        float sum_so_far = 0.0f;

        for (int k = 0; k < size_z; k++) {
            sum_so_far += aon::powf(h.get_prediction_acts(i)[k + j * size_z], temperature_inv);

            if (sum_so_far >= cusp) {
                view(j) = k;

                break;
            }
        }
    }

    return sample;
}

py::array_t<int> Hierarchy::get_hidden_cis(
    int l
) {
    if (l < 0 || l >= h.get_num_layers())
        throw std::runtime_error("error: " + std::to_string(l) + " is not a valid layer index!");

    py::array_t<int> hidden_cis(h.get_encoder(l).get_hidden_cis().size());

    auto view = hidden_cis.mutable_unchecked();

    for (int j = 0; j < view.size(); j++)
        view(j) = h.get_encoder(l).get_hidden_cis()[j];

    return hidden_cis;
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

    h.params.anticipation = params.anticipation;
}

void Hierarchy::merge(
    const std::vector<Hierarchy*> &hierarchies,
    Merge_Mode mode
) {
    aon::Array<aon::Hierarchy*> c_hierarchies(hierarchies.size());

    for (int h = 0; h < hierarchies.size(); h++)
        c_hierarchies[h] = &hierarchies[h]->h;

    h.merge(c_hierarchies, static_cast<aon::Merge_Mode>(mode));
}
