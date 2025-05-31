// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "py_image_encoder.h"

using namespace pyaon;

void Image_Visible_Layer_Desc::check_in_range() const {
    if (std::get<0>(size) < 1)
        throw std::runtime_error("error: size[0] < 1 is not allowed!");

    if (std::get<1>(size) < 1)
        throw std::runtime_error("error: size[1] < 1 is not allowed!");

    if (std::get<2>(size) < 1)
        throw std::runtime_error("error: size[2] < 1 is not allowed!");

    if (radius < 0)
        throw std::runtime_error("error: radius < 0 is not allowed!");
}

Image_Encoder::Image_Encoder(
    const std::tuple<int, int, int> &hidden_size,
    const std::vector<Image_Visible_Layer_Desc> &visible_layer_descs,
    const std::string &file_name,
    const py::array_t<unsigned char> &buffer
) {
    if (buffer.unchecked().size() > 0)
        init_from_buffer(buffer);
    else if (!file_name.empty())
        init_from_file(file_name);
    else {
        if (visible_layer_descs.empty())
            throw std::runtime_error("error: Image_Encoder constructor requires some non-empty arguments!");

        init_random(hidden_size, visible_layer_descs);
    }

    // copy params
    params = enc.params;

    c_inputs_backing.resize(enc.get_num_visible_layers());
    c_inputs.resize(enc.get_num_visible_layers());

    for (int i = 0; i < c_inputs_backing.size(); i++)
        c_inputs_backing[i].resize(enc.get_visible_layer_desc(i).size.x * enc.get_visible_layer_desc(i).size.y * enc.get_visible_layer_desc(i).size.z);
}

void Image_Encoder::init_random(
    const std::tuple<int, int, int> &hidden_size,
    const std::vector<Image_Visible_Layer_Desc> &visible_layer_descs
) {
    bool all_in_range = true;

    aon::Array<aon::Image_Encoder::Visible_Layer_Desc> c_visible_layer_descs(visible_layer_descs.size());

    for (int v = 0; v < visible_layer_descs.size(); v++) {
        visible_layer_descs[v].check_in_range();

        c_visible_layer_descs[v].size = aon::Int3(std::get<0>(visible_layer_descs[v].size), std::get<1>(visible_layer_descs[v].size), std::get<2>(visible_layer_descs[v].size));
        c_visible_layer_descs[v].radius = visible_layer_descs[v].radius;
    }

    if (std::get<0>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[0] < 1 is not allowed!");

    if (std::get<1>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[1] < 1 is not allowed!");

    if (std::get<2>(hidden_size) < 1)
        throw std::runtime_error("error: hidden_size[2] < 1 is not allowed!");

    if (!all_in_range)
        throw std::runtime_error(" - Image_Encoder: some parameters out of range!");

    enc.init_random(aon::Int3(std::get<0>(hidden_size), std::get<1>(hidden_size), std::get<2>(hidden_size)), c_visible_layer_descs);
}

void Image_Encoder::init_from_file(
    const std::string &file_name
) {
    File_Reader reader;
    reader.ins.open(file_name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != image_encoder_magic)
        throw std::runtime_error("attempted to initialize Image_Encoder from incompatible file - " + file_name);

    enc.read(reader);
}

void Image_Encoder::init_from_buffer(
    const py::array_t<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != image_encoder_magic)
        throw std::runtime_error("attempted to initialize Image_Encoder from incompatible buffer!");

    enc.read(reader);
}

void Image_Encoder::save_to_file(
    const std::string &file_name
) {
    File_Writer writer;
    writer.outs.open(file_name, std::ios::binary);

    writer.write(&image_encoder_magic, sizeof(int));

    enc.write(writer);
}

void Image_Encoder::set_state_from_buffer(
    const py::array_t<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    enc.read_state(reader);
}

void Image_Encoder::set_weights_from_buffer(
    const py::array_t<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    enc.read_weights(reader);
}

py::array_t<unsigned char> Image_Encoder::serialize_to_buffer() {
    Buffer_Writer writer(enc.size() + sizeof(int));

    writer.write(&image_encoder_magic, sizeof(int));

    enc.write(writer);

    return writer.buffer;
}

py::array_t<unsigned char> Image_Encoder::serialize_state_to_buffer() {
    Buffer_Writer writer(enc.state_size());

    enc.write_state(writer);

    return writer.buffer;
}

py::array_t<unsigned char> Image_Encoder::serialize_weights_to_buffer() {
    Buffer_Writer writer(enc.weights_size());

    enc.write_weights(writer);

    return writer.buffer;
}

void Image_Encoder::step(
    const std::vector<py::array_t<unsigned char, py::array::c_style | py::array::forcecast>> &inputs,
    bool learn_enabled,
    bool learn_recon
) {
    if (inputs.size() != enc.get_num_visible_layers())
        throw std::runtime_error("incorrect number of inputs given to Image_Encoder! expected " + std::to_string(enc.get_num_visible_layers()) + ", got " + std::to_string(inputs.size()));

    // copy params
    enc.params = params;

    for (int i = 0; i < inputs.size(); i++) {
        auto view = inputs[i].unchecked();

        for (int j = 0; j < view.size(); j++)
            c_inputs_backing[i][j] = view(j);

        c_inputs[i] = c_inputs_backing[i];
    }

    enc.step(c_inputs, learn_enabled, learn_recon);
}

void Image_Encoder::reconstruct(
    const py::array_t<int, py::array::c_style | py::array::forcecast> &recon_cis
) {
    if (recon_cis.size() != enc.get_hidden_cis().size())
        throw std::runtime_error("error: recon_cis must match the output_size of the Image_Encoder!");

    auto view = recon_cis.unchecked();

    aon::Int_Buffer c_recon_cis_backing(view.size());

    for (int j = 0; j < view.size(); j++) {
        if (view(j) < 0 || view(j) >= enc.get_hidden_size().z)
            throw std::runtime_error("recon csdr (recon_cis) has an out-of-bounds column index (" + std::to_string(view(j)) + ") at column index " + std::to_string(j) + ". it must be in the range [0, " + std::to_string(enc.get_hidden_size().z - 1) + "]");

        c_recon_cis_backing[j] = view(j);
    }

    enc.reconstruct(c_recon_cis_backing);
}

py::array_t<unsigned char> Image_Encoder::get_reconstruction(
    int i
) const {
    if (i < 0 || i >= enc.get_num_visible_layers())
        throw std::runtime_error("cannot get reconstruction at index " + std::to_string(i) + " - out of bounds [0, " + std::to_string(enc.get_num_visible_layers()) + "]");

    py::array_t<unsigned char> reconstruction(enc.get_reconstruction(i).size());

    auto view = reconstruction.mutable_unchecked();

    for (int j = 0; j < view.size(); j++)
        view(j) = enc.get_reconstruction(i)[j];

    return reconstruction;
}

py::array_t<int> Image_Encoder::get_hidden_cis() const {
    py::array_t<int> hidden_cis(enc.get_hidden_cis().size());

    auto view = hidden_cis.mutable_unchecked();

    for (int j = 0; j < view.size(); j++)
        view(j) = enc.get_hidden_cis()[j];

    return hidden_cis;
}

std::tuple<py::array_t<unsigned char>, std::tuple<int, int, int>> Image_Encoder::get_receptive_field(
    int vli,
    const std::tuple<int, int, int> &pos
) {
    int num_visible_layers = enc.get_num_visible_layers();

    if (vli < 0 || vli >= num_visible_layers)
        throw std::runtime_error("visible layer index " + std::to_string(vli) + " out of range [0, " + std::to_string(num_visible_layers - 1) + "]!");

    const aon::Int3 &hidden_size = enc.get_hidden_size();

    if (std::get<0>(pos) < 0 || std::get<0>(pos) >= hidden_size.x ||
        std::get<1>(pos) < 0 || std::get<1>(pos) >= hidden_size.y ||
        std::get<2>(pos) < 0 || std::get<2>(pos) >= hidden_size.z) {
        throw std::runtime_error("position (" + std::to_string(std::get<0>(pos)) + ", " + std::to_string(std::get<1>(pos)) + ", " + std::to_string(std::get<2>(pos)) + ") " +
                + " not in size (" + std::to_string(hidden_size.x) + ", " + std::to_string(hidden_size.y) + ", " + std::to_string(hidden_size.z) + ")!");
    }

    const aon::Image_Encoder::Visible_Layer &vl = enc.get_visible_layer(vli);
    const aon::Image_Encoder::Visible_Layer_Desc &vld = enc.get_visible_layer_desc(vli);

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    aon::Int2 column_pos(std::get<0>(pos), std::get<1>(pos));

    int hidden_column_index = aon::address2(column_pos, aon::Int2(hidden_size.x, hidden_size.y));
    int hidden_cells_start = hidden_size.z * hidden_column_index;

    // projection
    aon::Float2 h_to_v = aon::Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

    aon::Int2 visible_center = project(column_pos, h_to_v);

        // lower corner
    aon::Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

        // bounds of receptive field, clamped to input size
    aon::Int2 iter_lower_bound(aon::max(0, field_lower_bound.x), aon::max(0, field_lower_bound.y));
    aon::Int2 iter_upper_bound(aon::min(vld.size.x - 1, visible_center.x + vld.radius), aon::min(vld.size.y - 1, visible_center.y + vld.radius));

    int hidden_stride = vld.size.z * diam * diam;

    int field_count = area * vld.size.z;

    py::array_t<unsigned char> field(field_count);

    auto view = field.mutable_unchecked();

    // first clear
    for (int i = 0; i < field_count; i++)
        view(i) = 0;

    int hidden_cell_index = std::get<2>(pos) + hidden_cells_start;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int visible_column_index = address2(aon::Int2(ix, iy), aon::Int2(vld.size.x, vld.size.y));

            aon::Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

            int wi_start_partial = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

            for (int vc = 0; vc < vld.size.z; vc++) {
                int wi = std::get<2>(pos) + hidden_size.z * (vc + wi_start_partial);

                view(vc + vld.size.z * (offset.y + diam * offset.x)) = vl.weights[wi];
            }
        }

    std::tuple<int, int, int> field_size(diam, diam, vld.size.z);

    return std::make_tuple(field, field_size);
}

void Image_Encoder::merge(
    const std::vector<Image_Encoder*> &image_encoders,
    Merge_Mode mode
) {
    aon::Array<aon::Image_Encoder*> c_image_encoders(image_encoders.size());

    for (int i = 0; i < image_encoders.size(); i++)
        c_image_encoders[i] = &image_encoders[i]->enc;

    enc.merge(c_image_encoders, static_cast<aon::Merge_Mode>(mode));
}
