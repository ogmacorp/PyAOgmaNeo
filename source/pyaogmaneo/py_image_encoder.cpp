// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
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
    const std::vector<unsigned char> &buffer
) {
    if (!buffer.empty())
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
    const std::vector<unsigned char> &buffer
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

std::vector<unsigned char> Image_Encoder::serialize_to_buffer() {
    Buffer_Writer writer(enc.size() + sizeof(int));

    writer.write(&image_encoder_magic, sizeof(int));

    enc.write(writer);

    return writer.buffer;
}

void Image_Encoder::step(
    const std::vector<std::vector<unsigned char>> &inputs,
    bool learn_enabled
) {
    if (inputs.size() != enc.get_num_visible_layers())
        throw std::runtime_error("incorrect number of inputs given to Image_Encoder! expected " + std::to_string(enc.get_num_visible_layers()) + ", got " + std::to_string(inputs.size()));

    // copy params
    enc.params = params;

    aon::Array<aon::Byte_Buffer> c_inputs_backing(inputs.size());
    aon::Array<const aon::Byte_Buffer*> c_inputs(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        if (inputs[i].size() != enc.get_reconstruction(i).size())
            throw std::runtime_error("incorrect number of pixels given to Image_Encoder! at input " + std::to_string(i) + ": expected " + std::to_string(enc.get_reconstruction(i).size()) + ", got " + std::to_string(inputs[i].size()));

        c_inputs_backing[i].resize(inputs[i].size());
        
        for (int j = 0; j < inputs[i].size(); j++)
            c_inputs_backing[i][j] = inputs[i][j];

        c_inputs[i] = &c_inputs_backing[i];
    }

    enc.step(c_inputs, learn_enabled);
}

void Image_Encoder::reconstruct(
    const std::vector<int> &recon_cis
) {
    if (recon_cis.size() != enc.get_hidden_cis().size())
        throw std::runtime_error("error: recon_cis must match the output_size of the Image_Encoder!");

    aon::Int_Buffer c_recon_cis_backing(recon_cis.size());

    for (int j = 0; j < recon_cis.size(); j++) {
        if (recon_cis[j] < 0 || recon_cis[j] >= enc.get_hidden_size().z)
            throw std::runtime_error("recon csdr (recon_cis) has an out-of-bounds column index (" + std::to_string(recon_cis[j]) + ") at column index " + std::to_string(j) + ". it must be in the range [0, " + std::to_string(enc.get_hidden_size().z - 1) + "]");

        c_recon_cis_backing[j] = recon_cis[j];
    }

    enc.reconstruct(&c_recon_cis_backing);
}

std::tuple<std::vector<float>, std::tuple<int, int, int>> Image_Encoder::get_receptive_field(
    int i,
    const std::tuple<int, int, int> &cell_pos
) {
    assert(i >= 0 && i < enc.get_num_visible_layers());

    const aon::Int3 &hidden_size = enc.get_hidden_size();

    const aon::Image_Encoder::Visible_Layer &vl = enc.get_visible_layer(i);
    const aon::Image_Encoder::Visible_Layer_Desc &vld = enc.get_visible_layer_desc(i);

    int diam = vld.radius * 2 + 1;

    // projection
    aon::Float2 h_to_v = aon::Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
        static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

    aon::Int2 visible_center = project(aon::Int2(std::get<0>(cell_pos), std::get<1>(cell_pos)), h_to_v);

    // lower corner
    aon::Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

    // bounds of receptive field, clamped to input size
    aon::Int2 iter_lower_bound(aon::max(0, field_lower_bound.x), aon::max(0, field_lower_bound.y));
    aon::Int2 iter_upper_bound(aon::min(vld.size.x - 1, visible_center.x + vld.radius), aon::min(vld.size.y - 1, visible_center.y + vld.radius));

    aon::Int3 size(iter_upper_bound.x - iter_lower_bound.x, iter_upper_bound.y - iter_lower_bound.y, vld.size.z);

    int hidden_cell_index = aon::address3(aon::Int3(std::get<0>(cell_pos), std::get<1>(cell_pos), std::get<2>(cell_pos)), hidden_size);

    // get weights
    std::vector<float> field(size.x * size.y * size.z, 0.0f);

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            aon::Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

            int wi_start = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_cell_index));

            int field_start = vld.size.z * (offset.y + diam * offset.x);

            for (int vc = 0; vc < vld.size.z; vc++) {
                float w = vl.protos[vc + wi_start];

                field[vc + vld.size.z * (offset.y + diam * offset.x)] = w;
            }
        }

    return std::make_tuple(field, std::make_tuple(size.x, size.y, size.z));
}
