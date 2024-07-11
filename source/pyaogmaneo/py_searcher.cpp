// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "py_searcher.h"

using namespace pyaon;

Searcher::Searcher(
    const std::tuple<int, int, int> &config_size,
    int num_dendrites,
    const std::string &file_name,
    const py::array_t<unsigned char> &buffer
) {
    if (buffer.unchecked().size() > 0)
        init_from_buffer(buffer);
    else if (!file_name.empty())
        init_from_file(file_name);
    else
        init_random(config_size, num_dendrites);

    // copy params
    params = searcher.params;
}

void Searcher::init_random(
    const std::tuple<int, int, int> &config_size,
    int num_dendrites
) {
    if (std::get<0>(config_size) < 1)
        throw std::runtime_error("error: config_size[0] < 1 is not allowed!");

    if (std::get<1>(config_size) < 1)
        throw std::runtime_error("error: config_size[1] < 1 is not allowed!");

    if (std::get<2>(config_size) < 1)
        throw std::runtime_error("error: config_size[2] < 1 is not allowed!");

    if (num_dendrites < 2)
        throw std::runtime_error("error: num_dendrites < 2 is not allowed!");

    searcher.init_random(aon::Int3(std::get<0>(config_size), std::get<1>(config_size), std::get<2>(config_size)), num_dendrites);
}

void Searcher::init_from_file(
    const std::string &file_name
) {
    File_Reader reader;
    reader.ins.open(file_name, std::ios::binary);

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != searcher_magic)
        throw std::runtime_error("attempted to initialize Searcher from incompatible file - " + file_name);

    searcher.read(reader);
}

void Searcher::init_from_buffer(
    const py::array_t<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    int magic;
    reader.read(&magic, sizeof(int));

    if (magic != searcher_magic)
        throw std::runtime_error("attempted to initialize Searcher from incompatible buffer!");

    searcher.read(reader);
}

void Searcher::save_to_file(
    const std::string &file_name
) {
    File_Writer writer;
    writer.outs.open(file_name, std::ios::binary);

    writer.write(&searcher_magic, sizeof(int));

    searcher.write(writer);
}

void Searcher::set_state_from_buffer(
    const py::array_t<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    searcher.read_state(reader);
}

void Searcher::set_weights_from_buffer(
    const py::array_t<unsigned char> &buffer
) {
    Buffer_Reader reader;
    reader.buffer = &buffer;

    searcher.read_weights(reader);
}

py::array_t<unsigned char> Searcher::serialize_to_buffer() {
    Buffer_Writer writer(searcher.size() + sizeof(int));

    writer.write(&searcher_magic, sizeof(int));

    searcher.write(writer);

    return writer.buffer;
}

py::array_t<unsigned char> Searcher::serialize_state_to_buffer() {
    Buffer_Writer writer(searcher.state_size());

    searcher.write_state(writer);

    return writer.buffer;
}

py::array_t<unsigned char> Searcher::serialize_weights_to_buffer() {
    Buffer_Writer writer(searcher.weights_size());

    searcher.write_weights(writer);

    return writer.buffer;
}

void Searcher::step(
    float reward,
    bool learn_enabled
) {
    // copy params
    searcher.params = params;

    searcher.step(reward, learn_enabled);
}

py::array_t<int> Searcher::get_config_cis() const {
    py::array_t<int> config_cis(searcher.get_hidden_cis().size());

    auto view = config_cis.mutable_unchecked();

    for (int j = 0; j < view.size(); j++)
        view(j) = searcher.get_config_cis()[j];

    return config_cis;
}

void Searcher::merge(
    const std::vector<Searcher*> &searchers,
    Merge_Mode mode
) {
    aon::Array<aon::Searcher*> c_searchers(searchers.size());

    for (int i = 0; i < searchers.size(); i++)
        c_searchers[i] = &searchers[i]->searcher;

    searcher.merge(c_searchers, static_cast<aon::Merge_Mode>(mode));
}
