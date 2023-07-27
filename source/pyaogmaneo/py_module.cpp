// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_hierarchy.h"
#include "py_image_encoder.h"

namespace py = pybind11;

PYBIND11_MODULE(pyaogmaneo, m) {
    m.def("set_num_threads", &pyaon::set_num_threads);
    m.def("get_num_threads", &pyaon::get_num_threads);

    m.def("set_global_state", &pyaon::set_global_state);
    m.def("get_global_state", &pyaon::get_global_state);

    py::enum_<pyaon::IO_Type>(m, "IOType")
        .value("none", pyaon::none)
        .value("prediction", pyaon::prediction)
        .value("action", pyaon::action)
        .export_values();

    py::class_<pyaon::IO_Desc>(m, "IODesc")
        .def(py::init<
                std::tuple<int, int, int>,
                pyaon::IO_Type,
                int,
                int,
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("io_type") = pyaon::prediction,
            py::arg("up_radius") = 2,
            py::arg("down_radius") = 2,
            py::arg("history_capacity") = 64
        )
        .def_readwrite("size", &pyaon::IO_Desc::size)
        .def_readwrite("io_type", &pyaon::IO_Desc::type)
        .def_readwrite("up_radius", &pyaon::IO_Desc::up_radius)
        .def_readwrite("down_radius", &pyaon::IO_Desc::down_radius)
        .def_readwrite("history_capacity", &pyaon::IO_Desc::history_capacity);

    py::class_<pyaon::Layer_Desc>(m, "LayerDesc")
        .def(py::init<
                std::tuple<int, int, int>,
                int,
                int,
                int,
                int
            >(),
            py::arg("hidden_size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("up_radius") = 2,
            py::arg("down_radius") = 2,
            py::arg("ticks_per_update") = 2,
            py::arg("temporal_horizon") = 2
        )
        .def_readwrite("hidden_size", &pyaon::Layer_Desc::hidden_size)
        .def_readwrite("up_radius", &pyaon::Layer_Desc::up_radius)
        .def_readwrite("down_radius", &pyaon::Layer_Desc::down_radius)
        .def_readwrite("ticks_per_update", &pyaon::Layer_Desc::ticks_per_update)
        .def_readwrite("temporal_horizon", &pyaon::Layer_Desc::temporal_horizon);

    // bind params
    py::class_<aon::Encoder::Params>(m, "EncoderParams")
        .def(py::init<>())
        .def_readwrite("lr", &aon::Encoder::Params::lr);

    py::class_<aon::Decoder::Params>(m, "DecoderParams")
        .def(py::init<>())
        .def_readwrite("temperature", &aon::Decoder::Params::temperature)
        .def_readwrite("lr", &aon::Decoder::Params::lr);

    py::class_<aon::Actor::Params>(m, "ActorParams")
        .def(py::init<>())
        .def_readwrite("vlr", &aon::Actor::Params::vlr)
        .def_readwrite("alr", &aon::Actor::Params::alr)
        .def_readwrite("bias", &aon::Actor::Params::bias)
        .def_readwrite("discount", &aon::Actor::Params::discount)
        .def_readwrite("temperature", &aon::Actor::Params::temperature)
        .def_readwrite("min_steps", &aon::Actor::Params::min_steps)
        .def_readwrite("history_iters", &aon::Actor::Params::history_iters);

    py::class_<aon::Hierarchy::Layer_Params>(m, "LayerParams")
        .def(py::init<>())
        .def_readwrite("encoder", &aon::Hierarchy::Layer_Params::encoder)
        .def_readwrite("decoder", &aon::Hierarchy::Layer_Params::decoder);

    py::class_<aon::Hierarchy::IO_Params>(m, "IOParams")
        .def(py::init<>())
        .def_readwrite("decoder", &aon::Hierarchy::IO_Params::decoder)
        .def_readwrite("actor", &aon::Hierarchy::IO_Params::actor)
        .def_readwrite("importance", &aon::Hierarchy::IO_Params::importance);

    py::class_<pyaon::Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("layers", &pyaon::Params::layers)
        .def_readwrite("ios", &pyaon::Params::ios);

    py::class_<pyaon::Hierarchy>(m, "Hierarchy")
        .def(py::init<
                const std::vector<pyaon::IO_Desc>&,
                const std::vector<pyaon::Layer_Desc>&,
                const std::string&,
                const std::vector<unsigned char>&
            >(),
            py::arg("io_descs") = std::vector<pyaon::IO_Desc>(),
            py::arg("layer_descs") = std::vector<pyaon::Layer_Desc>(),
            py::arg("file_name") = std::string(),
            py::arg("buffer") = std::vector<unsigned char>()
        )
        .def_readwrite("params", &pyaon::Hierarchy::params)
        .def("save_to_file", &pyaon::Hierarchy::save_to_file)
        .def("serialize_to_buffer", &pyaon::Hierarchy::serialize_to_buffer)
        .def("set_state_from_buffer", &pyaon::Hierarchy::set_state_from_buffer)
        .def("serialize_state_to_buffer", &pyaon::Hierarchy::serialize_state_to_buffer)
        .def("step", &pyaon::Hierarchy::step,
            py::arg("input_cis"),
            py::arg("learn_enabled") = true,
            py::arg("reward") = 0.0f,
            py::arg("mimic") = 0.0f
        )
        .def("clear_state", &pyaon::Hierarchy::clear_state)
        .def("get_num_layers", &pyaon::Hierarchy::get_num_layers)
        .def("get_prediction_cis", &pyaon::Hierarchy::get_prediction_cis)
        .def("get_layer_prediction_cis", &pyaon::Hierarchy::get_layer_prediction_cis)
        .def("get_prediction_probs", &pyaon::Hierarchy::get_prediction_probs)
        .def("sample_prediction", &pyaon::Hierarchy::sample_prediction)
        .def("get_hidden_cis", &pyaon::Hierarchy::get_hidden_cis)
        .def("get_hidden_size", &pyaon::Hierarchy::get_hidden_size)
        .def("get_num_encoder_visible_layers", &pyaon::Hierarchy::get_num_encoder_visible_layers)
        .def("get_ticks", &pyaon::Hierarchy::get_ticks)
        .def("get_ticks_per_update", &pyaon::Hierarchy::get_ticks_per_update)
        .def("get_num_io", &pyaon::Hierarchy::get_num_io)
        .def("get_io_size", &pyaon::Hierarchy::get_io_size)
        .def("get_io_type", &pyaon::Hierarchy::get_io_type)
        .def("get_up_radius", &pyaon::Hierarchy::get_up_radius)
        .def("get_down_radius", &pyaon::Hierarchy::get_down_radius)
        .def("get_actor_history_capacity", &pyaon::Hierarchy::get_actor_history_capacity)
        .def("get_encoder_receptive_field", &pyaon::Hierarchy::get_encoder_receptive_field)
        .def("get_decoder_receptive_field", &pyaon::Hierarchy::get_decoder_receptive_field);

    py::class_<pyaon::Image_Visible_Layer_Desc>(m, "ImageVisibleLayerDesc")
        .def(py::init<
                std::tuple<int, int, int>,
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("radius") = 4
        )
        .def_readwrite("size", &pyaon::Image_Visible_Layer_Desc::size)
        .def_readwrite("radius", &pyaon::Image_Visible_Layer_Desc::radius);

    // bind params
    py::class_<aon::Image_Encoder::Params>(m, "ImageEncoderParams")
        .def(py::init<>())
        .def_readwrite("threshold", &aon::Image_Encoder::Params::threshold)
        .def_readwrite("falloff", &aon::Image_Encoder::Params::falloff)
        .def_readwrite("lr", &aon::Image_Encoder::Params::lr)
        .def_readwrite("rr", &aon::Image_Encoder::Params::rr);

    py::class_<pyaon::Image_Encoder>(m, "ImageEncoder")
        .def(py::init<
                const std::tuple<int, int, int>&,
                const std::vector<pyaon::Image_Visible_Layer_Desc>&,
                const std::string&,
                const std::vector<unsigned char>&
            >(),
            py::arg("hidden_size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("visible_layer_descs") = std::vector<pyaon::Image_Visible_Layer_Desc>(),
            py::arg("file_name") = std::string(),
            py::arg("buffer") = std::vector<unsigned char>()
        )
        .def_readwrite("params", &pyaon::Image_Encoder::params)
        .def("save_to_file", &pyaon::Image_Encoder::save_to_file)
        .def("serialize_to_buffer", &pyaon::Image_Encoder::serialize_to_buffer)
        .def("step", &pyaon::Image_Encoder::step,
            py::arg("inputs"),
            py::arg("learn_enabled") = true
        )
        .def("reconstruct", &pyaon::Image_Encoder::reconstruct)
        .def("get_num_visible_layers", &pyaon::Image_Encoder::get_num_visible_layers)
        .def("get_reconstruction", &pyaon::Image_Encoder::get_reconstruction)
        .def("get_hidden_cis", &pyaon::Image_Encoder::get_hidden_cis)
        .def("get_hidden_size", &pyaon::Image_Encoder::get_hidden_size)
        .def("get_visible_size", &pyaon::Image_Encoder::get_visible_size)
        .def("get_receptive_field", &pyaon::Image_Encoder::get_receptive_field);
}
