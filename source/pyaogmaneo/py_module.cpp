// ----------------------------------------------------------------------------
//  Py_aOgma_neo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of Py_aOgma_neo is licensed to you under the terms described
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

    py::enum_<pyaon::IO_Type>(m, "IO_Type")
        .value("none", pyaon::none)
        .value("prediction", pyaon::prediction)
        .value("action", pyaon::action)
        .export_values();

    py::class_<pyaon::IO_Desc>(m, "IO_Desc")
        .def(py::init<
                std::tuple<int, int, int>,
                pyaon::IO_Type,
                int,
                int,
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("type") = pyaon::prediction,
            py::arg("up_radius") = 2,
            py::arg("down_radius") = 2,
            py::arg("history_capacity") = 64
        )
        .def_readwrite("size", &pyaon::IO_Desc::size)
        .def_readwrite("type", &pyaon::IO_Desc::type)
        .def_readwrite("up_radius", &pyaon::IO_Desc::up_radius)
        .def_readwrite("down_radius", &pyaon::IO_Desc::down_radius)
        .def_readwrite("history_capacity", &pyaon::IO_Desc::history_capacity);

    py::class_<pyaon::Layer_desc>(m, "Layer_desc")
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
        .def_readwrite("hidden_size", &pyaon::Layer_desc::hidden_size)
        .def_readwrite("up_radius", &pyaon::Layer_desc::up_radius)
        .def_readwrite("down_radius", &pyaon::Layer_desc::down_radius)
        .def_readwrite("ticks_per_update", &pyaon::Layer_desc::ticks_per_update)
        .def_readwrite("temporal_horizon", &pyaon::Layer_desc::temporal_horizon);

    py::class_<pyaon::Hierarchy>(m, "Hierarchy")
        .def(py::init<
                const std::vector<pyaon::IO_Desc>&,
                const std::vector<pyaon::Layer_Desc>&,
                const std::string&,
                const std::vector<unsigned char>&
            >(),
            py::arg("io_descs") = std::vector<pyaon::IO_Desc>(),
            py::arg("layer_descs") = std::vector<pyaon::Layer_Desc>(),
            py::arg("name") = std::string(),
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
        .def("clear_state", &pyaon::hierarchy::clear_state)
        .def("get_num_layers", &pyaon::hierarchy::get_num_layers)
        .def("set_input_importance", &pyaon::hierarchy::set_input_importance)
        .def("get_input_importance", &pyaon::hierarchy::get_input_importance)
        .def("get_prediction_cis", &pyaon::hierarchy::get_prediction_cis)
        .def("get_hidden_cis", &pyaon::hierarchy::get_hidden_cis)
        .def("get_hidden_size", &pyaon::hierarchy::get_hidden_size)
        .def("get_num_encoder_visible_layers", &pyaon::hierarchy::get_num_encoder_visible_layers)
        .def("get_ticks", &pyaon::hierarchy::get_ticks)
        .def("get_ticks_per_update", &pyaon::hierarchy::get_ticks_per_update)
        .def("get_num_io", &pyaon::hierarchy::get_num_io)
        .def("get_io_size", &pyaon::hierarchy::get_io_size)
        .def("get_io_type", &pyaon::hierarchy::get_io_type)
        .def("get_up_radius", &pyaon::hierarchy::get_up_radius)
        .def("get_down_radius", &pyaon::hierarchy::get_down_radius)
        .def("get_actor_history_capacity", &pyaon::hierarchy::get_actor_history_capacity);

    py::class_<pyaon::image_encoder_visible_layer_desc>(m, "image_encoder_visible_layer_desc")
        .def(py::init<
                std::tuple<int, int, int>,
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("radius") = 4
        )
        .def_readwrite("size", &pyaon::image_encoder_visible_layer_desc::size)
        .def_readwrite("radius", &pyaon::image_encoder_visible_layer_desc::radius);

    py::class_<pyaon::Image_Encoder>(m, "Image_Encoder")
        .def(py::init<
                const std::tuple<int, int, int>&,
                const std::vector<pyaon::Image_Visible_Layer_Desc>&,
                const std::string&,
                const std::vector<unsigned char>&
            >(),
            py::arg("hidden_size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("visible_layer_descs") = std::vector<pyaon::Image_Visible_Layer_Desc>(),
            py::arg("name") = std::string(),
            py::arg("buffer") = std::vector<unsigned char>()
        )
        .def_readwrite("params", &pyaon::Image_Encoder::params)
        .def("save_to_file", &pyaon::image_encoder::save_to_file)
        .def("serialize_to_buffer", &pyaon::image_encoder::serialize_to_buffer)
        .def("step", &pyaon::image_encoder::step,
            py::arg("inputs"),
            py::arg("learn_enabled") = true
        )
        .def("reconstruct", &pyaon::image_encoder::reconstruct)
        .def("get_num_visible_layers", &pyaon::image_encoder::get_num_visible_layers)
        .def("get_reconstruction", &pyaon::image_encoder::get_reconstruction)
        .def("get_hidden_cis", &pyaon::image_encoder::get_hidden_cis)
        .def("get_hidden_size", &pyaon::image_encoder::get_hidden_size)
        .def("get_visible_size", &pyaon::image_encoder::get_visible_size);
}
