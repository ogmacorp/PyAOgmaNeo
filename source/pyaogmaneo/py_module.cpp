// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

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

    py::enum_<pyaon::Merge_Mode>(m, "MergeMode")
        .value("merge_random", pyaon::merge_random)
        .value("merge_average", pyaon::merge_average)
        .export_values();

    py::class_<pyaon::IO_Desc>(m, "IODesc")
        .def(py::init<
                std::tuple<int, int, int>,
                pyaon::IO_Type,
                int,
                int,
                int,
                int,
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("io_type") = pyaon::prediction,
            py::arg("num_dendrites_per_cell") = 4,
            py::arg("value_num_dendrites_per_cell") = 8,
            py::arg("up_radius") = 2,
            py::arg("down_radius") = 2,
            py::arg("history_capacity") = 512
        )
        .def_readwrite("size", &pyaon::IO_Desc::size)
        .def_readwrite("io_type", &pyaon::IO_Desc::type)
        .def_readwrite("num_dendrites_per_cell", &pyaon::IO_Desc::num_dendrites_per_cell)
        .def_readwrite("value_num_dendrites_per_cell", &pyaon::IO_Desc::value_num_dendrites_per_cell)
        .def_readwrite("up_radius", &pyaon::IO_Desc::up_radius)
        .def_readwrite("down_radius", &pyaon::IO_Desc::down_radius)
        .def_readwrite("history_capacity", &pyaon::IO_Desc::history_capacity)
        .def("__copy__", 
            [](const pyaon::IO_Desc &other) {
                return other;
            }
        )
        .def("__deepcopy__", 
            [](const pyaon::IO_Desc &other) {
                return other;
            }
        );

    py::class_<pyaon::Layer_Desc>(m, "LayerDesc")
        .def(py::init<
                std::tuple<int, int, int>,
                int,
                int,
                int,
                int
            >(),
            py::arg("hidden_size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("num_dendrites_per_cell") = 4,
            py::arg("up_radius") = 2,
            py::arg("recurrent_radius") = 0,
            py::arg("down_radius") = 2
        )
        .def_readwrite("hidden_size", &pyaon::Layer_Desc::hidden_size)
        .def_readwrite("num_dendrites_per_cell", &pyaon::Layer_Desc::num_dendrites_per_cell)
        .def_readwrite("up_radius", &pyaon::Layer_Desc::up_radius)
        .def_readwrite("recurrent_radius", &pyaon::Layer_Desc::recurrent_radius)
        .def_readwrite("down_radius", &pyaon::Layer_Desc::down_radius)
        .def("__copy__", 
            [](const pyaon::Layer_Desc &other) {
                return other;
            }
        )
        .def("__deepcopy__", 
            [](const pyaon::Layer_Desc &other) {
                return other;
            }
        );

    // bind params
    py::class_<aon::Encoder::Params>(m, "EncoderParams")
        .def(py::init<>())
        .def_readwrite("choice", &aon::Encoder::Params::choice)
        .def_readwrite("mismatch", &aon::Encoder::Params::mismatch)
        .def_readwrite("lr", &aon::Encoder::Params::lr)
        .def_readwrite("active_ratio", &aon::Encoder::Params::active_ratio)
        .def_readwrite("l_radius", &aon::Encoder::Params::l_radius);

    py::class_<aon::Decoder::Params>(m, "DecoderParams")
        .def(py::init<>())
        .def_readwrite("scale", &aon::Decoder::Params::scale)
        .def_readwrite("lr", &aon::Decoder::Params::lr)
        .def_readwrite("leak", &aon::Decoder::Params::leak);

    py::class_<aon::Actor::Params>(m, "ActorParams")
        .def(py::init<>())
        .def_readwrite("vlr", &aon::Actor::Params::vlr)
        .def_readwrite("plr", &aon::Actor::Params::plr)
        .def_readwrite("leak", &aon::Actor::Params::leak)
        .def_readwrite("smoothing", &aon::Actor::Params::smoothing)
        .def_readwrite("delay_rate", &aon::Actor::Params::delay_rate)
        .def_readwrite("policy_clip", &aon::Actor::Params::policy_clip)
        .def_readwrite("discount", &aon::Actor::Params::discount)
        .def_readwrite("td_scale_decay", &aon::Actor::Params::td_scale_decay)
        .def_readwrite("min_steps", &aon::Actor::Params::min_steps)
        .def_readwrite("history_iters", &aon::Actor::Params::history_iters);

    py::class_<aon::Hierarchy::Layer_Params>(m, "LayerParams")
        .def(py::init<>())
        .def_readwrite("encoder", &aon::Hierarchy::Layer_Params::encoder)
        .def_readwrite("decoder", &aon::Hierarchy::Layer_Params::decoder)
        .def_readwrite("recurrent_importance", &aon::Hierarchy::Layer_Params::recurrent_importance);

    py::class_<aon::Hierarchy::IO_Params>(m, "IOParams")
        .def(py::init<>())
        .def_readwrite("decoder", &aon::Hierarchy::IO_Params::decoder)
        .def_readwrite("actor", &aon::Hierarchy::IO_Params::actor)
        .def_readwrite("importance", &aon::Hierarchy::IO_Params::importance);

    py::class_<pyaon::Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("layers", &pyaon::Params::layers)
        .def_readwrite("ios", &pyaon::Params::ios)
        .def_readwrite("anticipation", &pyaon::Params::anticipation);

    py::class_<pyaon::Hierarchy>(m, "Hierarchy")
        .def(py::init<
                const std::vector<pyaon::IO_Desc>&,
                const std::vector<pyaon::Layer_Desc>&,
                const std::string&,
                const py::array_t<unsigned char>&
            >(),
            py::arg("io_descs") = std::vector<pyaon::IO_Desc>(),
            py::arg("layer_descs") = std::vector<pyaon::Layer_Desc>(),
            py::arg("file_name") = std::string(),
            py::arg("buffer") = py::array_t<unsigned char>()
        )
        .def_readwrite("params", &pyaon::Hierarchy::params)
        .def("save_to_file", &pyaon::Hierarchy::save_to_file)
        .def("set_state_from_buffer", &pyaon::Hierarchy::set_state_from_buffer)
        .def("set_weights_from_buffer", &pyaon::Hierarchy::set_weights_from_buffer)
        .def("serialize_to_buffer", &pyaon::Hierarchy::serialize_to_buffer)
        .def("serialize_state_to_buffer", &pyaon::Hierarchy::serialize_state_to_buffer)
        .def("serialize_weights_to_buffer", &pyaon::Hierarchy::serialize_weights_to_buffer)
        .def("get_size", &pyaon::Hierarchy::get_size)
        .def("get_state_size", &pyaon::Hierarchy::get_state_size)
        .def("get_weights_size", &pyaon::Hierarchy::get_weights_size)
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
        .def("get_prediction_acts", &pyaon::Hierarchy::get_prediction_acts)
        .def("sample_prediction", &pyaon::Hierarchy::sample_prediction)
        .def("get_hidden_cis", &pyaon::Hierarchy::get_hidden_cis)
        .def("get_hidden_size", &pyaon::Hierarchy::get_hidden_size)
        .def("get_num_encoder_visible_layers", &pyaon::Hierarchy::get_num_encoder_visible_layers)
        .def("get_num_io", &pyaon::Hierarchy::get_num_io)
        .def("get_io_size", &pyaon::Hierarchy::get_io_size)
        .def("get_io_type", &pyaon::Hierarchy::get_io_type)
        .def("get_up_radius", &pyaon::Hierarchy::get_up_radius)
        .def("get_down_radius", &pyaon::Hierarchy::get_down_radius)
        .def("merge", &pyaon::Hierarchy::merge)
        .def("__copy__", 
            [](const pyaon::Hierarchy &other) {
                return other;
            }
        )
        .def("__deepcopy__", 
            [](const pyaon::Hierarchy &other) {
                return other;
            }
        );

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
        .def_readwrite("choice", &aon::Image_Encoder::Params::choice)
        .def_readwrite("vigilance", &aon::Image_Encoder::Params::vigilance)
        .def_readwrite("lr", &aon::Image_Encoder::Params::lr)
        .def_readwrite("scale", &aon::Image_Encoder::Params::scale)
        .def_readwrite("rr", &aon::Image_Encoder::Params::rr)
        .def_readwrite("active_ratio", &aon::Image_Encoder::Params::active_ratio)
        .def_readwrite("l_radius", &aon::Image_Encoder::Params::l_radius);

    py::class_<pyaon::Image_Encoder>(m, "ImageEncoder")
        .def(py::init<
                const std::tuple<int, int, int>&,
                const std::vector<pyaon::Image_Visible_Layer_Desc>&,
                const std::string&,
                const py::array_t<unsigned char>&
            >(),
            py::arg("hidden_size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("visible_layer_descs") = std::vector<pyaon::Image_Visible_Layer_Desc>(),
            py::arg("file_name") = std::string(),
            py::arg("buffer") = py::array_t<unsigned char>()
        )
        .def_readwrite("params", &pyaon::Image_Encoder::params)
        .def("save_to_file", &pyaon::Image_Encoder::save_to_file)
        .def("set_state_from_buffer", &pyaon::Image_Encoder::set_state_from_buffer)
        .def("set_weights_from_buffer", &pyaon::Image_Encoder::set_weights_from_buffer)
        .def("serialize_to_buffer", &pyaon::Image_Encoder::serialize_to_buffer)
        .def("serialize_state_to_buffer", &pyaon::Image_Encoder::serialize_state_to_buffer)
        .def("serialize_weights_to_buffer", &pyaon::Image_Encoder::serialize_weights_to_buffer)
        .def("get_size", &pyaon::Image_Encoder::get_size)
        .def("get_state_size", &pyaon::Image_Encoder::get_state_size)
        .def("get_weights_size", &pyaon::Image_Encoder::get_weights_size)
        .def("step", &pyaon::Image_Encoder::step,
            py::arg("inputs"),
            py::arg("learn_enabled") = true,
            py::arg("learn_recon") = true
        )
        .def("reconstruct", &pyaon::Image_Encoder::reconstruct)
        .def("get_num_visible_layers", &pyaon::Image_Encoder::get_num_visible_layers)
        .def("get_reconstruction", &pyaon::Image_Encoder::get_reconstruction)
        .def("get_hidden_cis", &pyaon::Image_Encoder::get_hidden_cis)
        .def("get_hidden_size", &pyaon::Image_Encoder::get_hidden_size)
        .def("get_visible_size", &pyaon::Image_Encoder::get_visible_size)
        .def("merge", &pyaon::Image_Encoder::merge)
        .def("__copy__", 
            [](const pyaon::Image_Encoder &other) {
                return other;
            }
        )
        .def("__deepcopy__", 
            [](const pyaon::Image_Encoder &other) {
                return other;
            }
        );
}
