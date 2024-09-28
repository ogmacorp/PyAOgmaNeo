// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "py_hierarchy.h"
#include <pybind11/operators.h>
#include <sstream>

namespace py = pybind11;

template<int S, int L>
void declare_for_S_L(
    py::module &m,
    const std::string &typestr
) {
    using Vec_Class = aon::Vec<S, L>;
    using Bundle_Class = aon::Bundle<S, L>;
    using Hierarchy_Class = pyaon::Hierarchy<S, L>;

    std::string vec_pyclass_name = std::string("Vec") + typestr;
    std::string bundle_pyclass_name = std::string("Bundle") + typestr;
    std::string hierarchy_pyclass_name = std::string("Hierarchy") + typestr;

    py::class_<Vec_Class>(m, vec_pyclass_name.c_str())
        .def_property_readonly_static("segments", [](){ return Vec_Class::segments(); })
        .def_property_readonly_static("length", [](){ return Vec_Class::length(); })
        .def_property_readonly_static("size", [](){ return Vec_Class::size(); })
        .def_static("randomized", [](){ return Vec_Class::randomized(); })
        .def("fill", &Vec_Class::fill)
        .def("__getitem__", [](const Vec_Class &v, int index){ return v[index]; })
        .def("__setitem__", [](Vec_Class &v, int index, aon::Byte value){ v[index] = value; })
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self / py::self)
        .def(py::self /= py::self)
        .def(py::self + py::self)
        .def("dot", &Vec_Class::dot)
        .def("permute", &Vec_Class::permute,
            py::arg("shift") = 1
        )
        .def("__copy__", 
            [](const Vec_Class &other) {
                return other;
            }
        )
        .def("__deepcopy__", 
            [](const Vec_Class &other) {
                return other;
            }
        )
        .def("__repr__",
            [](const Vec_Class &v) {
                std::ostringstream os;

                os << "[ ";

                for (int i = 0; i < v.segments(); i++)
                    os << static_cast<int>(v[i]) << " ";

                os << "]";

                return os.str();
            }
        );

    py::class_<Bundle_Class>(m, bundle_pyclass_name.c_str())
        .def_property_readonly_static("segments", [](){ return Vec_Class::segments(); })
        .def_property_readonly_static("length", [](){ return Vec_Class::length(); })
        .def_property_readonly_static("size", [](){ return Vec_Class::size(); })
        .def("fill", &Bundle_Class::fill)
        .def("__getitem__", [](const Bundle_Class &v, int index){ return v[index]; })
        .def("__setitem__", [](Bundle_Class &v, int index, aon::Byte value){ v[index] = value; })
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self + Vec_Class())
        .def(py::self += Vec_Class())
        .def("thin", &Bundle_Class::thin)
        .def("__copy__", 
            [](const Bundle_Class &other) {
                return other;
            }
        )
        .def("__deepcopy__", 
            [](const Bundle_Class &other) {
                return other;
            }
        )
        .def("__repr__",
            [](const Bundle_Class &b) {
                std::ostringstream os;

                os << "[ ";

                for (int i = 0; i < b.size(); i++)
                    os << b[i] << " ";

                os << "]";

                return os.str();
            }
        );


    py::class_<Hierarchy_Class>(m, hierarchy_pyclass_name.c_str())
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
        .def_readwrite("params", &Hierarchy_Class::params)
        .def("save_to_file", &Hierarchy_Class::save_to_file)
        .def("set_state_from_buffer", &Hierarchy_Class::set_state_from_buffer)
        .def("set_weights_from_buffer", &Hierarchy_Class::set_weights_from_buffer)
        .def("serialize_to_buffer", &Hierarchy_Class::serialize_to_buffer)
        .def("serialize_state_to_buffer", &Hierarchy_Class::serialize_state_to_buffer)
        .def("serialize_weights_to_buffer", &Hierarchy_Class::serialize_weights_to_buffer)
        .def("get_size", &Hierarchy_Class::get_size)
        .def("get_state_size", &Hierarchy_Class::get_state_size)
        .def("get_weights_size", &Hierarchy_Class::get_weights_size)
        .def("step", &Hierarchy_Class::step,
            py::arg("input_vecs"),
            py::arg("learn_enabled") = true
        )
        .def("clear_state", &Hierarchy_Class::clear_state)
        .def("get_num_layers", &Hierarchy_Class::get_num_layers)
        .def("get_prediction_vecs", &Hierarchy_Class::get_prediction_vecs)
        .def("get_hidden_vecs", &Hierarchy_Class::get_hidden_vecs)
        .def("get_hidden_size", &Hierarchy_Class::get_hidden_size)
        .def("get_update", &Hierarchy_Class::get_update)
        .def("get_ticks", &Hierarchy_Class::get_ticks)
        .def("get_ticks_per_update", &Hierarchy_Class::get_ticks_per_update)
        .def("get_temporal_horizon", &Hierarchy_Class::get_temporal_horizon)
        .def("get_num_visible_layers", &Hierarchy_Class::get_num_visible_layers)
        .def("get_num_io", &Hierarchy_Class::get_num_io)
        .def("get_io_size", &Hierarchy_Class::get_io_size)
        .def("get_io_type", &Hierarchy_Class::get_io_type)
        .def("get_radius", &Hierarchy_Class::get_radius)
        .def("__copy__", 
            [](const Hierarchy_Class &other) {
                return other;
            }
        )
        .def("__deepcopy__", 
            [](const Hierarchy_Class &other) {
                return other;
            }
        );
}

PYBIND11_MODULE(pyaogmaneo, m) {
    m.def("set_num_threads", &pyaon::set_num_threads);
    m.def("get_num_threads", &pyaon::get_num_threads);

    m.def("set_global_state", &pyaon::set_global_state);
    m.def("get_global_state", &pyaon::get_global_state);

    py::enum_<pyaon::IO_Type>(m, "IOType")
        .value("none", pyaon::none)
        .value("prediction", pyaon::prediction)
        .export_values();

    py::class_<pyaon::IO_Desc>(m, "IODesc")
        .def(py::init<
                std::tuple<int, int>,
                pyaon::IO_Type,
                int,
                float
            >(),
            py::arg("size") = std::tuple<int, int>({ 4, 4 }),
            py::arg("io_type") = pyaon::prediction,
            py::arg("radius") = 2,
            py::arg("positional_scale") = 1.0f
        )
        .def_readwrite("size", &pyaon::IO_Desc::size)
        .def_readwrite("io_type", &pyaon::IO_Desc::type)
        .def_readwrite("radius", &pyaon::IO_Desc::radius)
        .def_readwrite("positional_scale", &pyaon::IO_Desc::positional_scale)
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
                std::tuple<int, int>,
                int,
                int,
                int,
                int,
                int,
                float
            >(),
            py::arg("hidden_size") = std::tuple<int, int>({ 4, 4 }),
            py::arg("hidden_segments") = 8,
            py::arg("hidden_length") = 32,
            py::arg("radius") = 2,
            py::arg("ticks_per_update") = 2,
            py::arg("temporal_horizon") = 2,
            py::arg("positional_scale") = 1.0f
        )
        .def_readwrite("hidden_size", &pyaon::Layer_Desc::hidden_size)
        .def_readwrite("hidden_segments", &pyaon::Layer_Desc::hidden_segments)
        .def_readwrite("hidden_length", &pyaon::Layer_Desc::hidden_length)
        .def_readwrite("radius", &pyaon::Layer_Desc::radius)
        .def_readwrite("ticks_per_update", &pyaon::Layer_Desc::ticks_per_update)
        .def_readwrite("temporal_horizon", &pyaon::Layer_Desc::temporal_horizon)
        .def_readwrite("positional_scale", &pyaon::Layer_Desc::positional_scale)
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
    py::class_<aon::Layer_Params>(m, "LayerParams")
        .def(py::init<>())
        .def_readwrite("lr", &aon::Layer_Params::lr);
        .def_readwrite("lr", &aon::Layer_Params::lr);

    py::class_<aon::IO_Params>(m, "IOParams")
        .def(py::init<>());

    py::class_<pyaon::Params>(m, "Params")
        .def(py::init<>())
        .def_readwrite("layers", &pyaon::Params::layers)
        .def_readwrite("ios", &pyaon::Params::ios);

    // declare a bunch of sizes to use
    declare_for_S_L<32, 16>(m, "32_16");
    declare_for_S_L<64, 16>(m, "64_16");
    declare_for_S_L<128, 16>(m, "128_16");
    declare_for_S_L<256, 16>(m, "256_16");

    declare_for_S_L<32, 32>(m, "32_32");
    declare_for_S_L<64, 32>(m, "64_32");
    declare_for_S_L<128, 32>(m, "128_32");
    declare_for_S_L<256, 32>(m, "256_32");

    declare_for_S_L<32, 64>(m, "32_64");
    declare_for_S_L<64, 64>(m, "64_64");
    declare_for_S_L<128, 64>(m, "128_64");
    declare_for_S_L<256, 64>(m, "256_64");
}
