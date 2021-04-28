// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PyHierarchy.h"
#include "PyImageEncoder.h"

namespace py = pybind11;

PYBIND11_MODULE(pyaogmaneo, m) {
    m.def("setNumThreads", &pyaon::setNumThreads);
    m.def("getNumThreads", &pyaon::getNumThreads);

    py::enum_<pyaon::IOType>(m, "IOType")
        .value("prediction", pyaon::prediction)
        .value("action", pyaon::action)
        .export_values();

    py::class_<pyaon::IODesc>(m, "IODesc")
        .def(py::init<
                std::tuple<int, int, int>,
                pyaon::IOType,
                int,
                int,
                int,
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("type") = pyaon::prediction,
            py::arg("eRadius") = 2,
            py::arg("dRadius") = 2,
            py::arg("bRadius") = 2,
            py::arg("historyCapacity") = 128
        )
        .def_readwrite("size", &pyaon::IODesc::size)
        .def_readwrite("type", &pyaon::IODesc::type)
        .def_readwrite("eRadius", &pyaon::IODesc::eRadius)
        .def_readwrite("dRadius", &pyaon::IODesc::dRadius)
        .def_readwrite("bRadius", &pyaon::IODesc::bRadius)
        .def_readwrite("historyCapacity", &pyaon::IODesc::historyCapacity);

    py::class_<pyaon::LayerDesc>(m, "LayerDesc")
        .def(py::init<
                std::tuple<int, int, int>,
                int,
                int,
                int,
                int,
                int
            >(),
            py::arg("hiddenSize") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("eRadius") = 2,
            py::arg("dRadius") = 2,
            py::arg("bRadius") = 2,
            py::arg("ticksPerUpdate") = 2,
            py::arg("temporalHorizon") = 2
        )
        .def_readwrite("hiddenSize", &pyaon::LayerDesc::hiddenSize)
        .def_readwrite("eRadius", &pyaon::LayerDesc::eRadius)
        .def_readwrite("dRadius", &pyaon::LayerDesc::dRadius)
        .def_readwrite("bRadius", &pyaon::LayerDesc::bRadius)
        .def_readwrite("ticksPerUpdate", &pyaon::LayerDesc::ticksPerUpdate)
        .def_readwrite("temporalHorizon", &pyaon::LayerDesc::temporalHorizon);

    py::class_<pyaon::Hierarchy>(m, "Hierarchy")
        .def(py::init<>())
        .def("initRandom", &pyaon::Hierarchy::initRandom)
        .def("initFromFile", &pyaon::Hierarchy::initFromFile)
        .def("initFromBuffer", &pyaon::Hierarchy::initFromBuffer)
        .def("saveToFile", &pyaon::Hierarchy::saveToFile)
        .def("serializeToBuffer", &pyaon::Hierarchy::serializeToBuffer)
        .def("setStateFromBuffer", &pyaon::Hierarchy::setStateFromBuffer)
        .def("serializeStateToBuffer", &pyaon::Hierarchy::serializeStateToBuffer)
        .def("step", &pyaon::Hierarchy::step,
            py::arg("inputCIs"),
            py::arg("learnEnabled") = true,
            py::arg("reward") = 0.0f,
            py::arg("mimic") = false
        )
        .def("getNumLayers", &pyaon::Hierarchy::getNumLayers)
        .def("getPredictionCIs", &pyaon::Hierarchy::getPredictionCIs)
        .def("getUpdate", &pyaon::Hierarchy::getUpdate)
        .def("getHiddenCIs", &pyaon::Hierarchy::getHiddenCIs)
        .def("getHiddenSize", &pyaon::Hierarchy::getHiddenSize)
        .def("getTicks", &pyaon::Hierarchy::getTicks)
        .def("getTicksPerUpdate", &pyaon::Hierarchy::getTicksPerUpdate)
        .def("getNumEncVisibleLayers", &pyaon::Hierarchy::getNumEncVisibleLayers)
        .def("getNumInputs", &pyaon::Hierarchy::getNumInputs)
        .def("getInputSize", &pyaon::Hierarchy::getInputSize)
        .def("aLayerExists", &pyaon::Hierarchy::aLayerExists)
        .def("setELR", &pyaon::Hierarchy::setELR)
        .def("getELR", &pyaon::Hierarchy::getELR)
        .def("setDLR", &pyaon::Hierarchy::setDLR)
        .def("getDLR", &pyaon::Hierarchy::getDLR)
        .def("setAVLR", &pyaon::Hierarchy::setAVLR)
        .def("getAVLR", &pyaon::Hierarchy::getAVLR)
        .def("setAALR", &pyaon::Hierarchy::setAALR)
        .def("getAALR", &pyaon::Hierarchy::getAALR)
        .def("setADiscount", &pyaon::Hierarchy::setADiscount)
        .def("getADiscount", &pyaon::Hierarchy::getADiscount)
        .def("getERadius", &pyaon::Hierarchy::getERadius)
        .def("getDRadius", &pyaon::Hierarchy::getDRadius)
        .def("getBRadius", &pyaon::Hierarchy::getBRadius)
        .def("getAHistoryCapacity", &pyaon::Hierarchy::getAHistoryCapacity);

    py::class_<pyaon::ImageEncoderVisibleLayerDesc>(m, "ImageEncoderVisibleLayerDesc")
        .def(py::init<
                std::tuple<int, int, int>,
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("radius") = 4
        )
        .def_readwrite("size", &pyaon::ImageEncoderVisibleLayerDesc::size)
        .def_readwrite("radius", &pyaon::ImageEncoderVisibleLayerDesc::radius);
        
    py::class_<pyaon::ImageEncoder>(m, "ImageEncoder")
        .def(py::init<>())
        .def("initRandom", &pyaon::ImageEncoder::initRandom)
        .def("initFromFile", &pyaon::ImageEncoder::initFromFile)
        .def("initFromBuffer", &pyaon::ImageEncoder::initFromBuffer)
        .def("saveToFile", &pyaon::ImageEncoder::saveToFile)
        .def("serializeToBuffer", &pyaon::ImageEncoder::serializeToBuffer)
        .def("step", &pyaon::ImageEncoder::step,
            py::arg("inputs"),
            py::arg("learnEnabled") = true
        )
        .def("reconstruct", &pyaon::ImageEncoder::reconstruct)
        .def("getNumVisibleLayers", &pyaon::ImageEncoder::getNumVisibleLayers)
        .def("getReconstruction", &pyaon::ImageEncoder::getReconstruction)
        .def("getHiddenCIs", &pyaon::ImageEncoder::getHiddenCIs)
        .def("getHiddenSize", &pyaon::ImageEncoder::getHiddenSize)
        .def("getVisibleSize", &pyaon::ImageEncoder::getVisibleSize)
        .def("setLR", &pyaon::ImageEncoder::setLR)
        .def("getLR", &pyaon::ImageEncoder::getLR)
        .def("setFalloff", &pyaon::ImageEncoder::setFalloff)
        .def("getFalloff", &pyaon::ImageEncoder::getFalloff);
}
