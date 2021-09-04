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

    m.def("setGlobalState", &pyaon::setGlobalState);
    m.def("getGlobalState", &pyaon::getGlobalState);

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
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("type") = pyaon::prediction,
            py::arg("ffRadius") = 2,
            py::arg("fbRadius") = 2,
            py::arg("historyCapacity") = 128
        )
        .def_readwrite("size", &pyaon::IODesc::size)
        .def_readwrite("type", &pyaon::IODesc::type)
        .def_readwrite("ffRadius", &pyaon::IODesc::ffRadius)
        .def_readwrite("fbRadius", &pyaon::IODesc::fbRadius)
        .def_readwrite("historyCapacity", &pyaon::IODesc::historyCapacity);

    py::class_<pyaon::LayerDesc>(m, "LayerDesc")
        .def(py::init<
                std::tuple<int, int, int>,
                int,
                int,
                int
            >(),
            py::arg("hiddenSize") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("ffRadius") = 2,
            py::arg("rRadius") = 2,
            py::arg("fbRadius") = 2
        )
        .def_readwrite("hiddenSize", &pyaon::LayerDesc::hiddenSize)
        .def_readwrite("ffRadius", &pyaon::LayerDesc::ffRadius)
        .def_readwrite("rRadius", &pyaon::LayerDesc::rRadius)
        .def_readwrite("fbRadius", &pyaon::LayerDesc::fbRadius);

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
            py::arg("reward") = 0.0f
        )
        .def("getNumLayers", &pyaon::Hierarchy::getNumLayers)
        .def("setImportance", &pyaon::Hierarchy::setImportance)
        .def("getImportance", &pyaon::Hierarchy::getImportance)
        .def("setRecurrence", &pyaon::Hierarchy::setRecurrence)
        .def("getRecurrence", &pyaon::Hierarchy::getRecurrence)
        .def("getPredictionCIs", &pyaon::Hierarchy::getPredictionCIs)
        .def("getHiddenCIs", &pyaon::Hierarchy::getHiddenCIs)
        .def("getHiddenSize", &pyaon::Hierarchy::getHiddenSize)
        .def("getNumEncVisibleLayers", &pyaon::Hierarchy::getNumEncVisibleLayers)
        .def("getNumInputs", &pyaon::Hierarchy::getNumInputs)
        .def("getInputSize", &pyaon::Hierarchy::getInputSize)
        .def("aLayerExists", &pyaon::Hierarchy::aLayerExists)
        .def("setELR", &pyaon::Hierarchy::setELR)
        .def("getELR", &pyaon::Hierarchy::getELR)
        .def("setDLR", &pyaon::Hierarchy::setDLR)
        .def("getDLR", &pyaon::Hierarchy::getDLR)
        .def("setALR", &pyaon::Hierarchy::setALR)
        .def("getALR", &pyaon::Hierarchy::getALR)
        .def("setADiscount", &pyaon::Hierarchy::setADiscount)
        .def("getADiscount", &pyaon::Hierarchy::getADiscount)
        .def("setAActionGap", &pyaon::Hierarchy::setAActionGap)
        .def("getAActionGap", &pyaon::Hierarchy::getAActionGap)
        .def("setAQSteps", &pyaon::Hierarchy::setAQSteps)
        .def("getAQSteps", &pyaon::Hierarchy::getAQSteps)
        .def("setAHistoryIters", &pyaon::Hierarchy::setAHistoryIters)
        .def("getAHistoryIters", &pyaon::Hierarchy::getAHistoryIters)
        .def("getFFRadius", &pyaon::Hierarchy::getFFRadius)
        .def("getRRadius", &pyaon::Hierarchy::getRRadius)
        .def("getFBRadius", &pyaon::Hierarchy::getFBRadius)
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
