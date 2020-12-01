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
        .value("none", pyaon::none)
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
            py::arg("type") = pyaon::none,
            py::arg("ffRadius") = 2,
            py::arg("pRadius") = 2,
            py::arg("aRadius") = 2,
            py::arg("historyCapacity") = 32
        )
        .def_readwrite("size", &pyaon::IODesc::size)
        .def_readwrite("type", &pyaon::IODesc::type)
        .def_readwrite("ffRadius", &pyaon::IODesc::ffRadius)
        .def_readwrite("pRadius", &pyaon::IODesc::pRadius)
        .def_readwrite("aRadius", &pyaon::IODesc::aRadius)
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
            py::arg("numPriorities") = 3,
            py::arg("ffRadius") = 2,
            py::arg("pRadius") = 2,
            py::arg("ticksPerUpdate") = 2,
            py::arg("temporalHorizon") = 2
        )
        .def_readwrite("hiddenSize", &pyaon::LayerDesc::hiddenSize)
        .def_readwrite("numPriorities", &pyaon::LayerDesc::numPriorities)
        .def_readwrite("ffRadius", &pyaon::LayerDesc::ffRadius)
        .def_readwrite("pRadius", &pyaon::LayerDesc::pRadius)
        .def_readwrite("ticksPerUpdate", &pyaon::LayerDesc::ticksPerUpdate)
        .def_readwrite("temporalHorizon", &pyaon::LayerDesc::temporalHorizon);

    py::class_<pyaon::Hierarchy>(m, "Hierarchy")
        .def(py::init<>())
        .def("initRandom", &pyaon::Hierarchy::initRandom)
        .def("initFromFile", &pyaon::Hierarchy::initFromFile)
        .def("initFromBuffer", &pyaon::Hierarchy::initFromBuffer)
        .def("saveToFile", &pyaon::Hierarchy::saveToFile)
        .def("serializeToBuffer", &pyaon::Hierarchy::serializeToBuffer)
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
        .def("getNumSCVisibleLayers", &pyaon::Hierarchy::getNumSCVisibleLayers)
        .def("getNumInputs", &pyaon::Hierarchy::getNumInputs)
        .def("getInputSize", &pyaon::Hierarchy::getInputSize)
        .def("pLayerExists", &pyaon::Hierarchy::pLayerExists)
        .def("aLayerExists", &pyaon::Hierarchy::aLayerExists)
        .def("setSCAlpha", &pyaon::Hierarchy::setSCAlpha)
        .def("getSCAlpha", &pyaon::Hierarchy::getSCAlpha)
        .def("setPAlpha", &pyaon::Hierarchy::setPAlpha)
        .def("getPAlpha", &pyaon::Hierarchy::getPAlpha)
        .def("setPTargetRange", &pyaon::Hierarchy::setPTargetRange)
        .def("getPTargetRange", &pyaon::Hierarchy::getPTargetRange)
        .def("setAAlpha", &pyaon::Hierarchy::setAAlpha)
        .def("getAAlpha", &pyaon::Hierarchy::getAAlpha)
        .def("setABeta", &pyaon::Hierarchy::setABeta)
        .def("getABeta", &pyaon::Hierarchy::getABeta)
        .def("setAGamma", &pyaon::Hierarchy::setAGamma)
        .def("getAGamma", &pyaon::Hierarchy::getAGamma)
        .def("setAMinSteps", &pyaon::Hierarchy::setAMinSteps)
        .def("getAMinSteps", &pyaon::Hierarchy::getAMinSteps)
        .def("setAHistoryIters", &pyaon::Hierarchy::setAHistoryIters)
        .def("getAHistoryIters", &pyaon::Hierarchy::getAHistoryIters)
        .def("getFFRadius", &pyaon::Hierarchy::getFFRadius)
        .def("getPRadius", &pyaon::Hierarchy::getPRadius)
        .def("getARadius", &pyaon::Hierarchy::getARadius)
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
        .def("setAlpha", &pyaon::ImageEncoder::setAlpha)
        .def("getAlpha", &pyaon::ImageEncoder::getAlpha);
}
