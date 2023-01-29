// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
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
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("type") = pyaon::prediction,
            py::arg("eRadius") = 2,
            py::arg("dRadius") = 2,
            py::arg("historyCapacity") = 64
        )
        .def_readwrite("size", &pyaon::IODesc::size)
        .def_readwrite("type", &pyaon::IODesc::type)
        .def_readwrite("eRadius", &pyaon::IODesc::eRadius)
        .def_readwrite("dRadius", &pyaon::IODesc::dRadius)
        .def_readwrite("historyCapacity", &pyaon::IODesc::historyCapacity);

    py::class_<pyaon::LayerDesc>(m, "LayerDesc")
        .def(py::init<
                std::tuple<int, int, int>,
                int,
                int,
                int,
                int
            >(),
            py::arg("hiddenSize") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("eRadius") = 2,
            py::arg("dRadius") = 2,
            py::arg("ticksPerUpdate") = 2,
            py::arg("temporalHorizon") = 2
        )
        .def_readwrite("hiddenSize", &pyaon::LayerDesc::hiddenSize)
        .def_readwrite("eRadius", &pyaon::LayerDesc::eRadius)
        .def_readwrite("dRadius", &pyaon::LayerDesc::dRadius)
        .def_readwrite("ticksPerUpdate", &pyaon::LayerDesc::ticksPerUpdate)
        .def_readwrite("temporalHorizon", &pyaon::LayerDesc::temporalHorizon);

    py::class_<pyaon::Hierarchy>(m, "Hierarchy")
        .def(py::init<
                const std::vector<pyaon::IODesc>&,
                const std::vector<pyaon::LayerDesc>&,
                const std::string&,
                const std::vector<unsigned char>&
            >(),
            py::arg("ioDescs") = std::vector<pyaon::IODesc>(),
            py::arg("layerDescs") = std::vector<pyaon::LayerDesc>(),
            py::arg("name") = std::string(),
            py::arg("buffer") = std::vector<unsigned char>()
        )
        .def("saveToFile", &pyaon::Hierarchy::saveToFile)
        .def("serializeToBuffer", &pyaon::Hierarchy::serializeToBuffer)
        .def("setStateFromBuffer", &pyaon::Hierarchy::setStateFromBuffer)
        .def("serializeStateToBuffer", &pyaon::Hierarchy::serializeStateToBuffer)
        .def("step", &pyaon::Hierarchy::step,
            py::arg("inputCIs"),
            py::arg("learnEnabled") = true,
            py::arg("reward") = 0.0f,
            py::arg("mimic") = 0.0f
        )
        .def("clearState", &pyaon::Hierarchy::clearState)
        .def("getNumLayers", &pyaon::Hierarchy::getNumLayers)
        .def("setInputImportance", &pyaon::Hierarchy::setInputImportance)
        .def("getInputImportance", &pyaon::Hierarchy::getInputImportance)
        .def("getPredictionCIs", &pyaon::Hierarchy::getPredictionCIs)
        .def("getHiddenCIs", &pyaon::Hierarchy::getHiddenCIs)
        .def("getHiddenSize", &pyaon::Hierarchy::getHiddenSize)
        .def("getNumEVisibleLayers", &pyaon::Hierarchy::getNumEVisibleLayers)
        .def("getTicks", &pyaon::Hierarchy::getTicks)
        .def("getTicksPerUpdate", &pyaon::Hierarchy::getTicksPerUpdate)
        .def("getNumIO", &pyaon::Hierarchy::getNumIO)
        .def("getIOSize", &pyaon::Hierarchy::getIOSize)
        .def("getIOType", &pyaon::Hierarchy::getIOType)
        .def("setEGap", &pyaon::Hierarchy::setEGap)
        .def("getEGap", &pyaon::Hierarchy::getEGap)
        .def("setEVigilance", &pyaon::Hierarchy::setEVigilance)
        .def("getEVigilance", &pyaon::Hierarchy::getEVigilance)
        .def("setELR", &pyaon::Hierarchy::setELR)
        .def("getELR", &pyaon::Hierarchy::getELR)
        .def("setELRadius", &pyaon::Hierarchy::setELRadius)
        .def("getELRadius", &pyaon::Hierarchy::getELRadius)
        .def("setDScale", &pyaon::Hierarchy::setDScale)
        .def("getDScale", &pyaon::Hierarchy::getDScale)
        .def("setDLR", &pyaon::Hierarchy::setDLR)
        .def("getDLR", &pyaon::Hierarchy::getDLR)
        .def("setDStability", &pyaon::Hierarchy::setDStability)
        .def("getDStability", &pyaon::Hierarchy::getDStability)
        .def("setAVLR", &pyaon::Hierarchy::setAVLR)
        .def("getAVLR", &pyaon::Hierarchy::getAVLR)
        .def("setAALR", &pyaon::Hierarchy::setAALR)
        .def("getAALR", &pyaon::Hierarchy::getAALR)
        .def("setABias", &pyaon::Hierarchy::setABias)
        .def("getABias", &pyaon::Hierarchy::getABias)
        .def("setADiscount", &pyaon::Hierarchy::setADiscount)
        .def("getADiscount", &pyaon::Hierarchy::getADiscount)
        .def("setATemperature", &pyaon::Hierarchy::setATemperature)
        .def("getATemperature", &pyaon::Hierarchy::getATemperature)
        .def("setAMinSteps", &pyaon::Hierarchy::setAMinSteps)
        .def("getAMinSteps", &pyaon::Hierarchy::getAMinSteps)
        .def("setAHistoryIters", &pyaon::Hierarchy::setAHistoryIters)
        .def("getAHistoryIters", &pyaon::Hierarchy::getAHistoryIters)
        .def("getERadius", &pyaon::Hierarchy::getERadius)
        .def("getDRadius", &pyaon::Hierarchy::getDRadius)
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
        .def(py::init<
                const std::tuple<int, int, int>&,
                const std::vector<pyaon::ImageEncoderVisibleLayerDesc>&,
                const std::string&,
                const std::vector<unsigned char>&
            >(),
            py::arg("hiddenSize") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("visibleLayerDescs") = std::vector<pyaon::ImageEncoderVisibleLayerDesc>(),
            py::arg("name") = std::string(),
            py::arg("buffer") = std::vector<unsigned char>()
        )
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
        .def("setGap", &pyaon::ImageEncoder::setGap)
        .def("getGap", &pyaon::ImageEncoder::getGap)
        .def("setVigilance", &pyaon::ImageEncoder::setVigilance)
        .def("getVigilance", &pyaon::ImageEncoder::getVigilance)
        .def("setLR", &pyaon::ImageEncoder::setLR)
        .def("getLR", &pyaon::ImageEncoder::getLR);
}
