// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020-2021 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PyHierarchy.h"
#include "PyImageEncoder.h"
#include "PyRLAdapter.h"

namespace py = pybind11;

PYBIND11_MODULE(pyaogmaneo, m) {
    m.def("setNumThreads", &pyaon::setNumThreads);
    m.def("getNumThreads", &pyaon::getNumThreads);

    m.def("setGlobalState", &pyaon::setGlobalState);
    m.def("getGlobalState", &pyaon::getGlobalState);

    py::enum_<pyaon::IOType>(m, "IOType")
        .value("none", pyaon::none)
        .value("prediction", pyaon::prediction)
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
            py::arg("historyCapacity") = 32
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
                int,
                int,
                int
            >(),
            py::arg("hiddenSize") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("gHiddenSizeZ") = 16,
            py::arg("eRadius") = 2,
            py::arg("dRadius") = 2,
            py::arg("historyCapacity") = 32,
            py::arg("ticksPerUpdate") = 2,
            py::arg("temporalHorizon") = 2
        )
        .def_readwrite("hiddenSize", &pyaon::LayerDesc::hiddenSize)
        .def_readwrite("gHiddenSizeZ", &pyaon::LayerDesc::gHiddenSizeZ)
        .def_readwrite("eRadius", &pyaon::LayerDesc::eRadius)
        .def_readwrite("dRadius", &pyaon::LayerDesc::dRadius)
        .def_readwrite("historyCapacity", &pyaon::LayerDesc::historyCapacity)
        .def_readwrite("ticksPerUpdate", &pyaon::LayerDesc::ticksPerUpdate)
        .def_readwrite("temporalHorizon", &pyaon::LayerDesc::temporalHorizon);

    py::class_<pyaon::GDesc>(m, "GDesc")
        .def(py::init<
                std::tuple<int, int, int>,
                int
            >(),
            py::arg("size") = std::tuple<int, int, int>({ 4, 4, 16 }),
            py::arg("radius") = 2
        )
        .def_readwrite("size", &pyaon::GDesc::size)
        .def_readwrite("radius", &pyaon::GDesc::radius);

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
            py::arg("goalCIs"),
            py::arg("actualCIs"),
            py::arg("learnEnabled") = true
        )
        .def("getNumLayers", &pyaon::Hierarchy::getNumLayers)
        .def("getTopHiddenCIs", &pyaon::Hierarchy::getTopHiddenCIs)
        .def("getTopHiddenSize", &pyaon::Hierarchy::getTopHiddenSize)
        .def("getTopUpdate", &pyaon::Hierarchy::getTopUpdate)
        .def("setImportance", &pyaon::Hierarchy::setImportance)
        .def("getImportance", &pyaon::Hierarchy::getImportance)
        .def("getPredictionCIs", &pyaon::Hierarchy::getPredictionCIs)
        .def("getUpdate", &pyaon::Hierarchy::getUpdate)
        .def("getHiddenCIs", &pyaon::Hierarchy::getHiddenCIs)
        .def("getHiddenSize", &pyaon::Hierarchy::getHiddenSize)
        .def("getTicks", &pyaon::Hierarchy::getTicks)
        .def("getTicksPerUpdate", &pyaon::Hierarchy::getTicksPerUpdate)
        .def("getNumEVisibleLayers", &pyaon::Hierarchy::getNumEVisibleLayers)
        .def("getNumIO", &pyaon::Hierarchy::getNumIO)
        .def("getNumGVisibleLayers", &pyaon::Hierarchy::getNumGVisibleLayers)
        .def("getIOSize", &pyaon::Hierarchy::getIOSize)
        .def("setELR", &pyaon::Hierarchy::setELR)
        .def("getELR", &pyaon::Hierarchy::getELR)
        .def("setGLR", &pyaon::Hierarchy::setGLR)
        .def("getGLR", &pyaon::Hierarchy::getGLR)
        .def("setDLR", &pyaon::Hierarchy::setDLR)
        .def("getDLR", &pyaon::Hierarchy::getDLR)
        .def("setDDecay", &pyaon::Hierarchy::setDDecay)
        .def("getDDecay", &pyaon::Hierarchy::getDDecay)
        .def("setDDiscount", &pyaon::Hierarchy::setDDiscount)
        .def("getDDiscount", &pyaon::Hierarchy::getDDiscount)
        .def("setDHistoryIters", &pyaon::Hierarchy::setDHistoryIters)
        .def("getDHistoryIters", &pyaon::Hierarchy::getDHistoryIters)
        .def("setDMaxSteps", &pyaon::Hierarchy::setDMaxSteps)
        .def("getDMaxSteps", &pyaon::Hierarchy::getDMaxSteps)
        .def("getERadius", &pyaon::Hierarchy::getERadius)
        .def("getDRadius", &pyaon::Hierarchy::getDRadius);

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

    py::class_<pyaon::RLAdapter>(m, "RLAdapter")
        .def(py::init<>())
        .def("initRandom", &pyaon::RLAdapter::initRandom)
        .def("initFromFile", &pyaon::RLAdapter::initFromFile)
        .def("initFromBuffer", &pyaon::RLAdapter::initFromBuffer)
        .def("saveToFile", &pyaon::RLAdapter::saveToFile)
        .def("serializeToBuffer", &pyaon::RLAdapter::serializeToBuffer)
        .def("step", &pyaon::RLAdapter::step,
            py::arg("hiddenCIs"),
            py::arg("reward"),
            py::arg("learnEnabled") = true
        )
        .def("getProgCIs", &pyaon::RLAdapter::getProgCIs)
        .def("getHiddenSize", &pyaon::RLAdapter::getHiddenSize)
        .def("setLR", &pyaon::RLAdapter::setLR)
        .def("getLR", &pyaon::RLAdapter::getLR);
}
