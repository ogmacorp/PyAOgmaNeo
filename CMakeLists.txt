# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13)

project(pyaogmaneo)

include(FetchContent)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

FetchContent_Declare(
    AOgmaNeo
    GIT_REPOSITORY https://github.com/ogmacorp/AOgmaNeo.git
    GIT_TAG origin/master
)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG origin/master 
)

#FetchContent_GetProperties(AOgmaNeo)

#if(NOT AOgmaNeo_POPULATED)
#    FetchContent_Populate(AOgmaNeo)
#endif()

FetchContent_MakeAvailable(AOgmaNeo)
FetchContent_MakeAvailable(pybind11)

FetchContent_GetProperties(pybind11)

if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

############################################################################
# Add the pyaogmaneo

set(PYAOGMANEO_INCLUDE_DIR "source/pyaogmaneo;")

include_directories(${PYAOGMANEO_INCLUDE_DIR})

set(PYAOGMANEO_SRC
    "source/pyaogmaneo/PyModule.cpp"
    "source/pyaogmaneo/PyHelpers.cpp"
    "source/pyaogmaneo/PyHierarchy.cpp"
    "source/pyaogmaneo/PyImageEncoder.cpp"
)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

pybind11_add_module(pyaogmaneo ${PYAOGMANEO_SRC})

target_link_libraries(pyaogmaneo PUBLIC AOgmaNeo)
