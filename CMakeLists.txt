# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13)

project(pyaogmaneo)

include(FetchContent)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/")

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(USE_SYSTEM_AOGMANEO)
    message(STATUS "Using system installation of AOgmaNeo")

    find_package(AOgmaNeo)
else()
    message(STATUS "Not using system installation of AOgmaNeo, will download from repository")

    FetchContent_Declare(
        AOgmaNeo
        GIT_REPOSITORY https://github.com/ogmacorp/AOgmaNeo.git
        GIT_TAG 85f5e9c4ed12cbc37124ddaa3f9e73fa252e1799
    )

    FetchContent_GetProperties(AOgmaNeo)

    if(NOT AOgmaNeo_POPULATED)
        FetchContent_Populate(AOgmaNeo)
        add_subdirectory(${aogmaneo_SOURCE_DIR} ${aogmaneo_BINARY_DIR})
    endif()
endif()

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG origin/master 
)

FetchContent_GetProperties(pybind11)

if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

include_directories(${AOgmaNeo_SOURCE_DIR}/source)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

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

if(USE_SYSTEM_AOGMANEO)
    message(STATUS ${AOGMANEO_LIBRARIES})
    target_link_libraries(pyaogmaneo PUBLIC ${AOGMANEO_LIBRARIES} ${OpenMP_CXX_FLAGS})
else()
    target_link_libraries(pyaogmaneo PUBLIC AOgmaNeo ${OpenMP_CXX_FLAGS})
endif()
