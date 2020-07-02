# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13)

project(PyAOgmaNeo)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake/")

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(PYTHON_VERSION)
    message("Python target version: ${PYTHON_VERSION}")            
endif()


############################################################################
# Add the OpenMP library

find_package(OpenMP REQUIRED)
 
include_directories(${OpenMP_CXX_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # This links -fopenmp


############################################################################
# Add the OgmaNeo library

find_package(AOgmaNeo REQUIRED)

include_directories(${AOGMANEO_INCLUDE_DIR})


############################################################################
# Add Raylib

find_package(Raylib REQUIRED)

include_directories(${raylib_INCLUDE_DIRS})

############################################################################
# Add the PyAOgmaNeoCpp library

set(PYAOGMANEO_INCLUDE_DIR "source/pyaogmaneo;")

include_directories(${PYAOGMANEO_INCLUDE_DIR})

file(GLOB_RECURSE PYAOGMANEO_SRC
    "source/pyaogmaneo/*.h"
    "source/pyaogmaneo/*.cpp"
)

add_library(PyAOgmaNeoCpp ${PYAOGMANEO_SRC})

target_link_libraries(PyAOgmaNeoCpp ${OpenMP_CXX_LIBRARIES})
target_link_libraries(PyAOgmaNeoCpp ${AOGMANEO_LIBRARIES})
target_link_libraries(PyAOgmaNeoCpp ${raylib_LIBRARIES})


############################################################################
# Find SWIG and setup building the Python bindings to PyAOgmaNeo library

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()


############################################################################
# Find Python package

if(PYTHON_VERSION)
    find_package(PythonLibs ${PYTHON_VERSION} EXACT REQUIRED)
else()
    find_package(PythonLibs REQUIRED)
endif()

include_directories(${PYTHON_INCLUDE_DIR})


############################################################################
# Setup SWIG bindings

set(CMAKE_SWIG_OUTDIR "${CMAKE_BINARY_DIR}")

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

#set_source_files_properties(pyaogmaneo.i PROPERTIES SWIG_MODULE_NAME pyaogmaneo)
set_source_files_properties(pyaogmaneo.i PROPERTIES CPLUSPLUS ON)

swig_add_library(pyaogmaneo LANGUAGE python SOURCES pyaogmaneo.i)

target_link_libraries(pyaogmaneo PyAOgmaNeoCpp ${PYTHON_LIBRARIES})

set(CMAKE_INSTALL_PREFIX ${CMAKE_SWIG_OUTDIR})
install(TARGETS pyaogmaneo DESTINATION ${CMAKE_SWIG_OUTDIR})
install(FILES ${CMAKE_BINARY_DIR}/pyaogmaneo.py DESTINATION ${CMAKE_SWIG_OUTDIR})