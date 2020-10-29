# ----------------------------------------------------------------------------
#  PyAOgmaNeo
#  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of PyAOgmaNeo is licensed to you under the terms described
#  in the PYAOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13)

project(PyAOgmaNeo)

include(FetchContent)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

############################################################################
# Find current git branch, used to match branch name in AOgmaNeo

execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


############################################################################
# Add the OpenMP library

find_package(OpenMP REQUIRED)
 
include_directories(${OpenMP_CXX_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") # This links -fopenmp


############################################################################
# Add the AOgmaNeo library

FetchContent_Declare(
    AOgmaNeo
    GIT_REPOSITORY https://github.com/ogmacorp/AOgmaNeo.git
    GIT_TAG origin/${GIT_BRANCH}
)

#FetchContent_GetProperties(AOgmaNeo)

#if(NOT AOgmaNeo_POPULATED)
#    FetchContent_Populate(AOgmaNeo)
#endif()

FetchContent_MakeAvailable(AOgmaNeo)

############################################################################
# Add the PyAOgmaNeolibrary

set(PYAOGMANEO_INCLUDE_DIR "source/pyaogmaneo;")

include_directories(${PYAOGMANEO_INCLUDE_DIR})

file(GLOB_RECURSE PYAOGMANEO_SRC
    "source/pyaogmaneo/*.cpp"
)

add_subdirectory(extern/pybind11)
pybind11_add_module(PyAOgmaNeo ${PYAOGMANEO_SRC})

target_link_libraries(PyAOgmaNeo PUBLIC ${OpenMP_CXX_LIBRARIES} AOgmaNeo)
