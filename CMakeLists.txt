# Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.31.8)

project(tritontensorrtbackend LANGUAGES C CXX)

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

#
# Options
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend." ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend." ON)
option(TRITON_ENABLE_NVTX "Include nvtx markers collection in backend." OFF)
set(TRITON_TENSORRT_LIB_PATHS "" CACHE PATH "Paths to TensorRT libraries. Multiple paths may be specified by separating them with a semicolon.")
set(TRITON_TENSORRT_INCLUDE_PATHS "" CACHE PATH "Paths to TensorRT includes. Multiple paths may be specified by separating them with a semicolon.")

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo.")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo.")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TRITON_TENSORRT_BACKEND_LIBNAME triton_tensorrt)
set(TRITON_TENSORRT_BACKEND_INSTALLDIR ${CMAKE_INSTALL_PREFIX}/backends/tensorrt)


if(LINUX)
  file(STRINGS "/etc/os-release" DISTRO_ID_LIKE REGEX "ID_LIKE")
  if(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
      set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/cuda/targets/sbsa-linux/lib")
    else()
      set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/cuda/targets/x86_64-linux/lib")
    endif()
  endif(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
endif(LINUX)

#
# Dependencies
#
# FetchContent's composability isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# CUDA
#
if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "Using CUDA ${CUDAToolkit_VERSION}")
  if (WIN32)
    set(CUDA_NVCC_FLAGS -std=c++17)
  else ()
    set(CUDA_NVCC_FLAGS -std=c++11)
  endif ()

  if(NOT DEFINED CUDAToolkit_VERSION)
    message( FATAL_ERROR "Unable to determine CUDAToolkit_VERSION, CMake will exit." )
  endif()

  if(CUDAToolkit_VERSION VERSION_GREATER "10.1" OR CUDAToolkit_VERSION VERSION_EQUAL "10.1")
    add_definitions(-DTRITON_ENABLE_CUDA_GRAPH=1)
  else()
    message(WARNING "CUDA ${CUDAToolkit_VERSION} does not support CUDA graphs.")
  endif()
else()
  message( FATAL_ERROR "TensorRT backend requires TRITON_ENABLE_GPU=1, CMake will exit." )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_NVTX})
  add_definitions(-DTRITON_ENABLE_NVTX=1)
endif() # TRITON_ENABLE_NVTX

add_library(
  triton-tensorrt-backend SHARED
  src/tensorrt.cc
  src/model_state.cc
  src/tensorrt_model.cc
  src/tensorrt_model.h
  src/instance_state.cc
  src/tensorrt_model_instance.cc
  src/tensorrt_model_instance.h
  src/shape_tensor.cc
  src/shape_tensor.h
  src/tensorrt_utils.cc
  src/tensorrt_utils.h
  src/filesystem.h
  src/filesystem.cc
  src/semaphore.h
  src/shared_library.h
  src/shared_library.cc
  src/loader.cc
  src/loader.h
  src/logging.cc
  src/logging.h
  src/output_allocator.cc
  src/output_allocator.h
  src/io_binding_info.cc
  src/io_binding_info.h
)
# Shared library implementing the Triton Backend API
configure_file(src/libtriton_tensorrt.ldscript libtriton_tensorrt.ldscript COPYONLY)

target_include_directories(
  triton-tensorrt-backend
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_library(
  TritonTensorRTBackend::triton-tensorrt-backend ALIAS triton-tensorrt-backend
)

target_include_directories(
    triton-tensorrt-backend
    PRIVATE ${TRITON_TENSORRT_INCLUDE_PATHS}
  )

target_compile_features(triton-tensorrt-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-tensorrt-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Wno-deprecated-declarations>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

# C/C++ defines that are used directly by this backend.
target_compile_definitions(
    triton-tensorrt-backend
    PRIVATE TRITON_ENABLE_GPU=1
)

if (WIN32)
  set_target_properties(
    triton-tensorrt-backend
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME ${TRITON_TENSORRT_BACKEND_LIBNAME}
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH "$\{ORIGIN\}"
  )
else ()
  set_target_properties(
    triton-tensorrt-backend
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      OUTPUT_NAME ${TRITON_TENSORRT_BACKEND_LIBNAME}
      SKIP_BUILD_RPATH TRUE
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      INSTALL_RPATH "$\{ORIGIN\}"
      LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_tensorrt.ldscript
      LINK_FLAGS "-Wl,--version-script libtriton_tensorrt.ldscript"
  )
endif()

FOREACH(p ${TRITON_TENSORRT_LIB_PATHS})
    set(TRITON_TENSORRT_LDFLAGS ${TRITON_TENSORRT_LDFLAGS} "-L${p}")
ENDFOREACH(p)

# NOTE: TRT 10 for Windows added the version suffix to the library names. See the release notes:
# https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html#tensorrt-10
find_library(NVINFER_LIBRARY NAMES nvinfer nvinfer_10)
find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin nvinfer_plugin_10)
target_link_libraries(
  triton-tensorrt-backend
  PRIVATE
    triton-core-serverapi   # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
    -lpthread
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
)

target_link_libraries(
    triton-tensorrt-backend
    PRIVATE ${TRITON_TENSORRT_LDFLAGS}
)

target_link_libraries(
    triton-tensorrt-backend
    PRIVATE
      CUDA::cudart
)


#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonTensorRTBackend)

if (NOT WIN32)
  install(
    TARGETS
      triton-tensorrt-backend
    EXPORT
      triton-tensorrt-backend-targets
    LIBRARY DESTINATION ${TRITON_TENSORRT_BACKEND_INSTALLDIR}
    ARCHIVE DESTINATION ${TRITON_TENSORRT_BACKEND_INSTALLDIR}
  )
else()
  install(
    TARGETS
      triton-tensorrt-backend
    EXPORT
      triton-tensorrt-backend-targets
    RUNTIME DESTINATION ${TRITON_TENSORRT_BACKEND_INSTALLDIR}
    ARCHIVE DESTINATION ${TRITON_TENSORRT_BACKEND_INSTALLDIR}
  )
endif() # WIN32

install(
  EXPORT
    triton-tensorrt-backend-targets
  FILE
    TritonTensorRTBackendTargets.cmake
  NAMESPACE
    TritonTensorRTBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonTensorRTBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorRTBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorRTBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-tensorrt-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonTensorRTBackendTargets.cmake
  NAMESPACE TritonTensorRTBackend::
)

export(PACKAGE TritonTensorRTBackend)
