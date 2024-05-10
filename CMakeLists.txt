# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.17)

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
  message(STATUS "Using CUDA ${CUDA_VERSION}")
  if (WIN32)
    set(CUDA_NVCC_FLAGS -std=c++17)
  else ()
    set(CUDA_NVCC_FLAGS -std=c++11)
  endif ()

  if(CUDA_VERSION VERSION_GREATER "10.1" OR CUDA_VERSION VERSION_EQUAL "10.1")
    add_definitions(-DTRITON_ENABLE_CUDA_GRAPH=1)
  else()
    message(WARNING "CUDA ${CUDA_VERSION} does not support CUDA graphs.")
  endif()
else()
  message( FATAL_ERROR "TensorRT backend requires TRITON_ENABLE_GPU=1, CMake will exit." )
endif() # TRITON_ENABLE_GPU

if(${TRITON_ENABLE_NVTX})
  add_definitions(-DTRITON_ENABLE_NVTX=1)
endif() # TRITON_ENABLE_NVTX

if(NOT DEFINED ENV{TRT_VERSION})
  set(ENV{TRT_VERSION} 10.0.1.6)
endif()

message(STATUS "TRT_VERSION envvar is $ENV{TRT_VERSION}")

#
# FIXME: Update src folder to build with TRT8 and then remove
# src_trt8 folder.
# Currently the iGPU builds are still using TensorRT 8. This means
# we need to keep supporting TRT 8 for this backend. As there has
# been significant clean-ups when moving from TRT 8 to TRT 10
# we are maintaining a previous version of backend for now.
#
if("$ENV{TRT_VERSION}" VERSION_LESS 10)
  set(SOURCE_DIR "src_trt8")
else()
  set(SOURCE_DIR "src")
endif()


#
# Shared library implementing the Triton Backend API
#
configure_file(${SOURCE_DIR}/libtriton_tensorrt.ldscript libtriton_tensorrt.ldscript COPYONLY)

add_library(
  triton-tensorrt-backend SHARED
  ${SOURCE_DIR}/tensorrt.cc
  ${SOURCE_DIR}/model_state.cc
  ${SOURCE_DIR}/tensorrt_model.cc
  ${SOURCE_DIR}/tensorrt_model.h
  ${SOURCE_DIR}/instance_state.cc
  ${SOURCE_DIR}/tensorrt_model_instance.cc
  ${SOURCE_DIR}/tensorrt_model_instance.h
  ${SOURCE_DIR}/tensorrt_utils.cc
  ${SOURCE_DIR}/tensorrt_utils.h
  ${SOURCE_DIR}/filesystem.h
  ${SOURCE_DIR}/filesystem.cc
  ${SOURCE_DIR}/semaphore.h
  ${SOURCE_DIR}/shared_library.h
  ${SOURCE_DIR}/shared_library.cc
  ${SOURCE_DIR}/loader.cc
  ${SOURCE_DIR}/loader.h
  ${SOURCE_DIR}/logging.cc
  ${SOURCE_DIR}/logging.h
  ${SOURCE_DIR}/output_allocator.cc
  ${SOURCE_DIR}/output_allocator.h
  ${SOURCE_DIR}/io_binding_info.cc
  ${SOURCE_DIR}/io_binding_info.h
)

add_library(
  TritonTensorRTBackend::triton-tensorrt-backend ALIAS triton-tensorrt-backend
)

target_include_directories(
  triton-tensorrt-backend
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_DIR}
)

target_include_directories(
    triton-tensorrt-backend
    PRIVATE ${TRITON_TENSORRT_INCLUDE_PATHS}
  )

target_compile_features(triton-tensorrt-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-tensorrt-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror -Wno-deprecated-declarations>
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

find_library(NVINFER_LIBRARY NAMES nvinfer)
find_library(NVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin)
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
