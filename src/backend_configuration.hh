// Copyright 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <NvInfer.h>

#include <future>
#include <map>
#include <unordered_map>

#include "io_binding_info.h"
#include "shape_tensor.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton_semaphore.h"

namespace triton::backend::tensorrt {

#define THROW_IF_BACKEND_INSTANCE_ERROR(X)                             \
  do {                                                                 \
    TRITONSERVER_Error* tie_err__ = (X);                               \
    if (tie_err__ != nullptr) {                                        \
      throw triton::backend::BackendModelInstanceException(tie_err__); \
    }                                                                  \
  } while (false)

// Number of CUDA event set for each instance.
// We need three sets to avoid event overlaps between issue
// and response threads.
static constexpr int EVENT_SET_COUNT = 3;

//
// BackendConfiguration
//
// Struct to hold values specified via backend config
struct BackendConfiguration {
  static const BackendConfiguration& RetrieveFrom(
      TRITONBACKEND_Backend* backend)
  {
    void* state = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_BackendState(backend, &state));
    return *reinterpret_cast<BackendConfiguration*>(state);
  }

  static const BackendConfiguration& RetrieveFrom(TRITONBACKEND_Model* model)
  {
    TRITONBACKEND_Backend* backend = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelBackend(model, &backend));
    return RetrieveFrom(backend);
  }

  static const BackendConfiguration& RetrieveFrom(
      TRITONBACKEND_ModelInstance* instance)
  {
    TRITONBACKEND_Model* model = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelInstanceModel(instance, &model));
    return RetrieveFrom(model);
  }

  bool coalesce_request_input_{false};
  bool enable_memory_tracker_{false};
};

}  // namespace triton::backend::tensorrt
