// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "triton/core/tritonbackend.h"
#include "trtmc/structured_ranking.h"
#include "trtmc/trtmc.h"

#define TRTMC_API_VERSION_MAJOR 1
#define TRTMC_API_VERSION_MINOR 0
#define TRTMC_API_STR_(value) #value
#define TRTMC_API_STR(value) TRTMC_API_STR_(value)

namespace triton::backend::tensorrt::model_connect {

// One SDK status, as a Triton error. Releases the SDK error object.
inline TRITONSERVER_Error*
TrtmcStatusError(
    const trtmc_core_api_v1* trtmc, trtmc_status status, trtmc_error* error)
{
  if (status == TRTMC_OK) {
    if (error != nullptr) {
      trtmc->error_release(error);
    }
    return nullptr;
  }

  std::string message = "TensorRT Model Connect request failed";
  if (error != nullptr) {
    const trtmc_string_view detail = trtmc->error_message(error);
    if (detail.data != nullptr && detail.size > 0) {
      message.assign(detail.data, static_cast<size_t>(detail.size));
    }
    trtmc->error_release(error);
  }

  TRITONSERVER_Error_Code code = TRITONSERVER_ERROR_INTERNAL;
  switch (status) {
    case TRTMC_INVALID_ARGUMENT:
    case TRTMC_INVALID_CONFIG:
      code = TRITONSERVER_ERROR_INVALID_ARG;
      break;
    case TRTMC_UNSUPPORTED:
    case TRTMC_VERSION_MISMATCH:
      code = TRITONSERVER_ERROR_UNSUPPORTED;
      break;
    case TRTMC_OUT_OF_MEMORY:
    case TRTMC_BUSY:
      code = TRITONSERVER_ERROR_UNAVAILABLE;
      break;
    default:
      break;
  }
  return TRITONSERVER_ErrorNew(code, message.c_str());
}

enum class TaskKind {
  kStructuredRanking,
  kTextGeneration,
  kTextContinuation,
  kUnsupported
};

inline TaskKind
TaskKindFor(const std::string& task)
{
  if (task == trtmc::IStructuredRanking::kTask) {
    return TaskKind::kStructuredRanking;
  } else if (task == "text_generation") {
    return TaskKind::kTextGeneration;
  } else if (task == TRTMC_TASK_TEXT_CONTINUATION) {
    return TaskKind::kTextContinuation;
  } else {
    return TaskKind::kUnsupported;
  }
}

// TensorRT kFLOAT, kHALF, kBF16, kINT8, kINT32, and kINT64. kBOOL, kUINT8,
// kFP8, kINT4, kFP4, and kE8M0 have no trtmc::DType.
inline bool
NumericType(TRITONSERVER_DataType triton_type, trtmc::DType* dtype)
{
  switch (triton_type) {
    case TRITONSERVER_TYPE_FP32:
      *dtype = trtmc::DType::kFloat32;
      break;
    case TRITONSERVER_TYPE_FP16:
      *dtype = trtmc::DType::kFloat16;
      break;
    case TRITONSERVER_TYPE_BF16:
      *dtype = trtmc::DType::kBFloat16;
      break;
    case TRITONSERVER_TYPE_INT8:
      *dtype = trtmc::DType::kInt8;
      break;
    case TRITONSERVER_TYPE_INT32:
      *dtype = trtmc::DType::kInt32;
      break;
    case TRITONSERVER_TYPE_INT64:
      *dtype = trtmc::DType::kInt64;
      break;
    default:
      return false;
  }
  return trtmc::dtype_size(*dtype) != 0;
}

inline bool
TritonType(trtmc::DType dtype, TRITONSERVER_DataType* triton_type)
{
  switch (dtype) {
    case trtmc::DType::kFloat32:
      *triton_type = TRITONSERVER_TYPE_FP32;
      return true;
    case trtmc::DType::kFloat16:
      *triton_type = TRITONSERVER_TYPE_FP16;
      return true;
    case trtmc::DType::kBFloat16:
      *triton_type = TRITONSERVER_TYPE_BF16;
      return true;
    case trtmc::DType::kInt8:
      *triton_type = TRITONSERVER_TYPE_INT8;
      return true;
    case trtmc::DType::kInt32:
      *triton_type = TRITONSERVER_TYPE_INT32;
      return true;
    case trtmc::DType::kInt64:
      *triton_type = TRITONSERVER_TYPE_INT64;
      return true;
    default:
      return false;
  }
}

}  // namespace triton::backend::tensorrt::model_connect
