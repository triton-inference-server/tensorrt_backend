// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "triton/backend/backend_model.h"

namespace triton { namespace backend { namespace tensorrt {

class TensorRTModel : public BackendModel {
 public:
  TensorRTModel(TRITONBACKEND_Model* triton_model);
  virtual ~TensorRTModel() = default;

  TRITONSERVER_Error* SetTensorRTModelConfig();

  TRITONSERVER_Error* ParseModelConfig();

  // The model configuration.
  common::TritonJson::Value& GraphSpecs() { return graph_specs_; }

  enum Priority { DEFAULT = 0, MIN = 1, MAX = 2 };
  Priority ModelPriority() { return priority_; }
  bool UseCudaGraphs() { return use_cuda_graphs_; }
  size_t GatherKernelBufferThreshold()
  {
    return gather_kernel_buffer_threshold_;
  }
  bool SeparateOutputStream() { return separate_output_stream_; }
  bool EagerBatching() { return eager_batching_; }
  bool BusyWaitEvents() { return busy_wait_events_; }

 protected:
  common::TritonJson::Value graph_specs_;
  Priority priority_;
  bool use_cuda_graphs_;
  size_t gather_kernel_buffer_threshold_;
  bool separate_output_stream_;
  bool eager_batching_;
  bool busy_wait_events_;
};

}}}  // namespace triton::backend::tensorrt
