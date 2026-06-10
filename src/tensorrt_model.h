// Copyright 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <vector>

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
  int GetCudaStreamPriority();
  bool UseCudaGraphs() { return use_cuda_graphs_; }
  size_t GatherKernelBufferThreshold()
  {
    return gather_kernel_buffer_threshold_;
  }
  bool SeparateOutputStream() { return separate_output_stream_; }
  bool EagerBatching() { return eager_batching_; }
  bool BusyWaitEvents() { return busy_wait_events_; }

  // TensorRT Multi-Device (MD) — TRT-28040. When enabled, a single KIND_MODEL
  // instance drives a sharded engine across MdDeviceIds() GPUs in one process.
  bool EnableMultiDevice() const { return enable_multi_device_; }
  const std::vector<int>& MdDeviceIds() const { return md_device_ids_; }
  int MdRankCount() const { return static_cast<int>(md_device_ids_.size()); }
  // When true, each rank loads its own weight-sharded engine
  // '<model_filename>.rank{r}' (true tensor parallelism). When false, the same
  // engine is loaded on every rank (context/activation parallel).
  bool MdPerRankEngines() const { return md_per_rank_engines_; }

 protected:
  common::TritonJson::Value graph_specs_;
  Priority priority_;
  bool use_cuda_graphs_;
  size_t gather_kernel_buffer_threshold_;
  bool separate_output_stream_;
  bool eager_batching_;
  bool busy_wait_events_;

  // MD config, populated by ModelState::ParseParameters().
  bool enable_multi_device_{false};
  std::vector<int> md_device_ids_{};
  bool md_per_rank_engines_{false};
};

}}}  // namespace triton::backend::tensorrt
