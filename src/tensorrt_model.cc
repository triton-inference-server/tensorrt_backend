// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_model.h"

namespace triton { namespace backend { namespace tensorrt {

TensorRTModel::TensorRTModel(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), priority_(Priority::DEFAULT),
      use_cuda_graphs_(false), gather_kernel_buffer_threshold_(0),
      separate_output_stream_(false), eager_batching_(false),
      busy_wait_events_(false)
{
  ParseModelConfig();
}

void
TensorRTModel::ParseModelConfig()
{
  triton::common::TritonJson::Value optimization;
  if (model_config_.Find("optimization", &optimization)) {
    optimization.MemberAsUInt(
        "gather_kernel_buffer_threshold", &gather_kernel_buffer_threshold_);
    optimization.MemberAsBool("eager_batching", &eager_batching_);
    // FIXME: Capture ModelPriority
    triton::common::TritonJson::Value cuda;
    if (optimization.Find("cuda", &cuda)) {
      cuda.MemberAsBool("graphs", &use_cuda_graphs_);
      cuda.MemberAsBool("busy_wait_events", &busy_wait_events_);
      cuda.MemberAsArray("graph_spec", &graph_specs_);
      cuda.MemberAsBool("output_copy_stream", &separate_output_stream_);
    }
  }
}

}}}  // namespace triton::backend::tensorrt
