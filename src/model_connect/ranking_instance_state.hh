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

#include <memory>
#include <vector>

#include "model_connect/instance_state.hh"
#include "trtmc/structured_ranking.h"

namespace triton::backend::tensorrt::model_connect {

// structured_ranking through IStructuredRanking::score. With max_batch_size
// greater than 0, the requests in one execute are concatenated on the batch
// dimension and scored once. With max_batch_size 0, each request is scored
// on its own. The task's KV cache statistics are published as Triton gauges
// for the life of the instance.
class RankingInstanceState : public ModelInstanceState {
 public:
  RankingInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  ~RankingInstanceState() override;

 private:
  TRITONSERVER_Error* LoadModel() override;
  TRITONSERVER_Error* ValidateInput(
      TRITONBACKEND_Input* input, const char* name,
      TRITONSERVER_DataType datatype, const int64_t* shape,
      uint32_t dims) const override;
  // With max_batch_size 0 or one pending request, scores each request through
  // RunRequest. Otherwise the collector concatenates the valid requests on the
  // batch dimension, score() runs once, and the responder splits each output
  // into the responses.
  TRITONSERVER_Error* Execute(
      TRITONBACKEND_Request** requests, uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>& responses) override;

  // Creates this instance's gauges and samples them until the instance is
  // destroyed.
  TRITONSERVER_Error* StartMetrics();
  // Writes the current metric values until the instance begins to shut down.
  void PollMetrics();

  std::shared_ptr<trtmc::ITask> task_;
  // Points into task_.
  trtmc::IStructuredRanking* ranking_{nullptr};
  struct Metrics;
  // Declared after the task so the publisher stops before the task is freed.
  std::unique_ptr<Metrics> metrics_;
};

}  // namespace triton::backend::tensorrt::model_connect
