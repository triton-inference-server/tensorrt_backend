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

#include "model_state.hh"
#include "triton/backend/backend_model_instance.h"

namespace triton::backend::tensorrt::model_connect {

// Loads one bundle and runs one request at a time. BackendInputCollector
// copies that request's tensors from the request to host buffers, the bundle
// task runs, and BackendOutputResponder writes numeric outputs.
// text_continuation and text_generation decode one string or one token-id
// tensor. structured_ranking
// passes the numeric tensors to IStructuredRanking::score. A TYPE_STRING
// output is length-prefixed bytes, which the responder cannot size, so that
// output is copied once from the result into the response. This instance
// does not call batch_text_continuation and does not combine requests.
class ModelInstanceState : public BackendModelInstance {
 public:
  // Requires KIND_GPU. Loads the bundle and returns the instance.
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  // Releases the loaded bundle.
  ~ModelInstanceState() override;

  // Runs each request on its own and sends its response.
  void ProcessRequests(
      TRITONBACKEND_Request** requests, uint32_t request_count);

 private:
  // Requires KIND_GPU.
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  // Sets this instance's CUDA device, selects the bundle task, and stores
  // sdk_->run.
  TRITONSERVER_Error* LoadAPI();

  // Checks the request inputs, copies them to host NamedTensors, runs the
  // task, and writes the outputs.
  TRITONSERVER_Error* RunRequest(
      TRITONBACKEND_Request* request, TRITONBACKEND_Response** response);

  ModelState* model_state_;
  struct Sdk;
  std::unique_ptr<Sdk> sdk_;
};

}  // namespace triton::backend::tensorrt::model_connect
