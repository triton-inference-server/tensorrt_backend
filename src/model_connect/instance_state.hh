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

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "model_connect/model_state.hh"
#include "triton/backend/backend_model_instance.h"
#include "trtmc/task.h"
#include "trtmc/types.h"

namespace triton::backend::tensorrt::model_connect {

// One loaded bundle. ProcessRequests creates every response, sends it, and
// reports statistics for the execute. The subclass for the bundle task loads
// that task, checks each input, and runs each request.
// BackendInputCollector copies request tensors to host buffers and
// BackendOutputResponder writes numeric outputs. A TYPE_STRING output is
// length-prefixed bytes, which the responder cannot size, so that output is
// copied once from the result into the response.
class ModelInstanceState : public BackendModelInstance {
 public:
  // Requires KIND_GPU. Creates the subclass for the bundle task and loads it.
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  // Runs this execute and sends every response.
  void ProcessRequests(
      TRITONBACKEND_Request** requests, uint32_t request_count);

 protected:
  // One task call on one request's host tensors.
  using RunFn = std::function<TRITONSERVER_Error*(
      std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs)>;

  // Requires KIND_GPU.
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  // Loads the bundle task. The CUDA device is already set.
  virtual TRITONSERVER_Error* Load() = 0;

  // Checks one input of one request before any copy, given the properties
  // from TRITONBACKEND_InputProperties. A failure fails only that request.
  virtual TRITONSERVER_Error* ValidateInput(
      TRITONBACKEND_Input* input, const char* name,
      TRITONSERVER_DataType datatype, const int64_t* shape,
      uint32_t dims) const = 0;

  // Runs every request. A nullptr response marks a request that already
  // failed, which is skipped. A failure confined to one request is sent on its
  // response, which is then set to nullptr. A returned error fails every
  // remaining response. Must not throw; RunRequest catches exceptions from
  // each request.
  virtual TRITONSERVER_Error* Execute(
      std::vector<TRITONBACKEND_Request*>& requests,
      std::vector<TRITONBACKEND_Response*>* responses) = 0;

  // Loads the bundle with trtmc::load_task and the model's KV cache
  // parameters.
  TRITONSERVER_Error* LoadSdkTask(std::shared_ptr<trtmc::ITask>* task);

  // Copies one request's configured inputs to host NamedTensors.
  TRITONSERVER_Error* CopyInputsToHost(
      TRITONBACKEND_Request* request, TRITONBACKEND_Response** response,
      std::vector<NamedTensor>* tensors);

  // Writes one request's outputs. A write failure is that request's error.
  TRITONSERVER_Error* CopyOutputsToResponse(
      TRITONBACKEND_Request* request, TRITONBACKEND_Response** response,
      const std::vector<NamedTensor>& output_tensors);

  // Validates, copies, runs, and writes each request on its own, skipping a
  // nullptr response. A failure or exception fails only that request.
  TRITONSERVER_Error* RunRequest(
      std::vector<TRITONBACKEND_Request*>& requests,
      std::vector<TRITONBACKEND_Response*>* responses, const RunFn& run);

  ModelState* model_state_;

 private:
  // Runs ValidateInput on every input of the request.
  TRITONSERVER_Error* ValidateRequest(TRITONBACKEND_Request* request) const;
};

// Checks a TYPE_BYTES input that holds one string of shape [1].
TRITONSERVER_Error* ValidateStringInput(
    TRITONBACKEND_Input* input, const char* name,
    TRITONSERVER_DataType datatype, const int64_t* shape, uint32_t dims);

// Borrowed host view of one Triton string. Valid while tensor.bytes is alive.
void TextView(const NamedTensor& tensor, trtmc_string_view& text);

// One borrowed text or token-id result. keep owns it until the response write
// finishes.
TRITONSERVER_Error* AppendTextOutput(
    ModelState* model_state, const trtmc_string_view& text,
    const int32_t* token_ids, uint64_t token_count, std::shared_ptr<void> keep,
    std::vector<NamedTensor>* outputs);

}  // namespace triton::backend::tensorrt::model_connect
