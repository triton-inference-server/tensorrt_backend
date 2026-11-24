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
#include "trtmc/task.h"

namespace triton::backend::tensorrt::model_connect {

// text_generation through ITextGeneration::generate. Each request is one
// string prompt and runs on its own.
class TextGenerationInstanceState : public ModelInstanceState {
 public:
  TextGenerationInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

 private:
  TRITONSERVER_Error* LoadModel() override;
  TRITONSERVER_Error* ValidateInput(
      TRITONBACKEND_Input* input, const char* name,
      TRITONSERVER_DataType datatype, const int64_t* shape,
      uint32_t dims) const override;
  TRITONSERVER_Error* Execute(
      TRITONBACKEND_Request** requests, uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>& responses) override;

  std::shared_ptr<trtmc::ITask> task_;
  // Points into task_.
  trtmc::ITextGeneration* generation_{nullptr};
};

}  // namespace triton::backend::tensorrt::model_connect
