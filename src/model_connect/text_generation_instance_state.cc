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

#include "model_connect/text_generation_instance_state.hh"

#include <memory>
#include <string>
#include <vector>

#include "triton/backend/backend_common.h"

namespace triton::backend::tensorrt::model_connect {

namespace {

// generate() takes an owning std::string, so the prompt is copied. The result
// is borrowed by the output until the response write finishes.
TRITONSERVER_Error*
RunTextTask(
    trtmc::ITextGeneration* generation, ModelState* model_state,
    const std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs)
{
  trtmc_string_view prompt_view{};
  TextView(inputs[0], prompt_view);
  const std::string prompt =
      prompt_view.size == 0
          ? std::string()
          : std::string(
                prompt_view.data, static_cast<size_t>(prompt_view.size));
  trtmc::TextGenerationConfig config;
  config.max_new_tokens = generation->default_max_new_tokens();
  auto result =
      std::make_shared<trtmc::TextResult>(generation->generate(prompt, config));
  const trtmc_string_view text_view{
      result->text.data(), static_cast<uint64_t>(result->text.size())};
  return AppendTextOutput(
      model_state, text_view,
      result->token_ids.empty() ? nullptr : result->token_ids.data(),
      result->token_ids.size(), result, outputs);
}

}  // namespace

TextGenerationInstanceState::TextGenerationInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : ModelInstanceState(model_state, triton_model_instance)
{
}

TRITONSERVER_Error*
TextGenerationInstanceState::LoadModel()
{
  RETURN_IF_ERROR(LoadSdkTask(&task_));
  generation_ = dynamic_cast<trtmc::ITextGeneration*>(task_.get());
  if (generation_ == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "bundle task text_generation did not load ITextGeneration");
  }
  return nullptr;
}

TRITONSERVER_Error*
TextGenerationInstanceState::ValidateInput(
    TRITONBACKEND_Input* input, const char* name,
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    const uint32_t dims) const
{
  return ValidateStringInput(input, name, datatype, shape, dims);
}

TRITONSERVER_Error*
TextGenerationInstanceState::Execute(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>& responses)
{
  return RunRequest(
      requests, request_count, responses,
      [this](
          std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs) {
        return RunTextTask(generation_, model_state_, inputs, outputs);
      });
}

}  // namespace triton::backend::tensorrt::model_connect
