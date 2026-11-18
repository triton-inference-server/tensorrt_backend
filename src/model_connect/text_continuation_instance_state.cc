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

#include "model_connect/text_continuation_instance_state.hh"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model_connect/trtmc_utils.hh"
#include "triton/backend/backend_common.h"

namespace triton::backend::tensorrt::model_connect {

namespace {

// The prefix borrows tensor.bytes. text->run reads it before it returns.
// ValidateInput has already accepted the input.
trtmc_text_continuation_request_v1
ContinuationRequest(const NamedTensor& tensor)
{
  trtmc_text_continuation_request_v1 request{};
  if (tensor.datatype == TRITONSERVER_TYPE_BYTES) {
    request.prefix.kind = TRTMC_TEXT_UTF8;
    TextView(tensor, request.prefix.as.text);
    return request;
  }
  const uint64_t count = static_cast<uint64_t>(tensor.shape[0]);
  request.prefix.kind = TRTMC_TEXT_TOKEN_IDS;
  request.prefix.as.token_ids = {
      count == 0 ? nullptr
                 : reinterpret_cast<const int32_t*>(tensor.bytes.data()),
      count};
  return request;
}

// One text->run call. The output borrows the C result, which keep releases
// after the response write finishes.
TRITONSERVER_Error*
RunTextTask(
    const trtmc_core_api_v1* trtmc, trtmc_model* model,
    const trtmc_text_continuation_api_v1* text, ModelState* model_state,
    const std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs)
{
  const trtmc_text_continuation_request_v1 request =
      ContinuationRequest(inputs[0]);
  trtmc_result* raw = nullptr;
  trtmc_error* error = nullptr;
  trtmc_status status = text->run(model, &request, nullptr, &raw, &error);
  std::shared_ptr<void> keep(raw, [trtmc](void* result) {
    if (trtmc != nullptr && result != nullptr) {
      trtmc->result_release(static_cast<trtmc_result*>(result));
    }
  });
  RETURN_IF_ERROR(TrtmcStatusError(trtmc, status, error));

  trtmc_text_result_view_v1 view{};
  RETURN_IF_TRTMC_ERROR(trtmc, error, text->result_view(raw, &view, &error));
  return AppendTextOutput(
      model_state, view.text, view.token_ids.data, view.token_ids.size, keep,
      outputs);
}

}  // namespace

TextContinuationInstanceState::TextContinuationInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : ModelInstanceState(model_state, triton_model_instance)
{
}

TextContinuationInstanceState::~TextContinuationInstanceState()
{
  if (trtmc_ != nullptr && model_ != nullptr) {
    trtmc_->model_release(model_);
  }
}

TRITONSERVER_Error*
TextContinuationInstanceState::Load()
{
  if (model_state_->Inputs().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "model_connect text task requires one input");
  }
  if (model_state_->Outputs().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "model_connect text task requires one output");
  }
  // model_load does not accept either cache budget argument.
  const std::string& bundle_task = model_state_->BundleTask();
  const Parameters& parameters = model_state_->Parameters();
  if (parameters.kv_cache_size_bytes != 0 ||
      parameters.kv_cache_memory_ratio != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("model_connect task '") + bundle_task +
         "' does not accept 'kv_cache_size_bytes' or "
         "'kv_cache_memory_ratio'")
            .c_str());
  }

  RETURN_IF_ERROR(GetTrtmcApi(&trtmc_));

  trtmc_error* error = nullptr;
  const std::string& bundle_path = model_state_->BundlePath();
  const trtmc_string_view bundle{bundle_path.data(), bundle_path.size()};
  RETURN_IF_TRTMC_ERROR(
      trtmc_, error, trtmc_->model_load(bundle, nullptr, &model_, &error));

  const trtmc_string_view task{
      TRTMC_TASK_TEXT_CONTINUATION, sizeof(TRTMC_TASK_TEXT_CONTINUATION) - 1};
  const trtmc_api_header* api = nullptr;
  RETURN_IF_TRTMC_ERROR(
      trtmc_, error,
      trtmc_->model_get_task_api(
          model_, task, TRTMC_API_VERSION_MAJOR, TRTMC_API_VERSION_MINOR, &api,
          &error));
  if (api == nullptr || api->major != TRTMC_API_VERSION_MAJOR ||
      api->minor != TRTMC_API_VERSION_MINOR ||
      api->byte_size < sizeof(trtmc_text_continuation_api_v1)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "loaded bundle does not provide text_continuation "
        "API " TRTMC_API_STR(TRTMC_API_VERSION_MAJOR) "." TRTMC_API_STR(
            TRTMC_API_VERSION_MINOR));
  }
  text_ = reinterpret_cast<const trtmc_text_continuation_api_v1*>(api);
  return nullptr;
}

// One string, or one rank-1 INT32 token-id sequence.
TRITONSERVER_Error*
TextContinuationInstanceState::ValidateInput(
    TRITONBACKEND_Input* input, const char* name,
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    const uint32_t dims) const
{
  if (datatype != TRITONSERVER_TYPE_INT32) {
    return ValidateStringInput(input, name, datatype, shape, dims);
  }
  if (shape == nullptr || dims != 1 || shape[0] < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "text continuation input must be one rank-1 token-id sequence");
  }
  return nullptr;
}

TRITONSERVER_Error*
TextContinuationInstanceState::Execute(
    std::vector<TRITONBACKEND_Request*>& requests,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  return RunRequest(
      requests, responses,
      [this](
          std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs) {
        return RunTextTask(
            trtmc_, model_, text_, model_state_, inputs, outputs);
      });
}

}  // namespace triton::backend::tensorrt::model_connect
