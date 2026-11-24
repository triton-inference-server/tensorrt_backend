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
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model_connect/trtmc_utils.hh"
#include "triton/backend/backend_common.h"

namespace triton::backend::tensorrt::model_connect {

namespace {

// Appends one batch item per string or token row of the request input and
// returns the item count. The items borrow tensor.bytes, which ValidateInput
// has already accepted.
uint64_t
AppendContinuationItems(
    const NamedTensor& tensor, const int max_batch_size,
    std::vector<trtmc_batch_text_continuation_item_v1>& items)
{
  const uint64_t count =
      max_batch_size > 0 ? static_cast<uint64_t>(tensor.shape[0]) : 1;
  if (tensor.datatype == TRITONSERVER_TYPE_BYTES) {
    const char* pos = reinterpret_cast<const char*>(tensor.bytes.data());
    for (uint64_t i = 0; i < count; ++i) {
      uint32_t length = 0;
      memcpy(&length, pos, sizeof(length));
      pos += sizeof(length);
      trtmc_batch_text_continuation_item_v1& item = items.emplace_back();
      item.input.prefix.kind = TRTMC_TEXT_UTF8;
      item.input.prefix.as.text = {length == 0 ? nullptr : pos, length};
      pos += length;
    }
    return count;
  }
  const uint64_t sequence = static_cast<uint64_t>(tensor.shape.back());
  const int32_t* tokens = reinterpret_cast<const int32_t*>(tensor.bytes.data());
  for (uint64_t i = 0; i < count; ++i) {
    trtmc_batch_text_continuation_item_v1& item = items.emplace_back();
    item.input.prefix.kind = TRTMC_TEXT_TOKEN_IDS;
    item.input.prefix.as.token_ids = {
        sequence == 0 ? nullptr : tokens + i * sequence, sequence};
  }
  return count;
}

// One run_batch call. On success, keep owns the batch result and views holds
// one borrowed view per item, in order.
TRITONSERVER_Error*
RunContinuationBatch(
    const trtmc_core_api_v1* trtmc, trtmc_model* model,
    const trtmc_batch_text_continuation_api_v1* batch,
    const std::vector<trtmc_batch_text_continuation_item_v1>& items,
    std::shared_ptr<void>* keep, std::vector<trtmc_text_result_view_v1>* views)
{
  const trtmc_batch_text_continuation_request_v1 request{
      items.data(), items.size()};
  trtmc_result* raw = nullptr;
  trtmc_error* error = nullptr;
  trtmc_status status = batch->run_batch(model, &request, &raw, &error);
  keep->reset(raw, [trtmc](void* result) {
    if (trtmc != nullptr && result != nullptr) {
      trtmc->result_release(static_cast<trtmc_result*>(result));
    }
  });
  RETURN_IF_ERROR(TrtmcStatusError(trtmc, status, error));

  uint64_t count = 0;
  RETURN_IF_TRTMC_ERROR(trtmc, error, batch->result_count(raw, &count, &error));
  if (count != items.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "model_connect text continuation batch result count does not match");
  }
  views->assign(count, trtmc_text_result_view_v1{});
  for (uint64_t i = 0; i < count; ++i) {
    RETURN_IF_TRTMC_ERROR(
        trtmc, error, batch->result_item_view(raw, i, &(*views)[i], &error));
  }
  return nullptr;
}

// One request's output from views [begin, begin + count). Strings borrow the
// batch result, which keep owns. Token rows must share a length so the tensor
// is rectangular, and are copied. With max_batch_size greater than 0, the
// shape keeps the batch dimension.
TRITONSERVER_Error*
PackContinuationOutput(
    const std::vector<trtmc_text_result_view_v1>& views, const size_t begin,
    const size_t count, const int max_batch_size, const NamedTensor& output,
    const std::shared_ptr<void>& keep, NamedTensor* packed)
{
  packed->name = output.name;
  if (output.datatype == TRITONSERVER_TYPE_BYTES) {
    packed->datatype = TRITONSERVER_TYPE_BYTES;
    packed->shape = {1};
    if (max_batch_size > 0) {
      packed->shape = {static_cast<int64_t>(count), 1};
    }
    packed->strings.reserve(count);
    for (size_t i = begin; i < begin + count; ++i) {
      packed->strings.push_back(views[i].text);
    }
    packed->external_keep = keep;
    return nullptr;
  }

  // CheckNamedTensor rejects these token ids for any other configured type.
  packed->datatype = TRITONSERVER_TYPE_INT32;
  const uint64_t width = views[begin].token_ids.size;
  for (size_t i = begin; i < begin + count; ++i) {
    const trtmc_i32_view& tokens = views[i].token_ids;
    if (tokens.size != width) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "text continuation token results have different lengths");
    }
    if (tokens.size > 0 && tokens.data == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "text result tokens are null");
    }
  }
  packed->shape = {static_cast<int64_t>(width)};
  if (max_batch_size > 0) {
    packed->shape = {static_cast<int64_t>(count), static_cast<int64_t>(width)};
  }
  const size_t row_bytes = static_cast<size_t>(width) * sizeof(int32_t);
  packed->bytes.resize(count * row_bytes);
  // A continuation can return no tokens. Its token_ids.data may then be null,
  // which memcpy does not accept even for zero bytes.
  for (size_t i = 0; i < count && row_bytes > 0; ++i) {
    memcpy(
        packed->bytes.data() + i * row_bytes, views[begin + i].token_ids.data,
        row_bytes);
  }
  return nullptr;
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
TextContinuationInstanceState::LoadModel()
{
  RETURN_IF_ERROR(GetTrtmcApi(&trtmc_));

  trtmc_error* error = nullptr;
  const std::string& bundle_path = model_state_->BundlePath();
  const trtmc_string_view bundle{bundle_path.data(), bundle_path.size()};
  RETURN_IF_TRTMC_ERROR(
      trtmc_, error, trtmc_->model_load(bundle, nullptr, &model_, &error));

  const trtmc_string_view task{
      TRTMC_TASK_BATCH_TEXT_CONTINUATION,
      sizeof(TRTMC_TASK_BATCH_TEXT_CONTINUATION) - 1};
  const trtmc_api_header* api = nullptr;
  RETURN_IF_TRTMC_ERROR(
      trtmc_, error,
      trtmc_->model_get_task_api(
          model_, task, TRTMC_API_VERSION_MAJOR, TRTMC_API_VERSION_MINOR, &api,
          &error));
  if (api == nullptr || api->major != TRTMC_API_VERSION_MAJOR ||
      api->minor != TRTMC_API_VERSION_MINOR ||
      api->byte_size < sizeof(trtmc_batch_text_continuation_api_v1)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "loaded bundle does not provide batch_text_continuation "
        "API " TRTMC_API_STR(TRTMC_API_VERSION_MAJOR) "." TRTMC_API_STR(
            TRTMC_API_VERSION_MINOR));
  }
  batch_ = reinterpret_cast<const trtmc_batch_text_continuation_api_v1*>(api);
  if (batch_->run_batch == nullptr || batch_->result_count == nullptr ||
      batch_->result_item_view == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "loaded bundle does not provide batch_text_continuation");
  }
  return nullptr;
}

// One string, or one INT32 token-id sequence. With max_batch_size greater
// than 0, each batch row is one continuation.
TRITONSERVER_Error*
TextContinuationInstanceState::ValidateInput(
    TRITONBACKEND_Input* input, const char* name,
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    const uint32_t dims) const
{
  const int max_batch_size = model_state_->MaxBatchSize();
  if (datatype != TRITONSERVER_TYPE_INT32) {
    return ValidateStringInput(
        input, name, datatype, shape, dims, max_batch_size);
  }
  if (max_batch_size > 0) {
    if (shape == nullptr || dims != 2 || shape[0] < 1 || shape[1] < 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "text continuation token input must be [batch, sequence]");
    }
  } else if (shape == nullptr || dims != 1 || shape[0] < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "text continuation input must be one rank-1 token-id sequence");
  }
  return nullptr;
}

// A copy failure fails only that request. A run_batch failure
// fails every request in the batch. A response write failure fails only that
// response.
TRITONSERVER_Error*
TextContinuationInstanceState::Execute(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>& responses)
{
  const int max_batch_size = model_state_->MaxBatchSize();
  std::vector<size_t> batch_indices;
  std::vector<uint64_t> counts;
  // The items borrow these input bytes until run_batch returns.
  std::vector<NamedTensor> inputs;
  inputs.reserve(request_count);
  std::vector<trtmc_batch_text_continuation_item_v1> items;
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response** response = &responses[r];
    if (*response == nullptr) {
      continue;
    }
    std::vector<NamedTensor> tensors;
    TRITONSERVER_Error* err = SetInputTensors(requests[r], response, &tensors);
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    if (*response == nullptr) {
      continue;
    }
    // ModelState requires exactly one input for a text task.
    inputs.push_back(std::move(tensors[0]));
    counts.push_back(
        AppendContinuationItems(inputs.back(), max_batch_size, items));
    batch_indices.push_back(r);
  }
  if (batch_indices.empty()) {
    return nullptr;
  }

  // The views borrow the batch result, which keep holds until every response
  // is written.
  std::shared_ptr<void> keep;
  std::vector<trtmc_text_result_view_v1> views;
  SET_TIMESTAMP(compute_start_ns_);
  RETURN_IF_ERROR(
      RunContinuationBatch(trtmc_, model_, batch_, items, &keep, &views));
  SET_TIMESTAMP(compute_end_ns_);
  size_t begin = 0;
  for (size_t i = 0; i < batch_indices.size(); ++i) {
    const size_t r = batch_indices[i];
    TRITONBACKEND_Response** response = &responses[r];
    std::vector<NamedTensor> outputs(1);
    TRITONSERVER_Error* err = PackContinuationOutput(
        views, begin, counts[i], max_batch_size, model_state_->Outputs()[0],
        keep, &outputs[0]);
    begin += counts[i];
    if (err == nullptr) {
      err = ReadOutputTensors(requests[r], response, outputs);
    }
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
  }
  return nullptr;
}

}  // namespace triton::backend::tensorrt::model_connect
