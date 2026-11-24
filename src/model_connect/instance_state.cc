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

#include "model_connect/instance_state.hh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model_connect/ranking_instance_state.hh"
#include "model_connect/text_continuation_instance_state.hh"
#include "model_connect/text_generation_instance_state.hh"
#include "model_connect/trtmc_utils.hh"
#include "tensorrt.hh"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"
#include "trtmc/runtime/family_loader.h"

namespace triton::backend::tensorrt::model_connect {

namespace {

// Writes tensor.strings as one TYPE_BYTES output with tensor.shape. Each string
// follows its 4-byte length. TYPE_BYTES has no fixed element size, so the
// responder cannot write it. A GPU output buffer is filled through one staged
// copy. tensor.external_keep keeps the strings alive through the copy.
TRITONSERVER_Error*
SetStringOutputBuffer(
    TRITONBACKEND_Response* response, const NamedTensor& tensor)
{
  uint64_t byte_size = 0;
  for (const trtmc_string_view& text : tensor.strings) {
    if (text.size > std::numeric_limits<uint32_t>::max()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "text result string is too long");
    }
    if (text.size > 0 && text.data == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "text result text is null");
    }
    byte_size += sizeof(uint32_t) + text.size;
  }

  TRITONBACKEND_Output* output = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
      response, &output, tensor.name.c_str(), TRITONSERVER_TYPE_BYTES,
      tensor.shape.data(), tensor.shape.size()));
  void* out_buffer = nullptr;
  TRITONSERVER_MemoryType out_memory = TRITONSERVER_MEMORY_CPU;
  int64_t out_memory_id = 0;
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
      output, &out_buffer, byte_size, &out_memory, &out_memory_id));
  if (byte_size == 0) {
    return nullptr;
  }

  const bool host = out_memory != TRITONSERVER_MEMORY_GPU;
  std::vector<char> staging(host ? 0 : byte_size);
  char* pos = host ? static_cast<char*>(out_buffer) : staging.data();
  for (const trtmc_string_view& text : tensor.strings) {
    const uint32_t length = static_cast<uint32_t>(text.size);
    memcpy(pos, &length, sizeof(length));
    pos += sizeof(length);
    if (length > 0) {
      memcpy(pos, text.data, length);
      pos += length;
    }
  }
  if (host) {
    return nullptr;
  }
  bool cuda_used = false;
  RETURN_IF_ERROR(CopyBuffer(
      tensor.name, TRITONSERVER_MEMORY_CPU, 0, out_memory, out_memory_id,
      byte_size, staging.data(), out_buffer, nullptr, &cuda_used));
  if (cuda_used) {
    RETURN_IF_CUDA_ERROR(
        cudaStreamSynchronize(nullptr), TRITONSERVER_ERROR_INTERNAL,
        std::string("failed to synchronize model_connect copy"));
  }
  return nullptr;
}

// The error for the exception being handled. Call only from a catch handler
// for std::exception.
TRITONSERVER_Error*
CaughtRequestError()
{
  try {
    throw;
  }
  catch (const std::bad_alloc&) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        "model_connect request ran out of memory");
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("model_connect request failed: ") + ex.what()).c_str());
  }
}

// The collector already sent the response and cleared it.
TRITONSERVER_Error*
SentCopyError(
    TRITONBACKEND_Response** response,
    const std::vector<TRITONBACKEND_Response*>& responses)
{
  if (!responses.empty() && responses[0] == nullptr) {
    *response = nullptr;
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "model_connect tensor copy failed");
  }
  return nullptr;
}

// Copies one request input from the collector to tensor->bytes.
// ValidateInput has already accepted the request input.
TRITONSERVER_Error*
SetInputTensor(
    TRITONBACKEND_Request* request, const NamedTensor& input,
    BackendInputCollector* collector, NamedTensor* tensor)
{
  TRITONBACKEND_Input* request_input = nullptr;
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInput(request, input.name.c_str(), &request_input));
  TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
  const int64_t* shape = nullptr;
  uint32_t dims = 0;
  uint64_t byte_size = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      request_input, nullptr, &datatype, &shape, &dims, &byte_size, nullptr));

  tensor->name = input.name;
  tensor->datatype = datatype;
  tensor->shape.assign(shape, shape + dims);
  // Copies the request input, including a GPU buffer, to tensor->bytes.
  tensor->bytes.assign(static_cast<size_t>(byte_size), 0);
  const char* copied = nullptr;
  size_t copied_bytes = 0;
  TRITONSERVER_MemoryType copied_type = TRITONSERVER_MEMORY_CPU;
  int64_t copied_id = 0;
  return collector->ProcessTensor(
      input.name.c_str(),
      reinterpret_cast<char*>(
          tensor->bytes.empty() ? nullptr : tensor->bytes.data()),
      tensor->bytes.size(), {{TRITONSERVER_MEMORY_CPU, 0}}, &copied,
      &copied_bytes, &copied_type, &copied_id);
}

// The responder copies the byte size of the shape and does not look at the
// task buffer's length. TYPE_BYTES is sized by SetStringOutputBuffer.
TRITONSERVER_Error*
CheckNamedTensor(const NamedTensor& tensor, TRITONSERVER_DataType expected)
{
  if (tensor.datatype != expected) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unexpected datatype TYPE_") +
         TRITONSERVER_DataTypeString(tensor.datatype) +
         " for inference output '" + tensor.name + "', expecting TYPE_" +
         TRITONSERVER_DataTypeString(expected))
            .c_str());
  }
  if (expected == TRITONSERVER_TYPE_BYTES) {
    return nullptr;
  }
  int64_t count = 0;
  RETURN_IF_ERROR(backend::GetElementCount(tensor.shape, &count));
  const uint32_t element_bytes = TRITONSERVER_DataTypeByteSize(expected);
  const uint64_t actual_bytes =
      tensor.external_keep ? tensor.external_size : tensor.bytes.size();
  if (count >= 0 &&
      actual_bytes != static_cast<uint64_t>(count) * element_bytes) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, ("model_connect output '" + tensor.name +
                                      "' byte size does not match its shape")
                                         .c_str());
  }
  return nullptr;
}

// Leading dimension when the model batches, otherwise one request.
uint64_t
RequestBatchSize(TRITONBACKEND_Request* request, const int max_batch_size)
{
  if (max_batch_size == 0) {
    return 1;
  }
  TRITONBACKEND_Input* input = nullptr;
  TRITONSERVER_Error* err =
      TRITONBACKEND_RequestInputByIndex(request, 0, &input);
  const int64_t* shape = nullptr;
  uint32_t dims = 0;
  if (err == nullptr) {
    err = TRITONBACKEND_InputProperties(
        input, nullptr, nullptr, &shape, &dims, nullptr, nullptr);
  }
  uint64_t batch_size = 0;
  if (err == nullptr && shape != nullptr && dims > 0 && shape[0] > 0) {
    batch_size = static_cast<uint64_t>(shape[0]);
  }
  LOG_IF_ERROR(err, "failed reading model_connect batch size");
  return batch_size;
}

}  // namespace

// Borrowed host view of one Triton string. Valid while tensor.bytes is alive.
// ValidateInput has already accepted the buffer. text_generation copies it to
// the std::string that generate() requires.
void
TextView(const NamedTensor& tensor, trtmc_string_view& text)
{
  const char* bytes = reinterpret_cast<const char*>(tensor.bytes.data());
  if (tensor.bytes.size() < sizeof(uint32_t)) {
    text.data = nullptr;
    text.size = 0;
    return;
  }
  // A Triton string is a 4-byte length followed by that many bytes.
  const uint32_t length = *reinterpret_cast<const uint32_t*>(bytes);
  text.data = bytes + sizeof(uint32_t);
  text.size = length;
}

// The bytes of one Triton string input. A host buffer is checked in place.
// A GPU buffer is read to the host for the check.
// FIXME: Duplicates the buffer gathering of GetContiguousInputContent in the
// TensorFlow and PyTorch backends, and copies a GPU input once here and again
// in SetInputTensors. Validate the host copy in tensor.bytes instead.
TRITONSERVER_Error*
ValidateStringInput(
    TRITONBACKEND_Input* input, const char* name,
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    const uint32_t dims, const int max_batch_size)
{
  if (datatype != TRITONSERVER_TYPE_BYTES) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "text input must be TYPE_STRING");
  }
  size_t expected_count = 1;
  if (max_batch_size > 0) {
    if (shape == nullptr || dims != 2 || shape[0] < 1 || shape[1] != 1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "text input must be one string per batch element");
    }
    expected_count = static_cast<size_t>(shape[0]);
  } else if (shape == nullptr || dims != 1 || shape[0] != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "text input must contain one string");
  }

  uint32_t buffer_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      input, nullptr, nullptr, nullptr, nullptr, nullptr, &buffer_count));
  const char* bytes = nullptr;
  size_t size = 0;
  std::vector<char> gathered;
  for (uint32_t i = 0; i < buffer_count; ++i) {
    const void* buffer = nullptr;
    uint64_t buffer_bytes = 0;
    TRITONSERVER_MemoryType memory = TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, i, &buffer, &buffer_bytes, &memory, &memory_id));
    if (buffer_bytes == 0) {
      continue;
    }
    // TRITONBACKEND_InputBuffer reports the size as uint64_t.
    const size_t length = static_cast<size_t>(buffer_bytes);
    const bool host = memory == TRITONSERVER_MEMORY_CPU ||
                      memory == TRITONSERVER_MEMORY_CPU_PINNED;
    if (buffer_count == 1 && host) {
      bytes = static_cast<const char*>(buffer);
      size = length;
      break;
    }
    const size_t offset = gathered.size();
    gathered.resize(offset + length);
    if (host) {
      memcpy(gathered.data() + offset, buffer, length);
    } else {
      RETURN_IF_CUDA_ERROR(
          cudaMemcpy(
              gathered.data() + offset, buffer, length, cudaMemcpyDeviceToHost),
          TRITONSERVER_ERROR_INTERNAL,
          std::string("failed to copy model_connect string input"));
    }
  }
  if (!gathered.empty()) {
    bytes = gathered.data();
    size = gathered.size();
  }
  std::vector<std::pair<const char*, const uint32_t>> strings;
  return backend::ValidateStringBuffer(
      bytes, size, expected_count, name != nullptr ? name : "", &strings);
}

TRITONSERVER_Error*
ModelInstanceState::LoadSdkTask(std::shared_ptr<trtmc::ITask>* task)
{
  const std::string& bundle_task = model_state_->BundleTask();
  const Parameters& parameters = model_state_->Parameters();
  try {
    *task = trtmc::load_task(
        model_state_->BundlePath(), {}, parameters.kv_cache_size_bytes, {},
        false, parameters.kv_cache_memory_ratio);
  }
  catch (const std::bad_alloc&) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNAVAILABLE,
        (bundle_task + " load ran out of memory").c_str());
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to load ") + bundle_task + " bundle: " + ex.what())
            .c_str());
  }
  return nullptr;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  if (Kind() != TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', model_connect backend supports only GPU device")
            .c_str()));
  }
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  *state = nullptr;
  const std::string& bundle_task = model_state->BundleTask();
  try {
    std::unique_ptr<ModelInstanceState> instance_state;
    switch (TaskKindFor(bundle_task)) {
      case TaskKind::kStructuredRanking:
        instance_state = std::make_unique<RankingInstanceState>(
            model_state, triton_model_instance);
        break;
      case TaskKind::kTextGeneration:
        instance_state = std::make_unique<TextGenerationInstanceState>(
            model_state, triton_model_instance);
        break;
      case TaskKind::kTextContinuation:
        instance_state = std::make_unique<TextContinuationInstanceState>(
            model_state, triton_model_instance);
        break;
      default:
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            TOSTRING(
                "model_connect bundle task '" << bundle_task
                                              << "' is not supported")
                .c_str());
    }

    // Execute is another thread and sets this device again.
    RETURN_IF_CUDA_ERROR(
        cudaSetDevice(instance_state->DeviceId()), TRITONSERVER_ERROR_INTERNAL,
        std::string("failed to set model_connect CUDA device"));
    RETURN_IF_ERROR(instance_state->LoadModel());

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        TOSTRING(
            "loaded model_connect bundle '"
            << model_state->BundlePath() << "' family '"
            << model_state->BundleFamily() << "' task '" << bundle_task << "'")
            .c_str());
    *state = instance_state.release();
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to create model_connect instance: ") + ex.what())
            .c_str());
  }
  return nullptr;
}

// One borrowed text or token-id result. keep owns it until the response write
// finishes.
TRITONSERVER_Error*
AppendTextOutput(
    ModelState* model_state, const trtmc_string_view& text,
    const int32_t* token_ids, uint64_t token_count, std::shared_ptr<void> keep,
    std::vector<NamedTensor>* outputs)
{
  // ModelState requires exactly one output for a text task.
  const NamedTensor& output = model_state->Outputs()[0];
  NamedTensor tensor;
  tensor.name = output.name;
  tensor.external_keep = std::move(keep);
  if (output.datatype == TRITONSERVER_TYPE_BYTES) {
    tensor.datatype = TRITONSERVER_TYPE_BYTES;
    tensor.shape = {1};
    tensor.strings = {text};
  } else {
    tensor.datatype = TRITONSERVER_TYPE_INT32;
    if (token_count > 0 && token_ids == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "text result tokens are null");
    }
    tensor.shape = {static_cast<int64_t>(token_count)};
    tensor.external = token_ids;
    tensor.external_size = token_count * sizeof(int32_t);
  }
  outputs->push_back(std::move(tensor));
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ValidateRequest(TRITONBACKEND_Request* request) const
{
  uint32_t count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &count));
  for (uint32_t i = 0; i < count; ++i) {
    TRITONBACKEND_Input* input = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(request, i, &input));
    const char* name = nullptr;
    TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
    const int64_t* shape = nullptr;
    uint32_t dims = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &name, &datatype, &shape, &dims, nullptr, nullptr));
    RETURN_IF_ERROR(ValidateInput(input, name, datatype, shape, dims));
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    TRITONBACKEND_Request* request, TRITONBACKEND_Response** response,
    std::vector<NamedTensor>* tensors)
{
  TRITONBACKEND_Request* requests[1] = {request};
  std::vector<TRITONBACKEND_Response*> responses(1, *response);
  cudaStream_t copy_stream = nullptr;
  BackendInputCollector collector(
      requests, 1, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), copy_stream, nullptr, nullptr, 0,
      HostPolicyName().c_str(), false, false);
  tensors->reserve(model_state_->Inputs().size());
  for (const NamedTensor& input : model_state_->Inputs()) {
    NamedTensor tensor;
    RETURN_IF_ERROR(SetInputTensor(request, input, &collector, &tensor));
    RETURN_IF_ERROR(SentCopyError(response, responses));
    tensors->push_back(std::move(tensor));
  }
  const bool need_sync = collector.Finalize();
  RETURN_IF_ERROR(SentCopyError(response, responses));
  if (need_sync) {
    RETURN_IF_CUDA_ERROR(
        cudaStreamSynchronize(copy_stream), TRITONSERVER_ERROR_INTERNAL,
        std::string("failed to synchronize model_connect copy"));
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::FindRequestedOutputs(
    TRITONBACKEND_Request* request,
    const std::vector<NamedTensor>& output_tensors,
    std::vector<const NamedTensor*>* pending) const
{
  const std::vector<NamedTensor>& outputs = model_state_->Outputs();
  pending->reserve(outputs.size());
  uint32_t requested = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestOutputCount(request, &requested));
  for (uint32_t i = 0; i < requested; ++i) {
    const char* name = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(request, i, &name));
    const NamedTensor* output = nullptr;
    // TODO: use hashmap for quicker lookup
    for (const NamedTensor& candidate : outputs) {
      if (candidate.name == name) {
        output = &candidate;
        break;
      }
    }
    const NamedTensor* found = nullptr;
    for (const NamedTensor& candidate : output_tensors) {
      if (candidate.name == name) {
        found = &candidate;
        break;
      }
    }
    if (found == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("model_connect did not return output '") + name + "'")
              .c_str());
    }
    RETURN_IF_ERROR(CheckNamedTensor(*found, output->datatype));
    pending->push_back(found);
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    TRITONBACKEND_Request* request, TRITONBACKEND_Response** response,
    const std::vector<NamedTensor>& output_tensors)
{
  TRITONBACKEND_Request* requests[1] = {request};
  std::vector<TRITONBACKEND_Response*> responses(1, *response);
  cudaStream_t copy_stream = nullptr;
  std::vector<const NamedTensor*> pending;
  RETURN_IF_ERROR(FindRequestedOutputs(request, output_tensors, &pending));

  // Strings are packed into the response. Numeric outputs use the responder.
  BackendOutputResponder responder(
      requests, 1, &responses, model_state_->TritonMemoryManager(), false,
      model_state_->EnablePinnedOutput(), copy_stream, nullptr, false);
  for (const NamedTensor* tensor : pending) {
    if (tensor->datatype == TRITONSERVER_TYPE_BYTES) {
      RETURN_IF_ERROR(SetStringOutputBuffer(*response, *tensor));
      continue;
    }
    std::vector<int64_t> shape = tensor->shape;
    const char* data =
        tensor->external_keep
            ? static_cast<const char*>(tensor->external)
            : (tensor->bytes.empty()
                   ? nullptr
                   : reinterpret_cast<const char*>(tensor->bytes.data()));
    responder.ProcessTensor(
        tensor->name, tensor->datatype, shape, data, TRITONSERVER_MEMORY_CPU,
        0);
    RETURN_IF_ERROR(SentCopyError(response, responses));
  }
  const bool need_sync = responder.Finalize();
  RETURN_IF_ERROR(SentCopyError(response, responses));
  if (need_sync) {
    RETURN_IF_CUDA_ERROR(
        cudaStreamSynchronize(copy_stream), TRITONSERVER_ERROR_INTERNAL,
        std::string("failed to synchronize model_connect copy"));
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::RunRequest(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>& responses, const RunFn& run)
{
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response** response = &responses[r];
    if (*response == nullptr) {
      continue;
    }
    TRITONSERVER_Error* err = nullptr;
    try {
      std::vector<NamedTensor> inputs;
      err = SetInputTensors(requests[r], response, &inputs);
      // Text results stay alive here until the write below returns.
      std::vector<NamedTensor> outputs;
      if (err == nullptr) {
        if (compute_start_ns_ == 0) {
          SET_TIMESTAMP(compute_start_ns_);
        }
        err = run(inputs, &outputs);
        SET_TIMESTAMP(compute_end_ns_);
      }
      if (err == nullptr) {
        err = ReadOutputTensors(requests[r], response, outputs);
      }
    }
    catch (const std::exception&) {
      err = CaughtRequestError();
    }
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
  }
  return nullptr;
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_IF_CUDA_ERROR(
      cudaSetDevice(DeviceId()), "failed to set model_connect CUDA device");

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);
  compute_start_ns_ = 0;
  compute_end_ns_ = 0;

  // A create failure is logged. A validation failure or exception is sent on
  // that response. Neither request is run. Each request's batch size counts as
  // one with max_batch_size 0, otherwise its leading dimension.
  std::vector<TRITONBACKEND_Response*> responses(request_count, nullptr);
  uint64_t total_batch_size = 0;
  const int max_batch_size = model_state_->MaxBatchSize();
  for (uint32_t r = 0; r < request_count; ++r) {
    total_batch_size += RequestBatchSize(requests[r], max_batch_size);
    TRITONSERVER_Error* err =
        TRITONBACKEND_ResponseNew(&responses[r], requests[r]);
    if (err != nullptr) {
      responses[r] = nullptr;
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
      continue;
    }
    try {
      err = ValidateRequest(requests[r]);
    }
    catch (const std::exception&) {
      err = CaughtRequestError();
    }
    RESPOND_AND_SET_NULL_IF_ERROR(&responses[r], err);
  }

  TRITONSERVER_Error* err = nullptr;
  try {
    err = Execute(requests, request_count, responses);
  }
  catch (const std::exception&) {
    err = CaughtRequestError();
  }
  RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses, request_count, err);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);
  uint64_t compute_start_ns =
      compute_start_ns_ != 0 ? compute_start_ns_ : exec_end_ns;
  uint64_t compute_end_ns =
      compute_end_ns_ != 0 ? compute_end_ns_ : exec_end_ns;

  // Leave successful responses set so statistics can tell them from
  // requests that already failed.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send model_connect backend response");
    }
  }

  // After send, including requests that failed.
  for (uint32_t r = 0; r < request_count; ++r) {
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), requests[r], responses[r] != nullptr,
            exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(
            requests[r], TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
}

}  // namespace triton::backend::tensorrt::model_connect
