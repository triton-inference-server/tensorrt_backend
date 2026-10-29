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
#include <functional>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "model_connect/trtmc_utils.hh"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/structured_ranking.h"
#include "trtmc/trtmc.h"

namespace triton::backend::tensorrt::model_connect {

namespace {

// Borrowed host view of one Triton string. Valid while tensor.bytes is alive.
// ValidateInputs has already accepted the buffer. text_continuation passes
// the view to text->run. text_generation copies it to the std::string that
// generate() requires.
void
TextView(const NamedTensor& tensor, trtmc_string_view* text)
{
  const char* bytes = reinterpret_cast<const char*>(tensor.bytes.data());
  if (tensor.bytes.size() < sizeof(uint32_t)) {
    text->data = nullptr;
    text->size = 0;
    return;
  }
  const uint32_t length = *reinterpret_cast<const uint32_t*>(bytes);
  text->data = bytes + sizeof(uint32_t);
  text->size = length;
}

// Borrowed host int32 ids. text->run reads them before it returns.
// The caller has already selected an INT32 input.
TRITONSERVER_Error*
TokenView(const NamedTensor& tensor, trtmc_i32_view* tokens)
{
  if (tensor.shape.size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "text continuation input must be one rank-1 token-id sequence");
  }
  const uint64_t count = static_cast<uint64_t>(tensor.shape[0]);
  tokens->data = count == 0
                     ? nullptr
                     : reinterpret_cast<const int32_t*>(tensor.bytes.data());
  tokens->size = count;
  return nullptr;
}

// Copies the text result from tensor.external to the response buffer.
// TYPE_BYTES has no fixed element size, so the responder cannot write it.
// tensor.external_keep keeps the result alive through the copy.
TRITONSERVER_Error*
WriteTextResult(TRITONBACKEND_Response* response, const NamedTensor& tensor)
{
  if (tensor.external_size > std::numeric_limits<uint32_t>::max()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "text result string is too long");
  }
  if (tensor.external_size > 0 && tensor.external == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "text result text is null");
  }
  const uint32_t length = static_cast<uint32_t>(tensor.external_size);
  const int64_t shape = 1;
  TRITONBACKEND_Output* output = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(
      response, &output, tensor.name.c_str(), TRITONSERVER_TYPE_BYTES, &shape,
      1));
  void* out_buffer = nullptr;
  TRITONSERVER_MemoryType out_memory = TRITONSERVER_MEMORY_CPU;
  int64_t out_memory_id = 0;
  const uint64_t byte_size = sizeof(uint32_t) + static_cast<uint64_t>(length);
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(
      output, &out_buffer, byte_size, &out_memory, &out_memory_id));
  bool cuda_used = false;
  RETURN_IF_ERROR(CopyBuffer(
      tensor.name, TRITONSERVER_MEMORY_CPU, 0, out_memory, out_memory_id,
      sizeof(length), &length, out_buffer, nullptr, &cuda_used));
  if (length > 0) {
    bool text_cuda = false;
    RETURN_IF_ERROR(CopyBuffer(
        tensor.name, TRITONSERVER_MEMORY_CPU, 0, out_memory, out_memory_id,
        length, tensor.external,
        static_cast<char*>(out_buffer) + sizeof(length), nullptr, &text_cuda));
    cuda_used = cuda_used || text_cuda;
  }
  if (cuda_used) {
    RETURN_IF_CUDA_ERROR(
        cudaStreamSynchronize(nullptr), TRITONSERVER_ERROR_INTERNAL,
        std::string("failed to synchronize model_connect copy"));
  }
  return nullptr;
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
// ValidateInputs has already accepted the request input.
TRITONSERVER_Error*
SetInputTensors(
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

// The bytes of one Triton string input. A host buffer is checked in place.
// A GPU buffer is read to the host for the check.
TRITONSERVER_Error*
ValidateStringInput(TRITONBACKEND_Input* input, const char* name)
{
  uint32_t buffer_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      input, nullptr, nullptr, nullptr, nullptr, nullptr, &buffer_count));
  const char* bytes = nullptr;
  size_t size = 0;
  std::vector<char> gathered;
  for (uint32_t index = 0; index < buffer_count; ++index) {
    const void* buffer = nullptr;
    uint64_t buffer_bytes = 0;
    TRITONSERVER_MemoryType memory = TRITONSERVER_MEMORY_CPU;
    int64_t memory_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, index, &buffer, &buffer_bytes, &memory, &memory_id));
    if (buffer_bytes == 0) {
      continue;
    }
    const bool host = memory == TRITONSERVER_MEMORY_CPU ||
                      memory == TRITONSERVER_MEMORY_CPU_PINNED;
    if (buffer_count == 1 && host) {
      bytes = static_cast<const char*>(buffer);
      size = static_cast<size_t>(buffer_bytes);
      break;
    }
    const size_t offset = gathered.size();
    gathered.resize(offset + static_cast<size_t>(buffer_bytes));
    if (host) {
      memcpy(
          gathered.data() + offset, buffer, static_cast<size_t>(buffer_bytes));
    } else {
      RETURN_IF_CUDA_ERROR(
          cudaMemcpy(
              gathered.data() + offset, buffer,
              static_cast<size_t>(buffer_bytes), cudaMemcpyDeviceToHost),
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
      bytes, size, 1, name != nullptr ? name : "", &strings);
}

// Each Triton request input, checked once for every task. After this returns,
// the inputs are copied and the task runs. A TYPE_BYTES input is one string.
// Any other input must be a numeric type.
TRITONSERVER_Error*
ValidateInputs(TRITONBACKEND_Request* request)
{
  uint32_t count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &count));
  for (uint32_t index = 0; index < count; ++index) {
    TRITONBACKEND_Input* input = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(request, index, &input));
    const char* name = nullptr;
    TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
    const int64_t* shape = nullptr;
    uint32_t dims = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &name, &datatype, &shape, &dims, nullptr, nullptr));
    if (datatype == TRITONSERVER_TYPE_BYTES) {
      if (dims != 1 || shape == nullptr || shape[0] != 1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "text input must contain one string");
      }
      RETURN_IF_ERROR(ValidateStringInput(input, name));
    } else {
      trtmc::DType dtype = trtmc::DType::kFloat32;
      if (!NumericType(datatype, &dtype)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unexpected datatype TYPE_") +
             TRITONSERVER_DataTypeString(datatype) + " for inference input '" +
             (name != nullptr ? name : "") + "'")
                .c_str());
      }
    }
  }
  return nullptr;
}

// The responder copies the byte size of the shape and does not look at the
// task buffer's length. TYPE_BYTES is sized by WriteTextResult.
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

// One loaded bundle. The task implementation is inside `run`; this struct
// has no member that belongs to a single task.
struct ModelInstanceState::Sdk {
  std::function<TRITONSERVER_Error*(
      std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs)>
      run;
};

struct TrtmcModel {
  const trtmc_core_api_v1* trtmc{nullptr};
  trtmc_model* model{nullptr};
  ~TrtmcModel()
  {
    if (trtmc != nullptr && model != nullptr) {
      trtmc->model_release(model);
    }
  }
};

// load_task for structured_ranking and text_generation. The shared_ptr keeps
// the ITask alive for the instance.
TRITONSERVER_Error*
LoadTaskObject(
    const std::string& path, const std::string& bundle_task,
    std::shared_ptr<trtmc::ITask>* trtmc_task)
{
  try {
    *trtmc_task = trtmc::load_task(path);
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

TRITONSERVER_Error* RunRankingTask(
    trtmc::IStructuredRanking* ranking, std::vector<NamedTensor>& inputs,
    std::vector<NamedTensor>* outputs);
TRITONSERVER_Error* RunTextTask(
    const trtmc_core_api_v1* trtmc, trtmc_model* model,
    const trtmc_text_continuation_api_v1* text,
    trtmc::ITextGeneration* generation, ModelState* model_state,
    const std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs);

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

ModelInstanceState::~ModelInstanceState() = default;

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  *state = nullptr;
  try {
    std::unique_ptr<ModelInstanceState> instance_state(
        new ModelInstanceState(model_state, triton_model_instance));
    RETURN_IF_ERROR(instance_state->LoadAPI());
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

TRITONSERVER_Error*
ModelInstanceState::LoadAPI()
{
  // Execute is another thread and sets this device again.
  RETURN_IF_CUDA_ERROR(
      cudaSetDevice(DeviceId()), TRITONSERVER_ERROR_INTERNAL,
      std::string("failed to set model_connect CUDA device"));

  const trtmc_core_api_v1* trtmc = nullptr;
  const trtmc_status api_status =
      trtmc_get_api(TRTMC_API_VERSION_MAJOR, TRTMC_API_VERSION_MINOR, &trtmc);
  if (api_status != TRTMC_OK || trtmc == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "trtmc_get_api(" TRTMC_API_STR(TRTMC_API_VERSION_MAJOR) ", "
        TRTMC_API_STR(TRTMC_API_VERSION_MINOR) ") did not return the "
        "TensorRT Model Connect trtmc API");
  }

  const std::string& bundle_task = model_state_->BundleTask();
  const std::string& family = model_state_->BundleFamily();
  auto sdk = std::make_unique<Sdk>();
  switch (TaskKindFor(bundle_task)) {
    case TaskKind::kStructuredRanking: {
      // score() takes the numeric host tensors.
      std::shared_ptr<trtmc::ITask> trtmc_task;
      RETURN_IF_ERROR(
          LoadTaskObject(model_state_->BundlePath(), bundle_task, &trtmc_task));
      auto* ranking =
          dynamic_cast<trtmc::IStructuredRanking*>(trtmc_task.get());
      if (ranking == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "bundle task structured_ranking did not load IStructuredRanking");
      }
      sdk->run = [trtmc_task, ranking](
                     std::vector<NamedTensor>& inputs,
                     std::vector<NamedTensor>* outputs) {
        return RunRankingTask(ranking, inputs, outputs);
      };
      break;
    }
    case TaskKind::kTextGeneration: {
      // generate() takes one owned prompt string.
      std::shared_ptr<trtmc::ITask> trtmc_task;
      RETURN_IF_ERROR(
          LoadTaskObject(model_state_->BundlePath(), bundle_task, &trtmc_task));
      auto* generation =
          dynamic_cast<trtmc::ITextGeneration*>(trtmc_task.get());
      if (generation == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "bundle task text_generation did not load ITextGeneration");
      }
      ModelState* model_state = model_state_;
      sdk->run = [trtmc_task, generation, model_state](
                     std::vector<NamedTensor>& inputs,
                     std::vector<NamedTensor>* outputs) {
        return RunTextTask(
            nullptr, nullptr, nullptr, generation, model_state, inputs,
            outputs);
      };
      break;
    }
    case TaskKind::kTextContinuation: {
      // text->run borrows one UTF-8 or token-id prefix.
      auto trtmc_model = std::make_shared<TrtmcModel>();
      trtmc_model->trtmc = trtmc;
      trtmc_error* error = nullptr;
      const std::string& bundle_path = model_state_->BundlePath();
      const trtmc_string_view bundle{bundle_path.data(), bundle_path.size()};
      RETURN_IF_ERROR(TrtmcStatusError(
          trtmc,
          trtmc->model_load(bundle, nullptr, &trtmc_model->model, &error),
          error));

      const trtmc_string_view task{
          TRTMC_TASK_TEXT_CONTINUATION,
          sizeof(TRTMC_TASK_TEXT_CONTINUATION) - 1};
      const trtmc_api_header* api = nullptr;
      RETURN_IF_ERROR(TrtmcStatusError(
          trtmc,
          trtmc->model_get_task_api(
              trtmc_model->model, task, TRTMC_API_VERSION_MAJOR,
              TRTMC_API_VERSION_MINOR, &api, &error),
          error));
      if (api == nullptr || api->major != TRTMC_API_VERSION_MAJOR ||
          api->minor != TRTMC_API_VERSION_MINOR ||
          api->byte_size < sizeof(trtmc_text_continuation_api_v1)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_UNSUPPORTED,
            "loaded bundle does not provide text_continuation "
            "API " TRTMC_API_STR(TRTMC_API_VERSION_MAJOR) "." TRTMC_API_STR(
                TRTMC_API_VERSION_MINOR));
      }
      const auto* text_api =
          reinterpret_cast<const trtmc_text_continuation_api_v1*>(api);

      ModelState* model_state = model_state_;
      sdk->run = [trtmc_model, text_api, model_state](
                     std::vector<NamedTensor>& inputs,
                     std::vector<NamedTensor>* outputs) {
        return RunTextTask(
            trtmc_model->trtmc, trtmc_model->model, text_api, nullptr,
            model_state, inputs, outputs);
      };
      break;
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          (std::string("model_connect bundle task '") + bundle_task +
           "' is not supported")
              .c_str());
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, (std::string("loaded model_connect bundle '") +
                              model_state_->BundlePath() + "' family '" +
                              family + "' task '" + bundle_task + "'")
                                 .c_str());
  sdk_ = std::move(sdk);
  return nullptr;
}

// Moves host inputs into score() and moves the scored tensors back out.
TRITONSERVER_Error*
RunRankingTask(
    trtmc::IStructuredRanking* ranking, std::vector<NamedTensor>& inputs,
    std::vector<NamedTensor>* outputs)
{
  // Move the host bytes in. score() returns its own buffers.
  std::vector<trtmc::FeatureTensor> features;
  features.reserve(inputs.size());
  for (NamedTensor& input : inputs) {
    trtmc::FeatureTensor feature;
    feature.name = std::move(input.name);
    feature.shape = std::move(input.shape);
    feature.bytes = std::move(input.bytes);
    // ValidateInputs accepted this datatype.
    NumericType(input.datatype, &feature.dtype);
    features.push_back(std::move(feature));
  }
  std::vector<trtmc::FeatureTensor> scored = ranking->score(features);
  outputs->reserve(scored.size());
  for (trtmc::FeatureTensor& feature : scored) {
    NamedTensor output;
    output.name = std::move(feature.name);
    output.shape = std::move(feature.shape);
    output.bytes = std::move(feature.bytes);
    if (!TritonType(feature.dtype, &output.datatype)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unexpected datatype for inference output '") +
           output.name + "'")
              .c_str());
    }
    outputs->push_back(std::move(output));
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
  if (model_state->Outputs().size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "model_connect text task requires one output");
  }
  const NamedTensor& output = model_state->Outputs()[0];
  NamedTensor tensor;
  tensor.name = output.name;
  tensor.external_keep = std::move(keep);
  if (output.datatype == TRITONSERVER_TYPE_BYTES) {
    tensor.datatype = TRITONSERVER_TYPE_BYTES;
    tensor.shape = {1};
    tensor.external = text.data;
    tensor.external_size = text.size;
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

// One text input. text_generation copies the prompt from the input tensor to
// the std::string generate() requires. text_continuation borrows a UTF-8
// string or token ids.
TRITONSERVER_Error*
RunTextTask(
    const trtmc_core_api_v1* trtmc, trtmc_model* model,
    const trtmc_text_continuation_api_v1* text,
    trtmc::ITextGeneration* generation, ModelState* model_state,
    const std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs)
{
  // This call takes one string, or one rank-1 token-id sequence.
  if (inputs.size() != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "model_connect text task requires one input");
  }

  if (generation != nullptr) {
    if (inputs[0].datatype != TRITONSERVER_TYPE_BYTES) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "text input must be TYPE_STRING");
    }
    trtmc_string_view prompt_view{};
    TextView(inputs[0], &prompt_view);
    // generate() takes an owning std::string.
    const std::string prompt =
        prompt_view.size == 0
            ? std::string()
            : std::string(
                  prompt_view.data, static_cast<size_t>(prompt_view.size));
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = generation->default_max_new_tokens();
    // Kept until RunRequest finishes writing the response.
    auto result = std::make_shared<trtmc::TextResult>(
        generation->generate(prompt, config));
    const trtmc_string_view text_view{
        result->text.data(), static_cast<uint64_t>(result->text.size())};
    return AppendTextOutput(
        model_state, text_view,
        result->token_ids.empty() ? nullptr : result->token_ids.data(),
        result->token_ids.size(), result, outputs);
  } else {
    // text_continuation. The C API borrows one UTF-8 or token-id prefix.
    trtmc_text_continuation_request_v1 text_request{};
    if (inputs[0].datatype == TRITONSERVER_TYPE_INT32) {
      text_request.prefix.kind = TRTMC_TEXT_TOKEN_IDS;
      RETURN_IF_ERROR(TokenView(inputs[0], &text_request.prefix.as.token_ids));
    } else if (inputs[0].datatype == TRITONSERVER_TYPE_BYTES) {
      text_request.prefix.kind = TRTMC_TEXT_UTF8;
      TextView(inputs[0], &text_request.prefix.as.text);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "text input must be TYPE_STRING");
    }

    trtmc_result* raw = nullptr;
    trtmc_error* error = nullptr;
    const trtmc_status run_status =
        text->run(model, &text_request, nullptr, &raw, &error);
    // Releases the C result after RunRequest writes the response.
    std::shared_ptr<void> keep(raw, [trtmc](void* result) {
      if (trtmc != nullptr && result != nullptr) {
        trtmc->result_release(static_cast<trtmc_result*>(result));
      }
    });
    RETURN_IF_ERROR(TrtmcStatusError(trtmc, run_status, error));

    trtmc_text_result_view_v1 view{};
    RETURN_IF_ERROR(
        TrtmcStatusError(trtmc, text->result_view(raw, &view, &error), error));
    return AppendTextOutput(
        model_state, view.text, view.token_ids.data, view.token_ids.size,
        std::move(keep), outputs);
  }
}

// One request in, one task call, one request out.
TRITONSERVER_Error*
ModelInstanceState::RunRequest(
    TRITONBACKEND_Request* request, TRITONBACKEND_Response** response)
{
  TRITONBACKEND_MemoryManager* memory_manager =
      model_state_->TritonMemoryManager();
  const std::vector<NamedTensor>& inputs = model_state_->Inputs();
  const std::vector<NamedTensor>& outputs = model_state_->Outputs();

  RETURN_IF_ERROR(ValidateInputs(request));

  TRITONBACKEND_Request* requests[1] = {request};
  std::vector<TRITONBACKEND_Response*> responses(1, *response);
  cudaStream_t copy_stream = nullptr;

  // Copies each configured input from the request to a host NamedTensor.
  std::vector<NamedTensor> input_tensors;
  input_tensors.reserve(inputs.size());
  {
    BackendInputCollector collector(
        requests, 1, &responses, memory_manager,
        model_state_->EnablePinnedInput(), copy_stream, nullptr, nullptr, 0,
        HostPolicyName().c_str(), false, false);
    for (const NamedTensor& input : inputs) {
      NamedTensor tensor;
      RETURN_IF_ERROR(SetInputTensors(request, input, &collector, &tensor));
      RETURN_IF_ERROR(SentCopyError(response, responses));
      input_tensors.push_back(std::move(tensor));
    }
    const bool need_sync = collector.Finalize();
    RETURN_IF_ERROR(SentCopyError(response, responses));
    if (need_sync) {
      RETURN_IF_CUDA_ERROR(
          cudaStreamSynchronize(copy_stream), TRITONSERVER_ERROR_INTERNAL,
          std::string("failed to synchronize model_connect copy"));
    }
  }

  // Text results stay alive here until the write below returns.
  std::vector<NamedTensor> output_tensors;
  RETURN_IF_ERROR(sdk_->run(input_tensors, &output_tensors));
  std::vector<const NamedTensor*> pending;
  // Only outputs this request asked for.
  pending.reserve(outputs.size());
  uint32_t requested = 0;
  RETURN_IF_ERROR(TRITONBACKEND_RequestOutputCount(request, &requested));
  for (uint32_t index = 0; index < requested; ++index) {
    const char* name = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_RequestOutputName(request, index, &name));
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
    pending.push_back(found);
  }

  // Strings are packed into the response. Numeric outputs use the responder.
  BackendOutputResponder responder(
      requests, 1, &responses, memory_manager, false,
      model_state_->EnablePinnedOutput(), copy_stream, nullptr, false);
  for (const NamedTensor* tensor : pending) {
    if (tensor->datatype == TRITONSERVER_TYPE_BYTES) {
      RETURN_IF_ERROR(WriteTextResult(*response, *tensor));
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

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_IF_CUDA_ERROR(
      cudaSetDevice(DeviceId()), "failed to set model_connect CUDA device");

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // A create failure is logged. That request is not run.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response* response = nullptr;
    TRITONSERVER_Error* err = TRITONBACKEND_ResponseNew(&response, requests[r]);
    if (err == nullptr) {
      responses.push_back(response);
    } else {
      responses.push_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // max_batch_size 0 counts as one. Otherwise the leading dimension.
  uint64_t total_batch_size = 0;
  const int max_batch_size = model_state_->MaxBatchSize();
  for (uint32_t r = 0; r < request_count; ++r) {
    total_batch_size += RequestBatchSize(requests[r], max_batch_size);
  }

  // Each request is a separate task call, so this execute is one compute
  // window.
  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);
  for (uint32_t r = 0; r < request_count; ++r) {
    if (responses[r] == nullptr) {
      continue;
    }
    TRITONSERVER_Error* err = nullptr;
    try {
      err = RunRequest(requests[r], &responses[r]);
    }
    catch (const std::bad_alloc&) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNAVAILABLE,
          "model_connect request ran out of memory");
    }
    catch (const std::exception& ex) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("model_connect request failed: ") + ex.what()).c_str());
    }
    RESPOND_AND_SET_NULL_IF_ERROR(&responses[r], err);
  }
  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

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
