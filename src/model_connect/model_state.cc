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

#include "model_connect/model_state.hh"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "model_connect/trtmc_utils.hh"
#include "tensorrt.hh"
#include "triton/backend/backend_common.h"
#include "trtmc/trtmc.h"

namespace triton::backend::tensorrt::model_connect {

namespace {

TRITONSERVER_Error*
ParseTensorGroup(
    common::TritonJson::Value& model_config, const char* group,
    std::vector<NamedTensor>* tensors)
{
  common::TritonJson::Value array;
  RETURN_IF_ERROR(model_config.MemberAsArray(group, &array));
  for (size_t i = 0; i < array.ArraySize(); ++i) {
    common::TritonJson::Value tensor;
    RETURN_IF_ERROR(array.IndexAsObject(i, &tensor));

    NamedTensor parsed;
    const char* data_type = nullptr;
    size_t data_type_len = 0;
    RETURN_IF_ERROR(
        tensor.MemberAsString("data_type", &data_type, &data_type_len));
    parsed.datatype = ModelConfigDataTypeToTritonServerDataType(
        std::string(data_type, data_type_len));

    const char* tensor_name = nullptr;
    size_t tensor_name_len = 0;
    RETURN_IF_ERROR(
        tensor.MemberAsString("name", &tensor_name, &tensor_name_len));
    parsed.name.assign(tensor_name, tensor_name_len);

    // Without allow_ragged_batch, Triton batches only requests whose
    // dimensions after the batch dimension match.
    common::TritonJson::Value ragged;
    bool allow_ragged_batch = false;
    if (tensor.Find("allow_ragged_batch", &ragged)) {
      RETURN_IF_ERROR(ragged.AsBool(&allow_ragged_batch));
    }
    if (allow_ragged_batch) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          ("model_connect does not support allow_ragged_batch for " +
           std::string(group) + " '" + parsed.name + "'")
              .c_str());
    }
    tensors->push_back(std::move(parsed));
  }
  return nullptr;
}

// Every input and output is a numeric type.
TRITONSERVER_Error*
RequireNumericTensors(
    const std::vector<NamedTensor>& inputs,
    const std::vector<NamedTensor>& outputs)
{
  const std::pair<const char*, const std::vector<NamedTensor>*> groups[] = {
      {"input", &inputs}, {"output", &outputs}};
  for (const auto& [group, tensors] : groups) {
    for (const NamedTensor& tensor : *tensors) {
      trtmc::DType dtype = trtmc::DType::kFloat32;
      if (!NumericType(tensor.datatype, &dtype)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("model_connect ") + group + " '" + tensor.name +
             "' must be TYPE_FP32, TYPE_FP16, TYPE_BF16, TYPE_INT8, "
             "TYPE_INT32, or TYPE_INT64")
                .c_str());
      }
    }
  }
  return nullptr;
}

// Every input and output holds TYPE_STRING text or TYPE_INT32 token ids. An
// input holds token ids only where input_token_ids is set.
TRITONSERVER_Error*
RequireTextTensors(
    const std::vector<NamedTensor>& inputs,
    const std::vector<NamedTensor>& outputs, const bool input_token_ids)
{
  const std::tuple<const char*, const std::vector<NamedTensor>*, bool>
      groups[] = {
          {"input", &inputs, input_token_ids}, {"output", &outputs, true}};
  for (const auto& [group, tensors, allow_token_ids] : groups) {
    for (const NamedTensor& tensor : *tensors) {
      if (tensor.datatype == TRITONSERVER_TYPE_BYTES ||
          (allow_token_ids && tensor.datatype == TRITONSERVER_TYPE_INT32)) {
        continue;
      }
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("model_connect text task ") + group + " '" +
           tensor.name + "' must be TYPE_STRING" +
           (allow_token_ids ? " or TYPE_INT32" : ""))
              .c_str());
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ReadBundleTask(const std::string& path, std::string* task, std::string* family)
{
  const trtmc_core_api_v1* trtmc = nullptr;
  RETURN_IF_ERROR(GetTrtmcApi(&trtmc));

  trtmc_error* error = nullptr;
  trtmc_bundle* bundle = nullptr;
  const trtmc_string_view bundle_path{path.data(), path.size()};
  RETURN_IF_TRTMC_ERROR(
      trtmc, error, trtmc->bundle_open(bundle_path, &bundle, &error));
  trtmc_bundle_info_v1 bundle_info{};
  const trtmc_status status = trtmc->bundle_info(bundle, &bundle_info, &error);
  TRITONSERVER_Error* info_error = TrtmcStatusError(trtmc, status, error);
  if (info_error == nullptr) {
    if (bundle_info.task.data != nullptr && bundle_info.task.size > 0) {
      task->assign(
          bundle_info.task.data, static_cast<size_t>(bundle_info.task.size));
    }
    if (bundle_info.family.data != nullptr && bundle_info.family.size > 0) {
      family->assign(
          bundle_info.family.data,
          static_cast<size_t>(bundle_info.family.size));
    }
  }
  trtmc->bundle_release(bundle);
  RETURN_IF_ERROR(info_error);
  if (task->empty()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "model_connect bundle header task is empty");
  }
  return nullptr;
}

}  // namespace

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  *state = nullptr;
  try {
    std::unique_ptr<ModelState> model_state(new ModelState(triton_model));
    RETURN_IF_ERROR(model_state->ValidateModelConfig());
    *state = model_state.release();
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to create model_connect model state: ") +
         ex.what())
            .c_str());
  }
  return nullptr;
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
}

TRITONSERVER_Error*
ModelState::ValidateTensors()
{
  RETURN_IF_ERROR(ParseTensorGroup(ModelConfig(), "input", &inputs_));
  RETURN_IF_ERROR(ParseTensorGroup(ModelConfig(), "output", &outputs_));
  const TaskKind kind = TaskKindFor(bundle_task_);
  switch (kind) {
    case TaskKind::kStructuredRanking:
      // A batched execute reads each request's batch size from its first
      // input.
      if (MaxBatchSize() > 0 && inputs_.empty()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "model_connect ranking model with max_batch_size greater than 0 "
            "requires an input");
      }
      RETURN_IF_ERROR(RequireNumericTensors(inputs_, outputs_));
      break;
    case TaskKind::kTextGeneration:
    case TaskKind::kTextContinuation: {
      if (inputs_.size() != 1 || outputs_.size() != 1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "model_connect text task requires one input and one output");
      }
      // text_generation runs one prompt string per request.
      const bool generation = kind == TaskKind::kTextGeneration;
      if (generation && MaxBatchSize() > 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("model_connect task '") + bundle_task_ +
             "' requires max_batch_size 0")
                .c_str());
      }
      RETURN_IF_ERROR(RequireTextTensors(inputs_, outputs_, !generation));
      break;
    }
    default:
      break;
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  std::string bundle_name = "model.bundle";
  common::TritonJson::Value filename;
  if (ModelConfig().Find("default_model_filename", &filename)) {
    std::string default_model_filename;
    RETURN_IF_ERROR(filename.AsString(&default_model_filename));
    if (!default_model_filename.empty()) {
      bundle_name = std::move(default_model_filename);
    }
  }

  // Validate bundle path
  if (!bundle_name.empty() && bundle_name.front() == '/') {
    bundle_path_ = bundle_name;
  } else {
    bundle_path_ =
        JoinPath({RepositoryPath(), std::to_string(Version()), bundle_name});
  }
  std::error_code exists_error;
  if (!std::filesystem::is_regular_file(bundle_path_, exists_error)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        ("model_connect bundle '" + bundle_path_ +
         "' was not found for model '" + Name() + "'")
            .c_str());
  }

  // Validate bundle task
  RETURN_IF_ERROR(ReadBundleTask(bundle_path_, &bundle_task_, &bundle_family_));
  if (TaskKindFor(bundle_task_) == TaskKind::kUnsupported) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("model_connect bundle task '") + bundle_task_ +
         "' is not supported")
            .c_str());
  }

  // Validate parameters
  RETURN_IF_ERROR(ParseParameters());
  // text_continuation's model_load does not accept either cache budget.
  if (TaskKindFor(bundle_task_) == TaskKind::kTextContinuation &&
      (params.kv_cache_size_bytes != 0 || params.kv_cache_memory_ratio != 0)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("model_connect task '") + bundle_task_ +
         "' does not accept 'kv_cache_size_bytes' or "
         "'kv_cache_memory_ratio'")
            .c_str());
  }
  return ValidateTensors();
}

TRITONSERVER_Error*
ModelState::ParseParameters()
{
  common::TritonJson::Value params;
  if (!ModelConfig().Find("parameters", &params)) {
    return nullptr;
  }

  RETURN_IF_ERROR(TryParseModelStringParameter(
      params, "kv_cache_size_bytes", &this->params.kv_cache_size_bytes,
      uint64_t{0}));
  RETURN_IF_ERROR(TryParseModelStringParameter(
      params, "kv_cache_memory_ratio", &this->params.kv_cache_memory_ratio,
      0.0));

  if (this->params.kv_cache_size_bytes != 0) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, TOSTRING(
                                   "'kv_cache_size_bytes' set to '"
                                   << this->params.kv_cache_size_bytes
                                   << "' for model '" << Name() << "'")
                                   .c_str());
  }
  if (this->params.kv_cache_memory_ratio != 0.0) {
    if (!std::isfinite(this->params.kv_cache_memory_ratio) ||
        this->params.kv_cache_memory_ratio < 0.0 ||
        this->params.kv_cache_memory_ratio > 1.0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          ("Invalid value for 'kv_cache_memory_ratio': '" +
           std::to_string(this->params.kv_cache_memory_ratio) +
           "' for model '" + Name() + "'. Must be a float in (0, 1].")
              .c_str());
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, TOSTRING(
                                   "'kv_cache_memory_ratio' set to '"
                                   << this->params.kv_cache_memory_ratio
                                   << "' for model '" << Name() << "'")
                                   .c_str());
  }
  return nullptr;
}

}  // namespace triton::backend::tensorrt::model_connect
