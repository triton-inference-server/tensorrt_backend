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

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "model_connect/trtmc_utils.hh"
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
  for (size_t index = 0; index < array.ArraySize(); ++index) {
    common::TritonJson::Value tensor;
    RETURN_IF_ERROR(array.IndexAsObject(index, &tensor));

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
    tensors->push_back(std::move(parsed));
  }
  return nullptr;
}

TRITONSERVER_Error*
RequireNumericTensors(
    const std::vector<NamedTensor>& tensors, const char* group)
{
  for (const NamedTensor& tensor : tensors) {
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
  return nullptr;
}

TRITONSERVER_Error*
ReadBundleTask(const std::string& path, std::string* task, std::string* family)
{
  const trtmc_core_api_v1* trtmc = nullptr;
  trtmc_status status =
      trtmc_get_api(TRTMC_API_VERSION_MAJOR, TRTMC_API_VERSION_MINOR, &trtmc);
  if (status != TRTMC_OK || trtmc == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "trtmc_get_api(" TRTMC_API_STR(TRTMC_API_VERSION_MAJOR) ", "
        TRTMC_API_STR(TRTMC_API_VERSION_MINOR) ") did not return the "
        "TensorRT Model Connect trtmc API");
  }

  trtmc_error* error = nullptr;
  trtmc_bundle* bundle = nullptr;
  const trtmc_string_view bundle_path{path.data(), path.size()};
  // Read the SDK error after the call returns.
  status = trtmc->bundle_open(bundle_path, &bundle, &error);
  RETURN_IF_ERROR(TrtmcStatusError(trtmc, status, error));
  trtmc_bundle_info_v1 bundle_info{};
  status = trtmc->bundle_info(bundle, &bundle_info, &error);
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
  if (TaskKindFor(bundle_task_) != TaskKind::kStructuredRanking) {
    return nullptr;
  }
  RETURN_IF_ERROR(RequireNumericTensors(inputs_, "input"));
  return RequireNumericTensors(outputs_, "output");
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

  RETURN_IF_ERROR(ReadBundleTask(bundle_path_, &bundle_task_, &bundle_family_));
  if (TaskKindFor(bundle_task_) == TaskKind::kUnsupported) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("model_connect bundle task '") + bundle_task_ +
         "' is not supported")
            .c_str());
  }
  return ValidateTensors();
}

}  // namespace triton::backend::tensorrt::model_connect
