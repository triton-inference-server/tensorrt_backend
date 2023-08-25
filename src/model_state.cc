// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "model_state.h"

#include "instance_state.h"
#include "loader.h"
#include "tensorrt_utils.h"

namespace triton { namespace backend { namespace tensorrt {

class DeviceBlockingArbitrator : public ExecutionArbitrator {
 public:
  DeviceBlockingArbitrator() {}
  void RegisterInstance(
      const int device_id, ModelInstanceState* instance) override
  {
    // instance may be created in parallel
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = device_instances_.find(device_id);
    if (it == device_instances_.end()) {
      it = device_instances_.emplace(std::make_pair(device_id, InstanceList()))
               .first;
      // The first instance of the device should have 1-less items because
      // it is naturally ready for next execution. See
      // TRITONBACKEND_ModelInstanceExecute for the implication of acquisition.
      instance->SemaphorePtr()->Acquire();
    }
    it->second.instances_.emplace_back(instance);
  }

  std::pair<ModelInstanceState*, Semaphore*> ExecutionState(
      const int device_id, ModelInstanceState* instance) override
  {
    std::lock_guard<std::mutex> lk(mtx_);
    auto& il = device_instances_[device_id];
    auto current_instance = il.instances_[il.idx_];
    il.idx_ = ((il.idx_ + 1) % il.instances_.size());
    // In DEVICE_BLOCKING, the semaphore of the "next instance" will be used to
    // determine whether the execution should wait for resources to be ready for
    // next execution.
    // Here we simplify the "available instance" look-up by round-robin the
    // instances associated with the given device, as the executions are less
    // likely to finish out of order.
    auto semaphore = il.instances_[il.idx_]->SemaphorePtr();
    return {current_instance, semaphore};
  }

 private:
  // A list of instances associated with the same device ID an the index
  // to the instance to be used for execution.
  struct InstanceList {
    InstanceList() : idx_(0) {}
    std::vector<ModelInstanceState*> instances_;
    int idx_;
  };
  // map between device ID and associated instances
  std::map<int, InstanceList> device_instances_;

  std::mutex mtx_;
};

class InstanceBlockingArbitrator : public ExecutionArbitrator {
 public:
  InstanceBlockingArbitrator() {}
  void RegisterInstance(
      const int device_id, ModelInstanceState* instance) override
  {
    // all instances are naturally ready for next execution after
    // initialization.
    instance->SemaphorePtr()->Acquire();
  }
  std::pair<ModelInstanceState*, Semaphore*> ExecutionState(
      const int device_id, ModelInstanceState* instance) override
  {
    // In BLOCKING, the TRITONBACKEND_ModelInstance and executing instance are
    // coupled
    return {instance, instance->SemaphorePtr()};
  }
};


TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));

  // Server core already detects if a GPU is present and
  // corrects the instance groups before backend model is
  // initialized. Since the TensorRT backend only works with
  // GPU instances, check if the model has a KIND_GPU
  // instance group. If KIND_GPU is not present, skip
  // autocomplete as the model cannot be loaded.
  bool has_instance_kind_gpu = false;
  (*state)->InstanceHasKindGPU(&has_instance_kind_gpu);

  if (auto_complete_config && has_instance_kind_gpu) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());
    RETURN_IF_ERROR((*state)->SetTensorRTModelConfig());
  }

  RETURN_IF_ERROR((*state)->ValidateModelConfig());
  RETURN_IF_ERROR((*state)->ParseParameters());

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : TensorRTModel(triton_model), engine_sharing_(true)
{
  // Obtain backend configuration
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));
  TRITONBACKEND_ExecutionPolicy policy;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_BackendExecutionPolicy(backend, &policy));
  switch (policy) {
    case TRITONBACKEND_EXECUTION_BLOCKING:
      execution_arbitrator_.reset(new InstanceBlockingArbitrator());
      break;
    case TRITONBACKEND_EXECUTION_DEVICE_BLOCKING: {
      // [FIXME] ad-hoc way to resolve the issue that in sequence batching,
      // Triton instance must tie to the actual execution instance so that the
      // model state is properly maintained.
      triton::common::TritonJson::Value value;
      bool found_sequence_batching =
          ModelConfig().Find("sequence_batching", &value);
      if (found_sequence_batching) {
        execution_arbitrator_.reset(new InstanceBlockingArbitrator());
      } else {
        execution_arbitrator_.reset(new DeviceBlockingArbitrator());
      }
      break;
    }
  }
}

ModelState::~ModelState()
{
  for (auto& device_engine : device_engines_) {
    cudaSetDevice(device_engine.first.first);
    auto& runtime = device_engine.second.first;
    auto& engine = device_engine.second.second;
    // Need to reset explicitly to ensure proper destruction order
    if (engine != nullptr) {
      engine.reset();
    }
    if (runtime != nullptr) {
      runtime.reset();
    }
  }
}

TRITONSERVER_Error*
ModelState::CreateEngine(
    int gpu_device, const int64_t dla_core_id, const std::string& model_path,
    std::shared_ptr<nvinfer1::ICudaEngine>* engine)
{
  // TensorRT engine creation is not thread-safe, so multiple creations
  // are serialized with a global lock.
  static std::mutex global_context_mu;
  std::lock_guard<std::mutex> glock(global_context_mu);

  // Create shared engine for the device if haven't tried so.
  auto device_pair = std::make_pair(gpu_device, dla_core_id);
  auto eit = device_engines_.find(device_pair);
  if (eit == device_engines_.end()) {
    eit = device_engines_.emplace(device_pair, std::make_pair(nullptr, nullptr))
              .first;
  }

  // We share the engine (for models that don't have dynamic shapes) and
  // runtime across instances that have access to the same GPU/NVDLA.
  if (eit->second.second == nullptr) {
    auto cuerr = cudaSetDevice(gpu_device);
    if (cuerr != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to set device for ") + Name() + ": " +
           cudaGetErrorString(cuerr))
              .c_str());
    }

    const bool new_runtime = (eit->second.first == nullptr);
    RETURN_IF_ERROR(LoadPlan(
        model_path, dla_core_id, &eit->second.first, &eit->second.second,
        &tensorrt_logger_));
    *engine = eit->second.second;

    if (new_runtime) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Created new runtime on GPU device ") +
           std::to_string(gpu_device) + ", NVDLA core " +
           std::to_string(dla_core_id) + " for " + Name())
              .c_str());
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Created new engine on GPU device ") +
         std::to_string(gpu_device) + ", NVDLA core " +
         std::to_string(dla_core_id) + " for " + Name())
            .c_str());

    if (IsEngineSharingEnabled()) {
      // This logic runs at least once to validate whether the engine
      // can be shared.
      bool is_dynamic = false;
      for (int idx = 0; idx < eit->second.second->getNbBindings(); idx++) {
        auto dims = eit->second.second->getBindingDimensions(idx);
        // Detect whether dynamic or not
        if (ContainsWildcard(dims)) {
          is_dynamic = true;
          break;
        }
      }
      if (is_dynamic) {
        // Model with dynamic shapes can't share engine
        DisableEngineSharing();
      }
    }

    if (!IsEngineSharingEnabled()) {
      // Set engine to 'nullptr' as hint, but keep runtime as it
      // can be used repeatedly
      if (eit->second.second != nullptr) {
        eit->second.second.reset();
      }
    }
  } else {
    *engine = eit->second.second;
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ParseParameters()
{
  return nullptr;  // success
}


TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  int current_device;
  cudaError_t cuerr = cudaGetDevice(&current_device);
  if (cuerr != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to get current CUDA device ") + ": " +
         cudaGetErrorString(cuerr))
            .c_str());
  }

  int64_t device_id;
  triton::common::TritonJson::Value groups;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("instance_group", &groups));
  triton::common::TritonJson::Value group;
  RETURN_IF_ERROR(groups.IndexAsObject(0, &group));
  triton::common::TritonJson::Value gpus;
  RETURN_IF_ERROR(group.MemberAsArray("gpus", &gpus));
  RETURN_IF_ERROR(gpus.IndexAsInt(0, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string(
           "Setting the CUDA device to GPU" + std::to_string(device_id) +
           " to auto-complete config for " + Name())
           .c_str()));

  cuerr = cudaSetDevice(device_id);
  if (cuerr != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to set CUDA device to GPU ") +
         std::to_string(device_id) + " : " + cudaGetErrorString(cuerr))
            .c_str());
  }

  std::string artifact_name;
  RETURN_IF_ERROR(
      ModelConfig().MemberAsString("default_model_filename", &artifact_name));

  cudaDeviceProp cuprops;
  cuerr = cudaGetDeviceProperties(&cuprops, device_id);
  if (cuerr != cudaSuccess) {
    throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to get CUDA device properties for ") + name_ +
         ": " + cudaGetErrorString(cuerr))
            .c_str()));
  }

  const std::string cc =
      std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);

  common::TritonJson::Value cc_names;
  common::TritonJson::Value cc_name;
  if ((ModelConfig().Find("cc_model_filenames", &cc_names)) &&
      (cc_names.Find(cc.c_str(), &cc_name))) {
    RETURN_IF_ERROR(cc_name.AsString(&artifact_name));
  }

  // If the model configuration doesn't have an explicit model file specified
  // then use the default name ("model.plan").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.plan";
  } else {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string(
             "Using explicit serialized file '" + cc_model_filename +
             "' to auto-complete config for " + Name())
             .c_str()));
  }

  std::string model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  RETURN_IF_ERROR(AutoCompleteConfigHelper(model_path));

  cuerr = cudaSetDevice(current_device);
  if (cuerr != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to revert CUDA device to GPU ") +
         std::to_string(current_device) + " : " + cudaGetErrorString(cuerr))
            .c_str());
  }

  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    triton::common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("post auto-complete:\n") + buffer.Contents()).c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfigHelper(const std::string& model_path)
{
  std::shared_ptr<nvinfer1::IRuntime> runtime;
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
  if (LoadPlan(
          model_path, -1 /* dla_core_id */, &runtime, &engine,
          &tensorrt_logger_) != nullptr) {
    if (engine.get() != nullptr) {
      engine.reset();
    }
    if (runtime.get() != nullptr) {
      runtime.reset();
    }
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string(
             "unable to load plan file to auto complete config: " + model_path)
             .c_str()));
  }

  size_t input_cnt = 0;
  size_t output_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (ModelConfig().Find("input", &inputs)) {
      input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (ModelConfig().Find("batch_input", &config_batch_inputs)) {
      input_cnt += config_batch_inputs.ArraySize();
    }

    triton::common::TritonJson::Value outputs;
    if (ModelConfig().Find("output", &outputs)) {
      output_cnt = outputs.ArraySize();
    }
  }

  int num_profile_bindings = 0;
  int num_profiles = 0;
  num_profiles = engine->getNbOptimizationProfiles();
  num_profile_bindings = engine->getNbBindings() / num_profiles;

  // For batching support, the number of dimensions specified in model config
  // should be 1 less than the number of dimensions present in engine.
  // Will use that as a hint to ascertain whether or not to enable batching.
  // However, ragged batching is an exception to this rule. A tensor
  // allowing ragged batch is in itself a batching hint.
  bool config_batch_hint = false;

  // The number of IO Tensors with shape specification in config
  int tensors_with_config_shape_cnt = 0;

  if ((input_cnt != 0) || (output_cnt != 0)) {
    std::vector<std::string> io_types{"input", "output"};
    std::map<std::string, std::set<std::string>> allowed_tensors;
    for (int i = 0; i < num_profile_bindings; ++i) {
      if (engine->bindingIsInput(i)) {
        allowed_tensors["input"].emplace(engine->getBindingName(i));
      } else {
        allowed_tensors["output"].emplace(engine->getBindingName(i));
      }
    }

    bool io_allow_ragged_batch = false;
    for (const auto& io_type : io_types) {
      triton::common::TritonJson::Value config_io;
      RETURN_IF_ERROR(ModelConfig().MemberAsArray(io_type.c_str(), &config_io));
      for (size_t i = 0;
           ((i < config_io.ArraySize()) && (!io_allow_ragged_batch)); i++) {
        triton::common::TritonJson::Value io;
        RETURN_IF_ERROR(config_io.IndexAsObject(i, &io));
        triton::common::TritonJson::Value allow_ragged_batch;
        if (io.Find("allow_ragged_batch", &allow_ragged_batch)) {
          RETURN_IF_ERROR(allow_ragged_batch.AsBool(&io_allow_ragged_batch));
        }
        if (io_allow_ragged_batch) {
          // Treat the presence of tensor allowing ragged batch as
          // a hint for batching.
          config_batch_hint = true;
        } else {
          common::TritonJson::Value model_config_dims;
          common::TritonJson::Value reshape;
          if (io.Find("reshape", &reshape)) {
            RETURN_IF_ERROR(reshape.MemberAsArray("shape", &model_config_dims));
          } else {
            RETURN_IF_ERROR(io.MemberAsArray("dims", &model_config_dims));
          }
          if (model_config_dims.ArraySize() != 0) {
            tensors_with_config_shape_cnt++;
          }
          std::string name;
          RETURN_IF_ERROR(io.MemberAsString("name", &name));
          if (io_type.compare("input") == 0) {
            RETURN_IF_ERROR(
                CheckAllowedModelInput(io, allowed_tensors[io_type]));
          } else {
            RETURN_IF_ERROR(
                CheckAllowedModelOutput(io, allowed_tensors[io_type]));
          }
          if (model_config_dims.ArraySize() != 0) {
            RETURN_IF_ERROR(ExtractBatchHintFromIOConfig(
                engine.get(), name, model_config_dims, &config_batch_hint));
          }
        }
      }
    }
  }

  int max_batch_size = 0;
  bool has_implicit_batch_dim = false;
  if (engine->hasImplicitBatchDimension()) {
    // If engine has implicit batch dimension then retrieve the value and exit
    max_batch_size = engine->getMaxBatchSize();
    has_implicit_batch_dim = (max_batch_size != 1) || (MaxBatchSize() != 0);
  } else {
    // Assuming the first dimension to be batch dimension, until and unless
    // proven the batching is not supported.
    RETURN_IF_ERROR(
        GetMaxSupportedBatchSize(engine.get(), num_profiles, &max_batch_size));
  }

  if (config_batch_hint && max_batch_size == 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("autofill failed for model '") + Name() +
         "': model tensor shape configuration hints for dynamic batching "
         "but the underlying engine doesn't support batching.")
            .c_str());
  } else if (
      (tensors_with_config_shape_cnt != 0) && (!config_batch_hint) &&
      (!has_implicit_batch_dim)) {
    // if an explicit hint for non batching in config io
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("The specified dimensions in model config for ") + Name() +
         " hints that batching is unavailable")
            .c_str());
    max_batch_size = 0;
  }

  if (MaxBatchSize() == 0) {
    triton::common::TritonJson::Value mbs_value;
    ModelConfig().Find("max_batch_size", &mbs_value);
    mbs_value.SetInt(max_batch_size);
    SetMaxBatchSize(max_batch_size);
  } else if (MaxBatchSize() > max_batch_size) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("autofill failed for model '") + Name() +
         "': configuration specified max-batch " +
         std::to_string(MaxBatchSize()) +
         " but TensorRT engine only supports max-batch " +
         std::to_string(max_batch_size))
            .c_str());
  }

  // Turn on dynamic batch scheduler if batch size is greater
  // than 1 and there is no scheduler defined in the configuration.
  if (max_batch_size > 1) {
    triton::common::TritonJson::Value value;
    bool found_sequence_batching =
        ModelConfig().Find("sequence_batching", &value);
    bool found_dynamic_batching =
        ModelConfig().Find("dynamic_batching", &value);
    if (!found_sequence_batching && !found_dynamic_batching) {
      triton::common::TritonJson::Value dynamic_batching(
          ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      RETURN_IF_ERROR(
          ModelConfig().Add("dynamic_batching", std::move(dynamic_batching)));
    }
  }

  triton::common::TritonJson::Value ref_inputs(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  RETURN_IF_ERROR(GetRefIO(true /*is_input*/, engine.get(), &ref_inputs));
  triton::common::TritonJson::Value mutable_inputs(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  bool found_inputs = ModelConfig().Find("input", &mutable_inputs);
  RETURN_IF_ERROR(FixIO(engine.get(), ref_inputs, &mutable_inputs));
  if (!found_inputs) {
    RETURN_IF_ERROR(ModelConfig().Add("input", std::move(mutable_inputs)));
  }

  triton::common::TritonJson::Value ref_outputs(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  RETURN_IF_ERROR(GetRefIO(false /*is_input*/, engine.get(), &ref_outputs));
  triton::common::TritonJson::Value mutable_outputs(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  bool found_outputs = ModelConfig().Find("output", &mutable_outputs);
  RETURN_IF_ERROR(FixIO(engine.get(), ref_outputs, &mutable_outputs));
  if (!found_outputs) {
    RETURN_IF_ERROR(ModelConfig().Add("output", std::move(mutable_outputs)));
  }

  if (engine != nullptr) {
    engine.reset();
  }
  if (runtime != nullptr) {
    runtime.reset();
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::GetMaxSupportedBatchSize(
    nvinfer1::ICudaEngine* engine, const int num_profiles, int* max_batch_size)
{
  std::set<int> profile_indices;
  RETURN_IF_ERROR(GetProfileIndices(num_profiles, &profile_indices));

  int running_max = 0;
  for (const auto profile_index : profile_indices) {
    int max_profile_batch_size;
    RETURN_IF_ERROR(
        GetProfileMaxBatchSize(engine, profile_index, &max_profile_batch_size));
    if (max_profile_batch_size > running_max) {
      running_max = max_profile_batch_size;
    }
  }

  *max_batch_size = running_max;

  return nullptr;
}

TRITONSERVER_Error*
ModelState::GetProfileIndices(
    const int num_profiles, std::set<int>* profile_indices)
{
  common::TritonJson::Value groups;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("instance_group", &groups));
  for (size_t i = 0; i < groups.ArraySize(); i++) {
    common::TritonJson::Value group;
    RETURN_IF_ERROR(groups.IndexAsObject(i, &group));
    common::TritonJson::Value profiles;
    RETURN_IF_ERROR(group.MemberAsArray("profile", &profiles));
    for (size_t j = 0; j < profiles.ArraySize(); j++) {
      std::string profile;
      RETURN_IF_ERROR(profiles.IndexAsString(j, &profile));
      int profile_idx;
      RETURN_IF_ERROR(GetProfileIndex(profile, &profile_idx));
      if (profile_idx < 0 || profile_idx >= num_profiles) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to autofill for '") + Name() +
             "', configuration specified invalid profile " + profile +
             " . Number of profiles supported by TensorRT engine: " +
             std::to_string(num_profiles))
                .c_str());
      }
      profile_indices->insert(profile_idx);
    }
  }

  if (profile_indices->empty()) {
    // If not specified then use the default.
    profile_indices->insert(0);
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::GetProfileMaxBatchSize(
    nvinfer1::ICudaEngine* engine, int profile_index, int* max_batch_size)
{
  *max_batch_size = INT_MAX;

  int num_profiles = engine->getNbOptimizationProfiles();
  int num_profile_bindings = engine->getNbBindings() / num_profiles;

  // Visit all the bindings of the profile to capture the maximum and
  // minimum batch size supported.
  for (int binding_index = 0; binding_index < num_profile_bindings;
       binding_index++) {
    int effective_binding_index =
        (profile_index * num_profile_bindings) + binding_index;
    if (engine->bindingIsInput(effective_binding_index)) {
      if (!engine->isShapeBinding(effective_binding_index)) {
        nvinfer1::Dims max_shape = engine->getProfileDimensions(
            effective_binding_index, profile_index,
            nvinfer1::OptProfileSelector::kMAX);
        if (*max_batch_size > max_shape.d[0]) {
          *max_batch_size = max_shape.d[0];
        }

      } else {
        const int32_t* max_shapes = engine->getProfileShapeValues(
            effective_binding_index, profile_index,
            nvinfer1::OptProfileSelector::kMAX);
        if (*max_batch_size > *max_shapes) {
          *max_batch_size = *max_shapes;
        }
      }
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::ExtractBatchHintFromIOConfig(
    nvinfer1::ICudaEngine* engine, const std::string& tensor_name,
    const common::TritonJson::Value& dims, bool* config_batch_hint)
{
  // look up corresponding io info from model
  int num_profiles = engine->getNbOptimizationProfiles();
  int num_profile_bindings = engine->getNbBindings() / num_profiles;

  for (int binding_index = 0; binding_index < num_profile_bindings;
       binding_index++) {
    if (tensor_name == engine->getBindingName(binding_index)) {
      nvinfer1::Dims shape = engine->getBindingDimensions(binding_index);
      bool should_batch;
      if (!engine->isShapeBinding(binding_index)) {
        should_batch = (shape.nbDims == ((int32_t)dims.ArraySize() + 1));
      } else {
        int64_t first_dim = 0;
        RETURN_IF_ERROR(dims.IndexAsInt(0, &first_dim));
        should_batch = (shape.d[0] == (first_dim + 1));
      }
      if (should_batch) {
        *config_batch_hint = true;
      }
      if (*config_batch_hint && (!should_batch)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to autofill for '") + Name() +
             "', model tensor configurations are contradicting " +
             "each other in terms of whether batching is supported")
                .c_str());
      }
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelState::InstanceHasKindGPU(bool* has_instance_kind_gpu)
{
  *has_instance_kind_gpu = false;
  triton::common::TritonJson::Value instance_groups(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  ModelConfig().Find("instance_group", &instance_groups);

  if (instance_groups.ArraySize() > 0) {
    // TensorRT backend does not support KIND_CPU at all
    // so only check the first instance group kind.
    triton::common::TritonJson::Value group;
    RETURN_IF_ERROR(instance_groups.IndexAsObject(0, &group));

    triton::common::TritonJson::Value kind;
    group.Find("kind", &kind);
    std::string kind_str;
    RETURN_IF_ERROR(kind.AsString(&kind_str));
    if (kind_str == "KIND_GPU") {
      *has_instance_kind_gpu = true;
    }
  }

  return nullptr;  // success
}


TRITONSERVER_Error*
ModelState::GetRefIO(
    const bool is_input, nvinfer1::ICudaEngine* engine,
    triton::common::TritonJson::Value* ref_io)
{
  int num_profiles = engine->getNbOptimizationProfiles();
  int num_profile_bindings = engine->getNbBindings() / num_profiles;

  for (int i = 0; i < num_profile_bindings; ++i) {
    nvinfer1::Dims dims = engine->getBindingDimensions(i);
    bool is_shape_binding = engine->isShapeBinding(i);
    if ((is_input && (!engine->bindingIsInput(i))) ||
        ((!is_input) && (engine->bindingIsInput(i)))) {
      continue;
    }
    triton::common::TritonJson::Value io(
        ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
    std::string input_name{engine->getBindingName(i)};
    RETURN_IF_ERROR(
        io.AddString("name", input_name.substr(0, input_name.find(" "))));
    RETURN_IF_ERROR(io.AddString(
        "data_type",
        ConvertTrtTypeToConfigDataType(engine->getBindingDataType(i))));
    RETURN_IF_ERROR(InitIODims(engine, dims, is_shape_binding, &io));
    RETURN_IF_ERROR(io.AddBool("is_shape_tensor", is_shape_binding));

    RETURN_IF_ERROR(ref_io->Append(std::move(io)));
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelState::InitIODims(
    nvinfer1::ICudaEngine* engine, nvinfer1::Dims& dims, bool is_shape_binding,
    triton::common::TritonJson::Value* io)
{
  bool skip_first =
      (MaxBatchSize() != 0) && (!engine->hasImplicitBatchDimension());
  triton::common::TritonJson::Value config_dims(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  if (!is_shape_binding) {
    for (int didx = (skip_first ? 1 : 0); didx < dims.nbDims; ++didx) {
      RETURN_IF_ERROR(config_dims.AppendInt(dims.d[didx]));
    }
    // If tensor dims are empty then must use a reshape for the
    // tensor, since 'dims' is not allowed to be empty.
    if (config_dims.ArraySize() == 0) {
      RETURN_IF_ERROR(config_dims.AppendInt(1));
      triton::common::TritonJson::Value reshape(
          ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      triton::common::TritonJson::Value reshape_dims(
          ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
      RETURN_IF_ERROR(reshape.Add("shape", std::move(reshape_dims)));
      RETURN_IF_ERROR(io->Add("reshape", std::move(reshape)));
    }
  } else {
    if (dims.nbDims != 0) {
      if (skip_first) {
        RETURN_IF_ERROR(config_dims.AppendInt(dims.d[0] - 1));
      } else {
        RETURN_IF_ERROR(config_dims.AppendInt(dims.d[0]));
      }
    }
  }
  RETURN_IF_ERROR(io->Add("dims", std::move(config_dims)));

  return nullptr;
}

TRITONSERVER_Error*
ModelState::FixIO(
    nvinfer1::ICudaEngine* engine,
    triton::common::TritonJson::Value& reference_ios,
    triton::common::TritonJson::Value* mutable_ios)
{
  if (mutable_ios->ArraySize() == 0) {
    RETURN_IF_ERROR(mutable_ios->Swap(reference_ios));
  } else {
    for (size_t i = 0; i < mutable_ios->ArraySize(); i++) {
      triton::common::TritonJson::Value mutable_io;
      RETURN_IF_ERROR(mutable_ios->IndexAsObject(i, &mutable_io));
      std::string io_name;
      RETURN_IF_ERROR(mutable_io.MemberAsString("name", &io_name));
      for (size_t j = 0; j < reference_ios.ArraySize(); j++) {
        triton::common::TritonJson::Value io_ref;
        RETURN_IF_ERROR(reference_ios.IndexAsObject(j, &io_ref));
        std::string ref_name;
        RETURN_IF_ERROR(io_ref.MemberAsString("name", &ref_name));
        if (io_name.compare(ref_name) == 0) {
          // only set type and shape if they are not set
          common::TritonJson::Value data_type;
          if (mutable_io.Find("data_type", &data_type)) {
            std::string dt_str;
            RETURN_IF_ERROR(data_type.AsString(&dt_str));
            if (dt_str.empty() || (dt_str.compare("TYPE_INVALID") == 0)) {
              std::string ref_data_type;
              RETURN_IF_ERROR(
                  io_ref.MemberAsString("data_type", &ref_data_type));
              RETURN_IF_ERROR(data_type.SetString(ref_data_type));
            }
          } else {
            std::string ref_data_type;
            RETURN_IF_ERROR(io_ref.MemberAsString("data_type", &ref_data_type));
            RETURN_IF_ERROR(mutable_io.AddString("data_type", ref_data_type));
          }

          common::TritonJson::Value dims;
          if (mutable_io.Find("dims", &dims)) {
            if (dims.ArraySize() == 0) {
              common::TritonJson::Value ref_dims;
              RETURN_IF_ERROR(io_ref.MemberAsArray("dims", &ref_dims));
              RETURN_IF_ERROR(dims.Swap(ref_dims));
              common::TritonJson::Value reshape;
              if (io_ref.Find("reshape", &reshape)) {
                RETURN_IF_ERROR(mutable_io.Add("reshape", std::move(reshape)));
              }
            }
          } else {
            common::TritonJson::Value ref_dims;
            RETURN_IF_ERROR(io_ref.MemberAsArray("dims", &ref_dims));
            RETURN_IF_ERROR(mutable_io.Add("dims", std::move(ref_dims)));
            common::TritonJson::Value reshape;
            if (io_ref.Find("reshape", &reshape)) {
              RETURN_IF_ERROR(mutable_io.Add("reshape", std::move(reshape)));
            }
          }

          // Check if the IO is a shape tensor.
          bool is_shape_tensor = false;
          int io_index = engine->getBindingIndex(io_name.c_str());
          if (io_index == -1) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("binding for '") + io_name +
                 "' not found in the model.")
                    .c_str());
          }
          is_shape_tensor = engine->isShapeBinding(io_index);

          common::TritonJson::Value shape_tensor;
          if (mutable_io.Find("is_shape_tensor", &shape_tensor)) {
            bool shape_tensor_val = false;
            RETURN_IF_ERROR(shape_tensor.AsBool(&shape_tensor_val));
            if (shape_tensor_val && (!is_shape_tensor)) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("'") + io_name +
                   "' is incorrectly specified as a shape tensor.")
                      .c_str());
            } else if (!shape_tensor_val && is_shape_tensor) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("'") + io_name +
                   "' is incorrectly specified as an execution tensor.")
                      .c_str());
            }
          } else {
            RETURN_IF_ERROR(
                mutable_io.AddBool("is_shape_tensor", is_shape_tensor));
          }
          break;
        }
      }
    }
  }

  return nullptr;
}

}}}  // namespace triton::backend::tensorrt
