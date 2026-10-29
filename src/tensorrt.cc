// Copyright 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt.hh"

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>

#include "backend_configuration.hh"
#include "logging.h"
#include "model_connect/instance_state.hh"
#include "model_connect/model_state.hh"
#include "plan/instance_state.hh"
#include "plan/model_state.hh"
#include "shared_library.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/device_memory_tracker.h"
#include "triton_utils.hh"

//
// TensorRT Backend that implements the TRITONBACKEND API.
//
namespace triton::backend::tensorrt {

static std::unordered_map<void*, bool> model_is_trtmc_map{};
static std::mutex model_is_trtmc_map_mutex{};

/// @brief Gets the model configuration and determine if the model is a TRT
/// Model Connect model bundle based on the platform field in the model
/// configuration.
/// @param model The TRITONBACKEND_Model for which to get the configuration and
/// determine if it's a TRT-MC model.
/// @param model_name The name of the model, used for logging purposes.
/// @param platform The platform of the model, used to determine if it's a
/// TRT-MC model.
/// @returns A TRITONSERVER_Error indicating success or failure of the
/// operation.
TRITONSERVER_Error*
GetModelConfigPlatform(
    TRITONBACKEND_Model* model, const std::string model_name,
    std::string& platform)
{
  TRITONSERVER_Message* config_msg{nullptr};
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(model, 1, &config_msg));

  const char* json_buffer{nullptr};
  size_t json_buffer_size{0};
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      config_msg, &json_buffer, &json_buffer_size));

  common::TritonJson::Value model_config;
  auto err = model_config.Parse(json_buffer, json_buffer_size);
  TRITONSERVER_MessageDelete(config_msg);
  RETURN_IF_ERROR(err);

  platform.clear();

  const char* platform_cstr{nullptr};
  size_t platform_size{0};
  err = model_config.MemberAsString("platform", &platform_cstr, &platform_size);
  if (err) {
    TRITON_LOG_WARN(
        "Failed to get platform from model config for '"
        << model_name << "': " << TRITONSERVER_ErrorMessage(err) << ".");
    return err;
  }

  platform = std::string{platform_cstr, platform_size};
  TRITON_LOG_INFO(
      "Model platform for '" << model_name << "': '" << platform << "'.")

  return nullptr;  // success
}

/// @brief Gets model information including the model name, version, artifact
/// type, and whether it's a TRT-MC model.
/// @param model The TRITONBACKEND_Model for which to get the information.
/// @param model_name The name of the model, used for logging purposes.
/// @param model_version The version of the model.
/// @param artifact_type The artifact type of the model.
/// @param is_trtmc A boolean indicating whether the model is a TRT-MC model.
/// @returns A TRITONSERVER_Error indicating success or failure of the
/// operation.
TRITONSERVER_Error*
GetModelInfo(
    TRITONBACKEND_Model* model, std::string& model_name, uint64_t model_version,
    TRITONBACKEND_ArtifactType& artifact_type, bool& is_trtmc)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));

  model_name = std::string{cname};

  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &model_version));

  const char* location{nullptr};
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &location));

  try {
    std::lock_guard lock{model_is_trtmc_map_mutex};
    auto it = model_is_trtmc_map.find(static_cast<void*>(model));
    if (it != model_is_trtmc_map.end()) {
      is_trtmc = it->second;
    } else {
      std::string platform;
      if (auto err = GetModelConfigPlatform(model, model_name, platform)) {
        model_is_trtmc_map[static_cast<void*>(model)] = is_trtmc = false;
        return err;
      }

      is_trtmc = (platform == "model_connect");
      model_is_trtmc_map[static_cast<void*>(model)] = is_trtmc;
    }
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        TOSTRING(
            "Failed to inspect model \"" << model_name << "\""
                                         << ", version " << model_version
                                         << ": " << ex.what())
            .c_str());
  }

  return nullptr;  // success
}

/// @brief Gets the model instance information including the model, model name,
/// model version, whether it's a TRT-MC model, instance name, device id, and
/// instance group kind.
/// @param instance The TRITONBACKEND_ModelInstance for which to get the
/// information.
/// @param model The TRITONBACKEND_Model associated with the instance.
/// @param model_name The name of the model, used for logging purposes.
/// @param model_version The version of the model.
/// @param is_trtmc A boolean indicating whether the model is a TRT-MC model.
/// @param instance_name The name of the model instance, used for logging
/// purposes.
/// @param device_id The device id on which the model instance is running.
/// @param kind The instance group kind of the model instance.
/// @returns A TRITONSERVER_Error indicating success or failure of the
/// operation.
TRITONSERVER_Error*
GetModelInstanceInfo(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Model** model,
    std::string& model_name, uint64_t& model_version, bool& is_trtmc,
    std::string& instance_name, int32_t& device_id,
    TRITONSERVER_InstanceGroupKind& kind)
{
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, model));

  TRITONBACKEND_ArtifactType artifact_type{TRITONBACKEND_ARTIFACT_FILESYSTEM};
  RETURN_IF_ERROR(
      GetModelInfo(*model, model_name, model_version, artifact_type, is_trtmc));

  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  instance_name = std::string{cname};

  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name{cname};

  TRITON_LOG_INFO("TRITONBACKEND_Initialize: " << name);

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  TRITON_LOG_INFO(
      "Triton TRITONBACKEND API version: " << api_version_major << "."
                                           << api_version_minor);
  TRITON_LOG_INFO(
      "'" << name
          << "' TRITONBACKEND API version: " << TRITONBACKEND_API_VERSION_MAJOR
          << "." << TRITONBACKEND_API_VERSION_MINOR);

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  TRITON_LOG_INFO("backend configuration:\n" << buffer);

  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  // Default execution policy, may be overridden by backend config
  auto execution_policy = TRITONBACKEND_EXECUTION_DEVICE_BLOCKING;

  std::unique_ptr<BackendConfiguration> backendConfig(
      new BackendConfiguration{});
  // Check if device memory tracker is explicitly enabled
  if (DeviceMemoryTracker::EnableFromBackendConfig(backend_config)) {
    backendConfig->enable_memory_tracker_ = DeviceMemoryTracker::Init();
  }

  triton::common::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value value;
    std::string value_str;

    if (cmdline.Find("coalesce-request-input", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      RETURN_IF_ERROR(
          ParseBoolValue(value_str, &backendConfig->coalesce_request_input_));
    }

    if (cmdline.Find("plugins", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      size_t pos = 0;
      std::string plugin;
      // Load individual plugins
      while (value_str.length() > 0) {
        pos = value_str.find(";");
        plugin = value_str.substr(0, pos);
        void* handle = nullptr;
        auto err = OpenLibraryHandle(plugin, &handle);
        if (err != nullptr) {
          TRITON_LOG_ERROR(TRITONSERVER_ErrorMessage(err));
          TRITONSERVER_ErrorDelete(err);
          err = nullptr;
        }

        if (pos != std::string::npos) {
          pos++;
        }
        value_str.erase(0, pos);
      }
    }

    // Set the execution policy according to backend config, default will be
    // DEVICE_BLOCKING
    if (cmdline.Find("execution-policy", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      if (value_str == "DEVICE_BLOCKING") {
        execution_policy = TRITONBACKEND_EXECUTION_DEVICE_BLOCKING;
      } else if (value_str == "BLOCKING") {
        execution_policy = TRITONBACKEND_EXECUTION_BLOCKING;
      }
    }

    if (cmdline.Find("version-compatible", &value)) {
      bool is_version_compatible{false};
      RETURN_IF_ERROR(value.AsString(&value_str));
      RETURN_IF_ERROR(ParseBoolValue(value_str, &is_version_compatible));
      if (is_version_compatible) {
        plan::ModelState::EnableVersionCompatibility();
      }
    }
  }

  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetExecutionPolicy(backend, execution_policy));

  // Register all the default and custom plugins that come with TensorRT
  bool success = true;
  std::once_flag onceFlag;
  {
    std::call_once(onceFlag, [&success] {
      TensorRTLogger tensorrt_logger;
      TRITON_LOG_INFO("Registering TensorRT Plugins");
      success = initLibNvInferPlugins(&tensorrt_logger, "");
    });
  }

  if (!success) {
    TRITON_LOG_ERROR("Failed to register TensorRT Plugins");
  }

  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(backendConfig.get())));

  backendConfig.release();

  return nullptr;  // success
}


// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  if (BackendConfiguration::RetrieveFrom(backend).enable_memory_tracker_) {
    DeviceMemoryTracker::Fini();
  }

  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  delete reinterpret_cast<BackendConfiguration*>(vstate);
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  std::string model_name{};
  uint64_t model_version{0};
  TRITONBACKEND_ArtifactType artifact_type{TRITONBACKEND_ARTIFACT_FILESYSTEM};
  bool is_trtmc{false};

  // Get model info to determine if it is a TRT Model or TRT Model Container.
  RETURN_IF_ERROR(
      GetModelInfo(model, model_name, model_version, artifact_type, is_trtmc));

  if (is_trtmc) {
    model_connect::ModelState* model_state = nullptr;
    RETURN_IF_ERROR(model_connect::ModelState::Create(model, &model_state));
    TRITONSERVER_Error* set_state_err = TRITONBACKEND_ModelSetState(
        model, reinterpret_cast<void*>(model_state));
    if (set_state_err != nullptr) {
      delete model_state;
      return set_state_err;
    }
  } else {
    /* TensorRT Plan File (default / legacy) */

    // Utilizing DeviceMemoryTracker behavior that function calls with 'nullptr'
    // for usage will be no-ops.
    std::unique_ptr<DeviceMemoryTracker::MemoryUsage> devmem_tracker;
    if (BackendConfiguration::RetrieveFrom(model).enable_memory_tracker_) {
      devmem_tracker.reset(new DeviceMemoryTracker::MemoryUsage{});
      DeviceMemoryTracker::TrackThreadMemoryUsage(devmem_tracker.get());
    }

    // With each model we create a ModelState object and associate it
    // with the TRITONBACKEND_Model.
    plan::ModelState* model_state = nullptr;
    RETURN_IF_ERROR(plan::ModelState::Create(model, &model_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
        model, reinterpret_cast<void*>(model_state)));

    if (devmem_tracker) {
      DeviceMemoryTracker::UntrackThreadMemoryUsage(devmem_tracker.get());
      TRITONSERVER_BufferAttributes** ba_array;
      uint32_t ba_len = 0;
      RETURN_IF_ERROR(
          devmem_tracker->SerializeToBufferAttributes(&ba_array, &ba_len));
      RETURN_IF_ERROR(
          TRITONBACKEND_ModelReportMemoryUsage(model, ba_array, ba_len));
    }
  }

  TRITON_LOG_INFO(
      "TRITONBACKEND_ModelInitialize: " << model_name << " (version "
                                        << model_version << ")");

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  std::string model_name{};
  uint64_t model_version{0};
  TRITONBACKEND_ArtifactType artifact_type{TRITONBACKEND_ARTIFACT_FILESYSTEM};
  bool is_trtmc{false};

  RETURN_IF_ERROR(
      GetModelInfo(model, model_name, model_version, artifact_type, is_trtmc));

  void* vmodel{nullptr};
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodel));

  if (is_trtmc) {
    delete reinterpret_cast<model_connect::ModelState*>(vmodel);
  } else {
    /* TensorRT Plan File (default / legacy) */
    delete reinterpret_cast<plan::ModelState*>(vmodel);
  }

  {
    std::lock_guard lock{model_is_trtmc_map_mutex};
    model_is_trtmc_map.erase(static_cast<void*>(model));
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      TOSTRING(
          "TRITONBACKEND_ModelFinalize: model state for"
          " model \""
          << model_name << "\" version " << model_version << " deleted.")
          .c_str());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  TRITONBACKEND_Model* model;
  std::string model_name{};
  uint64_t model_version{0};
  bool is_trtmc{false};
  std::string instance_name{};
  int32_t device_id{0};
  TRITONSERVER_InstanceGroupKind kind;

  // Get model info to determine if it is a PT2 model archive or not.
  RETURN_IF_ERROR(GetModelInstanceInfo(
      instance, &model, model_name, model_version, is_trtmc, instance_name,
      device_id, kind));

  TRITON_LOG_INFO(
      "TRITONBACKEND_ModelInstanceInitialize: "
      << instance_name << " (" << TRITONSERVER_InstanceGroupKindString(kind)
      << " device " << std::to_string(device_id) << ")");

  void* vmodel;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodel));

  if (is_trtmc) {
    auto* model_state = reinterpret_cast<model_connect::ModelState*>(vmodel);
    if (model_state == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "model_connect model state is not initialized");
    }
    model_connect::ModelInstanceState* instance_state = nullptr;
    RETURN_IF_ERROR(model_connect::ModelInstanceState::Create(
        model_state, instance, &instance_state));
    TRITONSERVER_Error* set_state_err = TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void*>(instance_state));
    if (set_state_err != nullptr) {
      delete instance_state;
      return set_state_err;
    }
  } else {
    /* TensorRT Plan File (default / legacy) */

    plan::ModelState* model_state = reinterpret_cast<plan::ModelState*>(vmodel);

    // Utilizing DeviceMemoryTracker behavior that function calls with
    // 'nullptr' for usage will be no-ops.
    std::unique_ptr<DeviceMemoryTracker::MemoryUsage> devmem_tracker;
    if (BackendConfiguration::RetrieveFrom(instance).enable_memory_tracker_) {
      devmem_tracker.reset(new DeviceMemoryTracker::MemoryUsage());
      DeviceMemoryTracker::TrackThreadMemoryUsage(devmem_tracker.get());
    }

    // With each instance we create a ModelInstanceState object and
    // associate it with the TRITONBACKEND_ModelInstance.
    plan::ModelInstanceState* instance_state;
    RETURN_IF_ERROR(plan::ModelInstanceState::Create(
        model_state, instance, &instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void*>(instance_state)));

    if (devmem_tracker) {
      DeviceMemoryTracker::UntrackThreadMemoryUsage(devmem_tracker.get());
      TRITONSERVER_BufferAttributes** ba_array;
      uint32_t ba_len = 0;
      RETURN_IF_ERROR(
          devmem_tracker->SerializeToBufferAttributes(&ba_array, &ba_len));
      RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceReportMemoryUsage(
          instance, ba_array, ba_len));
    }
  }

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  TRITONBACKEND_Model* model;
  std::string model_name{};
  uint64_t model_version{0};
  bool is_trtmc{false};
  std::string instance_name{};
  int32_t device_id{0};
  TRITONSERVER_InstanceGroupKind kind;

  // Get model info to determine if it is a PT2 model archive or not.
  RETURN_IF_ERROR(GetModelInstanceInfo(
      instance, &model, model_name, model_version, is_trtmc, instance_name,
      device_id, kind));

  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));

  if (is_trtmc) {
    delete reinterpret_cast<model_connect::ModelInstanceState*>(vstate);
  } else {
    /* TensorRT Plan File (default / legacy) */
    delete reinterpret_cast<plan::ModelInstanceState*>(vstate);
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      TOSTRING(
          "TRITONBACKEND_ModelInstanceFinalize: model instance state for model "
          "\""
          << model_name << "\" version " << model_version << " deleted.")
          .c_str());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  TRITONBACKEND_Model* parent_model = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &parent_model));
  bool is_trtmc = false;
  {
    std::lock_guard lock{model_is_trtmc_map_mutex};
    const auto found =
        model_is_trtmc_map.find(static_cast<void*>(parent_model));
    if (found != model_is_trtmc_map.end()) {
      is_trtmc = found->second;
    }
  }
  if (is_trtmc) {
    model_connect::ModelInstanceState* instance_state = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
        instance, reinterpret_cast<void**>(&instance_state)));
    if (instance_state == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "model_connect instance is not initialized");
    }
    // Each request is one text_continuation call. Ownership of 'requests'
    // stays with ProcessRequests, which responds to every request.
    instance_state->ProcessRequests(requests, request_count);
    return nullptr;
  }

  plan::ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  plan::ModelState* model_state = instance_state->StateForModel();

  // For TensorRT backend, the executing instance may not closely tie to
  // TRITONBACKEND_ModelInstance, the instance will be assigned based on
  // execution policy.
  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  auto instance_semaphore =
      model_state->ExecutionState(device_id, instance_state);
  auto curr_instance = instance_semaphore.first;
  auto semaphore = instance_semaphore.second;

  TRITON_LOG_VERBOSE(
      "model " << model_state->Name() << ", instance " << curr_instance->Name()
               << ", executing " << request_count << " requests");

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  curr_instance->ProcessRequests(requests, request_count);

  // Returning from TRITONBACKEND_ModelInstanceExecute signals Triton to start
  // preparing next batch. Due to the async execution in TensorRT backend, we
  // need to block the function if there is no instance ready for preparing the
  // next execution.
  // Acquire() implies that the next execution can be initiated, and it is done
  // at the end of current execution because we want to form the batch as late
  // as possible, otherwise we may get a smaller batch while more requests may
  // arrive between when the batch is formed and when batch is executed.
  semaphore->Acquire();

  return nullptr;  // success
}

}  // namespace triton::backend::tensorrt
