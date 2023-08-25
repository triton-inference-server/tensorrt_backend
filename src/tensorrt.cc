// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "instance_state.h"
#include "logging.h"
#include "model_state.h"
#include "shared_library.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/device_memory_tracker.h"

//
// TensorRT Backend that implements the TRITONBACKEND API.
//
namespace triton { namespace backend { namespace tensorrt {

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

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
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  // Default execution policy, may be overridden by backend config
  auto execution_policy = TRITONBACKEND_EXECUTION_DEVICE_BLOCKING;

  std::unique_ptr<BackendConfiguration> lconfig(new BackendConfiguration());
  // Check if device memory tracker is explicitly enabled
  if (DeviceMemoryTracker::EnableFromBackendConfig(backend_config)) {
    lconfig->enable_memory_tracker_ = DeviceMemoryTracker::Init();
  }

  triton::common::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value value;
    std::string value_str;

    if (cmdline.Find("coalesce-request-input", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      RETURN_IF_ERROR(
          ParseBoolValue(value_str, &lconfig->coalesce_request_input_));
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
          LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err));
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
        ModelState::EnableVersionCompatibility();
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
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Registering TensorRT Plugins");
      success = initLibNvInferPlugins(&tensorrt_logger, "");
    });
  }
  if (!success) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Failed to register TensorRT Plugins");
  }

  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(lconfig.get())));

  lconfig.release();
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
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Utilizing DeviceMemoryTracker behavior that function calls with
  // 'nullptr' for usage will be no-ops.
  std::unique_ptr<DeviceMemoryTracker::MemoryUsage> lusage;
  if (BackendConfiguration::RetrieveFrom(model).enable_memory_tracker_) {
    lusage.reset(new DeviceMemoryTracker::MemoryUsage());
    DeviceMemoryTracker::TrackThreadMemoryUsage(lusage.get());
  }

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state = nullptr;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  if (lusage) {
    DeviceMemoryTracker::UntrackThreadMemoryUsage(lusage.get());
    TRITONSERVER_BufferAttributes** ba_array;
    uint32_t ba_len = 0;
    RETURN_IF_ERROR(lusage->SerializeToBufferAttributes(&ba_array, &ba_len));
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelReportMemoryUsage(model, ba_array, ba_len));
  }

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Utilizing DeviceMemoryTracker behavior that function calls with
  // 'nullptr' for usage will be no-ops.
  std::unique_ptr<DeviceMemoryTracker::MemoryUsage> lusage;
  if (BackendConfiguration::RetrieveFrom(instance).enable_memory_tracker_) {
    lusage.reset(new DeviceMemoryTracker::MemoryUsage());
    DeviceMemoryTracker::TrackThreadMemoryUsage(lusage.get());
  }


  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  if (lusage) {
    DeviceMemoryTracker::UntrackThreadMemoryUsage(lusage.get());
    TRITONSERVER_BufferAttributes** ba_array;
    uint32_t ba_len = 0;
    RETURN_IF_ERROR(lusage->SerializeToBufferAttributes(&ba_array, &ba_len));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceReportMemoryUsage(
        instance, ba_array, ba_len));
  }

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

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
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // For TensorRT backend, the executing instance may not closely tie to
  // TRITONBACKEND_ModelInstance, the instance will be assigned based on
  // execution policy.
  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  auto instance_semaphore =
      model_state->ExecutionState(device_id, instance_state);
  auto curr_instance = instance_semaphore.first;
  auto semaphore = instance_semaphore.second;

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       curr_instance->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

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

}  // extern "C"
}}}  // namespace triton::backend::tensorrt
