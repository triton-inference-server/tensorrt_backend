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

#pragma once

#include <NvInfer.h>

#include "logging.h"
#include "semaphore.h"
#include "tensorrt_model.h"
#include "tensorrt_model_instance.h"

namespace triton { namespace backend { namespace tensorrt {

class ModelInstanceState;
// Helper class to determine the ModelInstanceState to be used for current
// execution and the semaphore to be blocked for next execution.
class ExecutionArbitrator {
 public:
  virtual void RegisterInstance(
      const int device_id, ModelInstanceState* instance) = 0;
  virtual std::pair<ModelInstanceState*, Semaphore*> ExecutionState(
      const int device_id, ModelInstanceState* instance) = 0;
  virtual ~ExecutionArbitrator(){};
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public TensorRTModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState();

  TRITONSERVER_Error* CreateEngine(
      int gpu_device, const int64_t dla_core_id, const std::string& model_path,
      std::shared_ptr<nvinfer1::ICudaEngine>* engine);

  void DisableEngineSharing() { engine_sharing_ = false; }
  bool IsEngineSharingEnabled() { return engine_sharing_; }

  static void EnableVersionCompatibility() { is_version_compatible_ = true; }
  static bool isVersionCompatible() { return is_version_compatible_; }

  // Register the instance and its associated device ID to execution arbitrator.
  void RegisterInstance(const int device_id, ModelInstanceState* instance)
  {
    execution_arbitrator_->RegisterInstance(device_id, instance);
  }

  // Query the execution arbitrator to return the instance for the execution and
  // the semaphore to check whether the next execution should be initiated.
  // 'device_id', 'instance' are the metadata associated with the
  // TRITONBACKEND_ModelInstance.
  std::pair<ModelInstanceState*, Semaphore*> ExecutionState(
      const int device_id, ModelInstanceState* instance)
  {
    return execution_arbitrator_->ExecutionState(device_id, instance);
  }

  TensorRTLogger& GetTensorRTLogger() { return tensorrt_logger_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Auto-complete the model configuration
  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteConfigHelper(const std::string& model_path);
  TRITONSERVER_Error* GetMaxSupportedBatchSize(
      nvinfer1::ICudaEngine* engine, const int num_profiles,
      int* max_batch_size);
  TRITONSERVER_Error* GetProfileIndices(
      const int num_profiles, std::set<int>* profile_indices);
  TRITONSERVER_Error* GetProfileMaxBatchSize(
      nvinfer1::ICudaEngine* engine, int profile_index, int* max_batch_size);
  TRITONSERVER_Error* ExtractBatchHintFromIOConfig(
      nvinfer1::ICudaEngine* engine, const std::string& tensor_name,
      const common::TritonJson::Value& dims, bool* config_batch_hint);
  TRITONSERVER_Error* InstanceHasKindGPU(bool* has_instance_kind_gpu);
  TRITONSERVER_Error* GetRefIO(
      const bool is_input, nvinfer1::ICudaEngine* engine,
      triton::common::TritonJson::Value* ref_io);
  TRITONSERVER_Error* InitIODims(
      nvinfer1::ICudaEngine* engine, nvinfer1::Dims& dims,
      bool is_shape_binding, triton::common::TritonJson::Value* io);
  TRITONSERVER_Error* FixIO(
      nvinfer1::ICudaEngine* engine,
      triton::common::TritonJson::Value& reference_ios,
      triton::common::TritonJson::Value* mutable_ios);


  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Parses the parameters in config
  TRITONSERVER_Error* ParseParameters();

  // TensorRT logger for this model
  TensorRTLogger tensorrt_logger_;

  // CUDA engine shared across all model instances using the same (or no) DLA
  // core on same GPU. The first element in the key pair is the GPU ID, the
  // second is the DLA core ID.
  std::map<
      std::pair<int, int64_t>, std::pair<
                                   std::shared_ptr<nvinfer1::IRuntime>,
                                   std::shared_ptr<nvinfer1::ICudaEngine>>>
      device_engines_;
  bool engine_sharing_;

  std::unique_ptr<ExecutionArbitrator> execution_arbitrator_;

  // Whether the backend should support version-compatible TensorRT models.
  static inline bool is_version_compatible_{false};
};


}}}  // namespace triton::backend::tensorrt
