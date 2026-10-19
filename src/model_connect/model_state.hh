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

#pragma once

#include "instance_state.hh"
#include "../logging.h"
#include "../triton_semaphore.h"
#include "triton/backend/backend_model.h"

#include <NvInfer.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace triton::backend::tensorrt::model_connect
{
  class ModelState
    : public triton::backend::BackendModel
  {
    public:

      enum class Priority
      {
        /// @brief Default value.
        unspecified = 0,
        /// @brief Minimum priority.
        minimum = 1,
        /// @brief Maximum priority.
        maximum = 2,
      };

    protected:

      nvinfer1::ExecutionContextAllocationStrategy alloc_strategy_;
      bool cuda_graphs_enabled_{false};
      std::map<std::pair<int, int64_t>,
               std::pair<std::shared_ptr<nvinfer1::IRuntime>,
                         std::shared_ptr<nvinfer1::ICudaEngine>>> device_engines_;
      bool busy_wait_events_enabled_{false};
      bool eager_batching_enabled_{false};
      bool engine_sharing_enabled_{false};
      size_t gather_kernel_buffer_threshold_{0};
      common::TritonJson::Value graph_specs_;
      TensorRTLogger logger_;
      bool multidevice_enabled_{false};
      std::vector<int> multidevice_ids_{};
      bool multidevice_per_rank_engines_{false};
      Priority priority_{Priority::unspecified};
      bool separate_output_stream_enabled_{false};

    public:

      ModelState(
          TRITONBACKEND_Model* triton_model);

      virtual ~ModelState();

      TRITONSERVER_Error*
      CreateEngine(
          int gpu_device,
          int64_t dla_core_id,
          const std::string& model_path,
          std::shared_ptr<nvinfer1::ICudaEngine>* engine);

      bool
      GetBusyWaitEventsEnabled() const;

      bool
      GetCudaGraphsEnabled() const;

      bool
      GetEagerBatchingEnabled() const;

      bool
      GetEngineSharingEnabled() const;

      std::pair<InstanceState*, Semaphore*>
      GetExecutionState(
          int device_id,
          InstanceState* instance);

      nvinfer1::ExecutionContextAllocationStrategy
      GetExecutionContextAllocationStrategy() const;

      size_t
      GetGatherKernelBufferThreshold() const;

      common::TritonJson::Value&
      GetGraphSpecs();

      int
      GetCudaStreamPriority() const;

      Priority
      GetPriority() const;

      bool
      GetMultideviceEnabled() const;

      const std::vector<int>&
      GetMultideviceIds() const;

      bool
      GetMultidevicePerRankEngines() const;

      bool
      GetSeparateOutputStreamEnabled() const;

      TensorRTLogger&
      GetTensorRTLogger() const;

      TRITONSERVER_Error*
      ParseModelConfig();

      void
      SetBusyWaitEventsEnabled(
          bool enabled);

      void
      SetCudaGraphsEnabled(
          bool enabled);

      void
      SetEagerBatchingEnabled(
          bool enabled);

      void
      SetEngineSharingEnabled(
          bool enabled);

      void
      SetMultidevice(
          bool enabled,
          const std::vector<int>& device_ids,
          bool per_rank_engines);
      void
      SetSeparateOutputStreamEnabled(
          bool enabled);

      TRITONSERVER_Error*
      SetTensorRTModelConfig();

    private:

      TRITONSERVER_Error*
      AutoCompleteConfig();

      TRITONSERVER_Error*
      AutoCompleteConfigHelper(
          const std::string& model_path);

      TRITONSERVER_Error*
      ExtractBatchHintFromIOConfig(
          nvinfer1::ICudaEngine* engine,
          const std::string& tensor_name,
          const common::TritonJson::Value& dims,
          bool* config_batch_hint);

      TRITONSERVER_Error* FixIO(
          nvinfer1::ICudaEngine* engine,
          triton::common::TritonJson::Value& reference_ios,
          triton::common::TritonJson::Value* mutable_ios);

      TRITONSERVER_Error*
      GetMaxSupportedBatchSize(
          nvinfer1::ICudaEngine* engine,
          int num_profiles,
          int* max_batch_size);

      TRITONSERVER_Error*
      GetProfileIndices(
          const int num_profiles,
          std::set<int>* profile_indices);

      TRITONSERVER_Error*
      GetProfileMaxBatchSize(
          nvinfer1::ICudaEngine* engine,
          int profile_index,
          int* max_batch_size);

      TRITONSERVER_Error*
      GetRefIO(
          bool is_input,
          nvinfer1::ICudaEngine* engine,
          triton::common::TritonJson::Value* ref_io);

      TRITONSERVER_Error* InitIODims(
          nvinfer1::ICudaEngine* engine,
          nvinfer1::Dims& dims,
          bool is_shape_tensor,
          triton::common::TritonJson::Value* io);

      TRITONSERVER_Error*
      InstanceHasKindGPU(
          bool* has_instance_kind_gpu);

      // Validate that model configuration is supported by this backend.
      TRITONSERVER_Error*
      ValidateModelConfig();

      // Parses the parameters in config
      TRITONSERVER_Error*
      ParseParameters();

      // Parses TensorRT Multi-Device (MD) parameters from the 'parameters' config
      // block (TRT-28040): 'enable_multi_device' and 'multi_device_gpus'.
      TRITONSERVER_Error*
      ParseMultiDeviceParameters(
          common::TritonJson::Value& params);
  };
}
