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

#include "model_state.hh"
#include "../io_binding_info.h"
#include "../shape_tensor.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"

#include <NvInfer.h>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace triton::backend::tensorrt::model_connect
{
  class InstanceState
    : public triton::backend::BackendModelInstance
  {
    public:

      static constexpr int EVENT_SET_COUNT = 3;

    protected:

      struct CudaEventSet
      {
        cudaEvent_t compute_input_start_{};
        cudaEvent_t compute_input_end_{};
        cudaEvent_t compute_output_start_{};
        cudaEvent_t compute_output_end_{};
        cudaEvent_t input_ready_{};
        cudaEvent_t output_ready_{};
        cudaEvent_t ready_for_input_{};
        cudaEvent_t ready_for_output_{};
      };

      struct Payload
      {
        // The collector and responder of the payload, need to extend their lifetime to match the payload to ensure content is intact until the end of execution.
        std::unique_ptr<BackendInputCollector> backend_input_collector_{nullptr};
        std::unique_ptr<BackendOutputResponder> backend_output_responder_{nullptr};
        size_t compute_input_end_ns_{0};
        size_t compute_input_start_ns_{0};
        size_t compute_output_end_ns_{0};
        size_t compute_output_start_ns_{0};
        size_t event_set_idx_{0};
        std::vector<std::pair<void*, size_t>> io_buffers_{};
        size_t total_batch_size_{0};
        // The active request and response objects.
        std::vector<TRITONBACKEND_Request*> requests_{};
        std::vector<TRITONBACKEND_Response*> responses_{};
        // The State objects for the inference requests
        std::vector<TRITONBACKEND_State*> sequence_states_{};

        explicit Payload(
            size_t event_set_idx,
            TRITONBACKEND_Request** requests,
            uint32_t request_count)
          : event_set_idx_{event_set_idx}
          , requests_{requests, requests + request_count}
        { }
      };

      // Whether inexact match is allowed for finding CUDA graph
      bool allow_inexact_match_{false};
      // Holds up the execution on issue thread unless promise is fulfilled.
      std::unique_ptr<std::promise<void>> barrier_{nullptr};
      // Whether the input collector will coalesce request inputs as if they form one contiguous buffer when possible.
      bool coalesce_request_input_{false};
      // Completion thread for handling items in the corresponding completion queue.
      // One thread per instance so that the thread logic is simple as this avoids busy-looping on different model executions' event states.
      std::thread completion_thread_{};
      // Use two sets of events each for current request and next request.
      CudaEventSet cuda_events_[EVENT_SET_COUNT];
      size_t cuda_events_next_set_{0};
      int32_t cuda_stream_priority_{0};
      // CUDA stream use to track execution status
      cudaStream_t cuda_signal_stream_{};
      // Additional CUDA streams to overlap copy and execution.
      cudaStream_t cuda_stream_input_copy_{};
      cudaStream_t cuda_stream_output_copy_{};
      int32_t cuda_streams_num_copy_{0};
      // The engine used for the instance.
      // If the model uses dynamic shape, then the CUDA engine is owned by the instance.
      // Otherwise, the engine is shared across all contexts and it must not be destroyed by the instance.
      // In the future version of TensorRT, the engine may be shared even in the dynamic shape case.
      std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine_;
      // Whether or not the model uses implicit state.
      bool implicit_state_{false};
      // There are Context::total_io_tensors_ number of `IOBindingInfo` elements for copy stream.
      std::vector<std::vector<IOBindingInfo>> io_binding_info_{};
      // Assume that the lifetime of composing completion data to extend till the responses are returned.
      triton::common::SyncQueue<std::unique_ptr<Payload>> io_completion_queue_{};
      // There will be two sets of input/output buffers when separate_output_stream is selected to overlap copy and execution safely.
      int32_t io_next_buffer_binding_set{0};
      ModelState* model_{nullptr};
      // TRT model instance performs execution asynchronously and thus may go ahead to prepare further executions.
      // Use semaphore to prevent going too far ahead and overwriting resources that are still in use.
      std::unique_ptr<Semaphore> semaphore_{nullptr};
      // Is set true if the configuration supports batching.
      bool support_batching_{false};
      // Whether zero copy is supported on this device
      bool support_zero_copy_{false};
      // Mapping from Input/Output Tensor Name to its corresponding Index
      std::map<std::string, int> tensor_index_map_{};
      std::map<int, std::string> tensor_names_{};
      // The number of input and output tensors for the network from which the engine was built.
      int32_t total_io_tensors_{0};

#ifndef TRITON_ENABLE_TRT_MULTI_DEVICE
      // --- TensorRT Multi-Device (MD) state — TRT-28040 ---
      // Enabled when the model config sets enable_multi_device=true (KIND_MODEL).
      bool multidevice_enabled_{false};
      // Physical GPU id per rank; size == world size.
      // `multidevice_device_ids_[0]` is rank `0`, which is also DeviceId() / the device the rest of this instance runs on.
      std::vector<int> multidevice_device_ids_{};
      int multidevice_world_size_{0};

      // NCCL communicators, one per rank (stored as void* to keep nccl.h out of this header; cast to ncclComm_t in the .cc).
      // Index == rank.
      std::vector<void*> multidevice_comms_{};

      // Per-rank resources for ranks 1..N-1 (index r-1).
      // Rank 0 reuses the instance's engine_/trt_contexts_/stream_/io_binding_infos_.
      std::vector<std::shared_ptr<nvinfer1::IRuntime>> multidevice_runtimes_{};
      std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> multidevice_engines_{};
      std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> multidevice_contexts_{};
      std::vector<cudaStream_t> multidevice_streams_{};
      // Per non-zero rank, per IO tensor name -> (device buffer, byte size).
      std::vector<std::map<std::string, std::pair<void*, size_t>>> multidevice_io_buffers_{};
      // Pinned host bounce buffer per non-zero rank, used to replicate rank-0 inputs when direct P2P copy is unavailable.
      // Sized to the largest IO tensor, allocated once and reused across requests.
      std::vector<void*> multidevice_stage_{};
      // Whether rank 0's GPU and rank r's GPU support a direct peer copy.
      // When `true` (e.g. NVLink or working PCIe P2P) inputs are replicated with `cudaMemcpyPeer`; otherwise via the pinned host bounce buffer.
      std::vector<char> multidevice_peer_{};
#endif

      InstanceState(
          ModelState* model,
          TRITONBACKEND_ModelInstance* triton_model_instance);

    public:

      virtual ~InstanceState();

      static TRITONSERVER_Error*
      Create(
          ModelState* model,
          TRITONBACKEND_ModelInstance* triton_model_instance);

      int
      CudaStreamPriority();

      TRITONSERVER_Error*
      DestroyEventSet();

      ModelState*
      GetModelState() const;

      void GetConfiguredProfiles(
          std::string& profiles_desc);

      TRITONSERVER_Error*
      InitStreamsAndEvents();

      TRITONSERVER_Error*
      InitEventSet(
          bool busy_wait_events);

      TRITONSERVER_Error*
      InitIOIndexMap();

      TRITONSERVER_Error*
      InitOptimizationProfiles();

      TRITONSERVER_Error*
      InitIOBindingBuffers();

      TRITONSERVER_Error*
      InitializeConfigShapeInputBindings(
          common::TritonJson::Value& config_inputs);

      TRITONSERVER_Error*
      InitializeConfigExecuteInputBindings(
          common::TritonJson::Value& config_inputs);

      TRITONSERVER_Error*
      InitializeSequenceControlInputBindings(
          common::TritonJson::Value& config);

      TRITONSERVER_Error*
      InitializeSequenceStateInputBindings(
          common::TritonJson::Value& config);

      TRITONSERVER_Error*
      InitializeBatchInputBindings(
          common::TritonJson::Value& config);

      TRITONSERVER_Error*
      InitializeBatchOutputBindings(
          common::TritonJson::Value& config);

      // This function sets the output bindings for shape tensors.
      TRITONSERVER_Error*
      InitializeConfigShapeOutputBindings(
          common::TritonJson::Value& config_output);

      TRITONSERVER_Error*
      InitializeConfigExecuteOutputBindings(
          common::TritonJson::Value& config_output);

      TRITONSERVER_Error*
      InitializeExecuteInputBinding(
          const std::string& input_name,
          const std::string& input_datatype,
          common::TritonJson::Value& input_dims,
          const bool is_control = false,
          const bool is_ragged = false,
          const bool is_state = false);

      TRITONSERVER_Error*
      InitializeExecuteOutputBinding(
          const std::string& output_name,
          const std::string& output_datatype,
          common::TritonJson::Value& output_dims,
          bool is_state = false);

      TRITONSERVER_Error*
      InitializeShapeInputBinding(
          const std::string& input_name,
          const TRITONSERVER_DataType input_datatype,
          common::TritonJson::Value& model_config_dims);

      TRITONSERVER_Error*
      InitializeSequenceStateOutputBindings(
          common::TritonJson::Value& config);

      void
      ProcessResponse();

      bool
      SetOutputShapeTensorBuffer(
          const int64_t* content,
          TRITONBACKEND_Response** response,
          TRITONBACKEND_Output* response_output,
          size_t tensor_element_count,
          int64_t batch_size,
          cudaStream_t stream);

      TRITONSERVER_Error*
      ValidateIO();

      TRITONSERVER_Error*
      ValidateIOHelper(
          common::TritonJson::Value& ios,
          const std::set<std::string>& allowed_shape_tensors,
          const std::set<std::string>& allowed_non_linear_format_io,
          const bool is_input);

  };
}
