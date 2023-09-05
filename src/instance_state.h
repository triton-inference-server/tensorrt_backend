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

#include <future>
#include <map>
#include <unordered_map>

#include "io_binding_info.h"
#include "model_state.h"
#include "semaphore.h"
#include "tensorrt_model_instance.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"

namespace triton { namespace backend { namespace tensorrt {

// Number of CUDA event set for each instance.
// We need three sets to avoid event overlaps between issue
// and response threads.
static constexpr int EVENT_SET_COUNT = 3;

//
// BackendConfiguration
//
// Struct to hold values specified via backend config
struct BackendConfiguration {
  static const BackendConfiguration& RetrieveFrom(
      TRITONBACKEND_Backend* backend)
  {
    void* state = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_BackendState(backend, &state));
    return *reinterpret_cast<BackendConfiguration*>(state);
  }

  static const BackendConfiguration& RetrieveFrom(TRITONBACKEND_Model* model)
  {
    TRITONBACKEND_Backend* backend = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelBackend(model, &backend));
    return RetrieveFrom(backend);
  }

  static const BackendConfiguration& RetrieveFrom(
      TRITONBACKEND_ModelInstance* instance)
  {
    TRITONBACKEND_Model* model = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelInstanceModel(instance, &model));
    return RetrieveFrom(model);
  }

  bool coalesce_request_input_{false};
  bool enable_memory_tracker_{false};
};

class ModelInstanceState;
// A struct to hold TensorRT execution context and its meta data, a
// backend context can have multiple of this struct if multiple
// optimization profiles is specified.
struct TensorRTContext {
  TensorRTContext(
      const std::string& profile_name, const int profile_idx,
      const int binding_cnts, const int event_set_cnts)
      : profile_name_(profile_name), profile_idx_(profile_idx),
        cuda_graph_execs_(event_set_cnts), min_dims_(binding_cnts),
        max_dims_(binding_cnts), opt_dims_(binding_cnts),
        min_shapes_(binding_cnts), max_shapes_(binding_cnts),
        opt_shapes_(binding_cnts), is_dynamic_per_binding_(binding_cnts)
  {
  }
  std::string profile_name_;
  int profile_idx_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_{nullptr};

  // Struct that holds cudaGraphExec_t and the dimensions of the
  // inputs used to capture the graph
  struct CudaGraph {
    CudaGraph() : cuda_graph_exec_(nullptr) {}
    std::vector<int64_t> lower_bound_key_;
    // Store in the order of the bindng index
    std::vector<std::vector<int64_t>> input_dims_;
    cudaGraphExec_t cuda_graph_exec_;
  };

  // The key is packed input dimensions prepended by batch size, so
  // that uniqueness is guaranteed and the CUDA graphs are sorted to
  // provide convenience to find the closest CUDA graph in the
  // future.
  //
  // vector is used to map index of event sets to corresponding
  // collection.
  // [DLIS-4283] need to be careful in the case of having multiple sets of
  // binding buffer, see BuildCudaGraph that in such a case
  // (num_copy_streams_ != 1), the index of binding set is tied to the same
  // value for index of event set. This is fine for now as there is at most
  // 2 sets of bindings (the same as number of event sets), but wrong CUDA graph
  // may be used if we change to mismatching number such that the assumption
  // fails.
  std::vector<std::map<std::vector<int64_t>, CudaGraph>> cuda_graph_execs_;

  // Min Dimensions per bindings
  std::vector<nvinfer1::Dims> min_dims_{};

  // Max Dimensions per bindings
  std::vector<nvinfer1::Dims> max_dims_{};

  // Optimized Dimensions per bindings
  std::vector<nvinfer1::Dims> opt_dims_{};

  // Min shape values per bindings
  std::vector<const int32_t*> min_shapes_{};

  // Max shape values per bindings
  std::vector<const int32_t*> max_shapes_{};

  // Optimized shape values per bindings
  std::vector<const int32_t*> opt_shapes_{};

  // The number of shape values
  size_t nb_shape_values_{0};

  // Whether or not the binding contains a dynamic shape
  std::vector<bool> is_dynamic_per_binding_{};
};

struct GraphSpec {
  int64_t batch_size_{0};
  std::map<std::string, std::vector<int64_t>> shapes_{};
  int64_t lower_bound_batch_size_{0};
  std::map<std::string, std::vector<int64_t>> lower_bound_shapes_{};
  bool captured_{false};
};

// [DLIS-4283] temporary workaround to separate TRT v1 and TRT v3 usage
// in polymorphic style
class TRTInterface {
 public:
  TRTInterface(ModelInstanceState* i) : instance_(i) {}

  // NOTE: before calling this function, members of 'instance_' must be properly
  // set for all variants of IExecutionContext::enqueue(), i.e. input shapes and
  // I/O bindings
  virtual bool Enqueue(nvinfer1::IExecutionContext* context) = 0;

  // This function will be called to specify the runtime shape of the input and
  // adding metadata into existing 'cuda_graph_key' for graph lookup.
  virtual TRITONSERVER_Error* SetBindingDimensions(
      const std::string& input_name, const std::vector<int64_t>& shape,
      const TensorRTContext& trt_context, const size_t io_index,
      const size_t binding_index, std::vector<int64_t>* cuda_graph_key) = 0;

  // Return the max byte size of the binding
  virtual int64_t GetFullByteSize(
      nvinfer1::IExecutionContext* context, const std::string& tensor_name,
      int32_t binding_index) = 0;

  virtual TRITONSERVER_Error* SetFormat(
      int binding_index, TensorFormat* format) = 0;

  // get max input shape of the binding buffer based on the given
  // TensorRTContext (optimization profile),member of 'context' may be updated
  virtual TRITONSERVER_Error* ConfigureInputDimensions(
      TensorRTContext* context, int io_index, int binding_index,
      std::vector<int64_t> full_config_dims,
      std::vector<int64_t>* maximum_dims) = 0;

 protected:
  ModelInstanceState* instance_;

#ifdef TRITON_ENABLE_CUDA_GRAPH
 public:
  virtual bool BuildCudaGraph(
      TensorRTContext* trt_context, const GraphSpec& graph_spec) = 0;
#endif  // TRITON_ENABLE_CUDA_GRAPH
};

class TRTv1Interface : public TRTInterface {
 public:
  TRTv1Interface(ModelInstanceState* i) : TRTInterface(i) {}
  bool Enqueue(nvinfer1::IExecutionContext* context) override;
  TRITONSERVER_Error* SetBindingDimensions(
      const std::string& input_name, const std::vector<int64_t>& shape,
      const TensorRTContext& trt_context, const size_t io_index,
      const size_t binding_index,
      std::vector<int64_t>* cuda_graph_key) override;
  int64_t GetFullByteSize(
      nvinfer1::IExecutionContext* context, const std::string& tensor_name,
      int32_t binding_index) override;
  TRITONSERVER_Error* SetFormat(
      int binding_index, TensorFormat* format) override;
  TRITONSERVER_Error* ConfigureInputDimensions(
      TensorRTContext* context, int io_index, int binding_index,
      std::vector<int64_t> full_config_dims,
      std::vector<int64_t>* maximum_dims) override;
#ifdef TRITON_ENABLE_CUDA_GRAPH
 public:
  bool BuildCudaGraph(
      TensorRTContext* trt_context, const GraphSpec& graph_spec) override;
#endif  // TRITON_ENABLE_CUDA_GRAPH
};

class TRTv3Interface : public TRTInterface {
 public:
  TRTv3Interface(ModelInstanceState* i) : TRTInterface(i) {}
  bool Enqueue(nvinfer1::IExecutionContext* context) override;
  TRITONSERVER_Error* SetBindingDimensions(
      const std::string& input_name, const std::vector<int64_t>& shape,
      const TensorRTContext& trt_context, const size_t io_index,
      const size_t binding_index,
      std::vector<int64_t>* cuda_graph_key) override;
  int64_t GetFullByteSize(
      nvinfer1::IExecutionContext* context, const std::string& tensor_name,
      int32_t binding_index) override;
  TRITONSERVER_Error* SetFormat(
      int binding_index, TensorFormat* format) override;
  TRITONSERVER_Error* ConfigureInputDimensions(
      TensorRTContext* context, int io_index, int binding_index,
      std::vector<int64_t> full_config_dims,
      std::vector<int64_t>* maximum_dims) override;

 private:
  // Helper function to be called in Enqueue(). In v3, the binding buffer is set
  // in execution context instead of being provided on enqueue, so in the case
  // where different buffers are used alternatively in execution, we need to set
  // tensor address to proper buffers.
  bool SetTensorAddress(nvinfer1::IExecutionContext* context);
  TRITONSERVER_Error* MaximumDims(
      const nvinfer1::Dims& max_profile_dims, const std::vector<int64_t>& dims,
      std::vector<int64_t>* max_dims);
#ifdef TRITON_ENABLE_CUDA_GRAPH
 public:
  bool BuildCudaGraph(
      TensorRTContext* trt_context, const GraphSpec& graph_spec) override;

 private:
  TRITONSERVER_Error* SetCudaGraphShape(
      TensorRTContext* trt_context, const GraphSpec& graph_spec,
      std::vector<int64_t>* cuda_graph_key,
      TensorRTContext::CudaGraph* cuda_graph);
#endif  // TRITON_ENABLE_CUDA_GRAPH
};

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public TensorRTModelInstance {
 public:
  // GPU device number that indicates that no gpu is available for a
  // context (which is an invalid state since TensorRT requires a
  // GPU).
  static constexpr int NO_GPU_DEVICE = -1;

  // GPU device number that indicates model will be loaded on GPUs
  // as specified in model graph
  static constexpr int MODEL_DEVICE = -2;

  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  std::shared_ptr<nvinfer1::ICudaEngine>* EnginePtr() { return &engine_; }
  std::shared_ptr<nvinfer1::ICudaEngine> Engine() { return engine_; }

  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  void Run(TRITONBACKEND_Request** requests, const uint32_t request_count);

  Semaphore* SemaphorePtr() { return semaphore_.get(); }

 protected:
  friend class TRTv1Interface;
  friend class TRTv3Interface;

  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  void InitSemaphore();
  TRITONSERVER_Error* InitStreamsAndEvents();
  TRITONSERVER_Error* InitEventSet(bool busy_wait_events);
  TRITONSERVER_Error* DestroyEventSet();
  TRITONSERVER_Error* InitOptimizationProfiles();

  TRITONSERVER_Error* ValidateIO();
  TRITONSERVER_Error* ValidateIOHelper(
      common::TritonJson::Value& ios,
      const std::set<std::string>& allowed_shape_tensors, const bool is_input);

  TRITONSERVER_Error* InitIOBindingBuffers();
  TRITONSERVER_Error* InitializeConfigShapeInputBindings(
      common::TritonJson::Value& config_inputs);
  TRITONSERVER_Error* InitializeConfigExecuteInputBindings(
      common::TritonJson::Value& config_inputs);
  TRITONSERVER_Error* InitializeSequenceControlInputBindings(
      common::TritonJson::Value& config);
  TRITONSERVER_Error* InitializeSequenceStateInputBindings(
      common::TritonJson::Value& config);
  TRITONSERVER_Error* InitializeBatchInputBindings(
      common::TritonJson::Value& config);
  TRITONSERVER_Error* InitializeBatchOutputBindings(
      common::TritonJson::Value& config);
  // This function sets the output bindings for shape tensors.
  TRITONSERVER_Error* InitializeConfigShapeOutputBindings(
      common::TritonJson::Value& config_output);
  TRITONSERVER_Error* InitializeConfigExecuteOutputBindings(
      common::TritonJson::Value& config_output);
  TRITONSERVER_Error* InitializeExecuteInputBinding(
      const std::string& input_name, const std::string& input_datatype,
      common::TritonJson::Value& input_dims, const bool is_control = false,
      const bool is_ragged = false, const bool is_state = false);
  TRITONSERVER_Error* InitializeExecuteOutputBinding(
      const std::string& output_name, const std::string& output_datatype,
      common::TritonJson::Value& output_dims, bool is_state = false);
  TRITONSERVER_Error* InitializeShapeInputBinding(
      const std::string& input_name, const TRITONSERVER_DataType input_datatype,
      common::TritonJson::Value& model_config_dims);
  TRITONSERVER_Error* InitializeSequenceStateOutputBindings(
      common::TritonJson::Value& config);

  TRITONSERVER_Error* GetProfileDimensions(
      const int io_index, const int profile_index, TensorRTContext* context);

  TRITONSERVER_Error* GetRequestShapeValues(
      size_t total_batch_size, TRITONBACKEND_Request* request,
      std::map<int, std::vector<int32_t>>* request_shape_values);
  TRITONSERVER_Error* GetMostOptimizedProfile(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      uint32_t request_count,
      const std::map<int, std::vector<int32_t>>& request_shape_values,
      std::map<int, TensorRTContext>::iterator* citr);
  TRITONSERVER_Error* EvaluateTensorRTContext(
      std::map<int, TensorRTContext>::iterator& citr, size_t total_batch_size,
      TRITONBACKEND_Request** requests, uint32_t request_count,
      const std::map<int, std::vector<int32_t>>& request_shape_values,
      int64_t* error_distance);

  bool SetOutputShapeTensorBuffer(
      const int32_t* content, TRITONBACKEND_Response** response,
      TRITONBACKEND_Output* response_output, const size_t tensor_element_count,
      const int64_t batch_size, cudaStream_t stream);
  void ProcessResponse();

  void GetConfiguredProfiles(std::string* profiles_desc);
  int CudaStreamPriority() { return cuda_stream_priority_; }

  void FindClosestCudaGraph(
      const TensorRTContext& trt_context,
      const std::vector<int64_t>& cuda_graph_key,
      const TensorRTContext::CudaGraph** cuda_graph, bool* found_exact);

#ifdef TRITON_ENABLE_CUDA_GRAPH
  TRITONSERVER_Error* InitializeCudaGraph();
  TRITONSERVER_Error* InitializeGraphSpecs(
      std::vector<GraphSpec>* graph_specs, bool* allow_inexact_match);
  TRITONSERVER_Error* ValidateGraphSpec(const GraphSpec& graph_spec);
#endif  // TRITON_ENABLE_CUDA_GRAPH

  // The engine used for the instance. If the model uses dynamic
  // shape, then the CUDA engine is owned by the instance. Otherwise,
  // the engine is shared across all contexts and it must not be
  // destroyed by the instance. In the future version of TensorRT, the
  // engine may be shared even in the dynamic shape case.
  std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};

  // Map from profile index to the corresponding TensorRT context. Use
  // map to ensure each profile index is mapped to exactly one
  // TensorRT context.
  std::map<int, TensorRTContext> trt_contexts_{};

  // Is set true if the configuration supports batching
  bool support_batching_{false};

  // Whether inexact match is allowed for finding CUDA graph
  bool allow_inexact_match_{false};

  // The total number of bindings
  int total_bindings_{0};

  // The number of expected bindings to the model. In case of dynamic
  // shapes, it is the number of expected bindings to the configured
  // optimization profile.
  int num_expected_bindings_{0};

  int cuda_stream_priority_{0};

  // Additional CUDA streams to overlap copy and execution.
  cudaStream_t input_copy_stream_{};
  cudaStream_t output_copy_stream_{};
  int num_copy_streams_{0};

  // CUDA stream use to track execution status
  cudaStream_t signal_stream_{};

  // A group of CUDA events that signals different stages of the
  // request. One group should be used for one request at any given
  // moment.
  struct CUDAEventSet {
    // CUDA event to signal input buffer availability.
    cudaEvent_t ready_for_input_{};
    cudaEvent_t input_ready_{};

    // CUDA event for capturing correct timestamp.
    cudaEvent_t ready_for_output_{};
    cudaEvent_t output_ready_{};

    // CUDA event for synchronizing the order of timestamp capture.
    cudaEvent_t compute_output_start_{};
    cudaEvent_t compute_input_end_{};
    cudaEvent_t compute_input_start_{};
  };

  // Use two sets of events each for current request and next request.
  CUDAEventSet events_[EVENT_SET_COUNT]{};
  size_t next_set_{0};

  // Completion thread for handling items in the corresponding
  // completion queue. One thread per instance so that the thread
  // logic is simple as this avoids busy-looping on different model
  // executions' event states.
  std::thread completion_thread_{};

  // The details needed by the completion thread to finalize the
  // response for a model execution.
  struct Payload {
    explicit Payload(
        size_t event_set_idx, TRITONBACKEND_Request** requests,
        uint32_t request_count)
        : event_set_idx_(event_set_idx), requests_(requests),
          request_count_(request_count)
    {
    }

    // The index to the event set handling the request
    size_t event_set_idx_{0};

    // The total batch size for the request
    size_t total_batch_size_{0};

    // The timestamps for reporting stats
    uint64_t compute_start_ns_{0};
    uint64_t compute_input_end_ns_{0};
    uint64_t compute_output_start_ns_{0};

    // All the composing InferenceRequest objects
    std::vector<TRITONBACKEND_Request*> requests_list_{};
    TRITONBACKEND_Request** requests_{};
    uint32_t request_count_{0};

    // All the generated InferenceResponse objects
    std::vector<TRITONBACKEND_Response*> responses_{};

    // The State objects for the inference requests
    std::vector<TRITONBACKEND_State*> seq_states_{};

    // The collector and responder of the payload, need to extend
    // their lifetime to match the payload to ensure content is intact
    // until the end of execution.
    std::unique_ptr<BackendInputCollector> collector_{nullptr};
    std::unique_ptr<BackendOutputResponder> responder_{nullptr};

    std::vector<std::pair<void*, size_t>> buffer_input_binding_pairs_{};
  };

  // Assume that the lifetime of composing completion data to extend
  // till the responses are returned.
  triton::common::SyncQueue<std::unique_ptr<Payload>> completion_queue_{};

  // There will be two sets of input/output buffers when
  // separate_output_stream is selected to overlap copy and execution
  // safely.
  int next_buffer_binding_set_{0};

  // There are Context::num_expected_bindings_ number of IOBindingInfo
  // elements for copy stream.
  std::vector<std::vector<IOBindingInfo>> io_binding_infos_{};

  // [DLIS-4283] no longer needed for v3, but v1 still needs it. Should
  // encapsulate to v1 specific handling and gradually remove it from regular
  // workflow.
  // The pointer to the CUDA buffer for each binding index of the
  // TensorRT engine. This is used to match the TensorRT context
  // execution declaration while minimizing memory allocation. The
  // array size is equal to Context::total_bindings_ One of for each
  // copy stream
  std::vector<std::vector<void*>> buffer_bindings_{};

  // The request details of the ongoing model execution
  std::unique_ptr<Payload> payload_{nullptr};

  // Whether zero copy is supported on this device
  bool zero_copy_support_{false};

  // Whether the input collector will coalesce request inputs as if they form
  // one contiguous buffer when possible
  bool coalesce_request_input_{false};

  // Whether or not the model uses implicit state.
  bool uses_implicit_state_{false};

  // Holds up the execution on issue thread unless promise is fulfilled.
  std::unique_ptr<std::promise<void>> barrier_{nullptr};

  ModelState* model_state_{nullptr};

  std::unique_ptr<TRTInterface> interface_{nullptr};

  // TRT model instance performs execution asynchorously and thus may go
  // ahead to prepare further executions. Use semaphore to prevent going too
  // far ahead and overwriting resources that are still in use.
  std::unique_ptr<Semaphore> semaphore_{nullptr};
};

}}}  // namespace triton::backend::tensorrt
