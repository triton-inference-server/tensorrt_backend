// Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "instance_state.h"

#include "tensorrt_utils.h"
#include "triton/common/nvtx.h"

namespace triton { namespace backend { namespace tensorrt {

namespace {

TRITONSERVER_Error*
CreateCudaEvent(
    const std::string& event_name, unsigned int event_flags, cudaEvent_t* event)
{
  // Not adding 'cudaEventBlockingSync' to reduce gaps between the
  // time of event record and the time of signaling blocking thread.
  // The busy waiting only happens when there is an inflight request.
  auto cuerr = cudaEventCreateWithFlags(event, event_flags);
  if (cuerr != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to create CUDA event for ") + event_name + ": " +
         cudaGetErrorString(cuerr))
            .c_str());
  }
  return nullptr;
}

}  // namespace

#ifdef TRITON_ENABLE_STATS
#define FAIL_ALL_AND_RETURN_IF_ERROR(                                          \
    REQUESTS, REQUEST_COUNT, RESPONSES, S, LOG_MSG)                            \
  do {                                                                         \
    TRITONSERVER_Error* farie_err_ = (S);                                      \
    if (farie_err_ != nullptr) {                                               \
      for (uint32_t r = 0; r < REQUEST_COUNT; ++r) {                           \
        if (RESPONSES[r] != nullptr) {                                         \
          LOG_IF_ERROR(                                                        \
              TRITONBACKEND_ResponseSend(                                      \
                  RESPONSES[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,          \
                  farie_err_),                                                 \
              "failed to send TensorRT backend response");                     \
          LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (LOG_MSG));                      \
        }                                                                      \
        LOG_IF_ERROR(                                                          \
            TRITONBACKEND_ModelInstanceReportStatistics(                       \
                TritonModelInstance(), REQUESTS[r], false /* success */, 0, 0, \
                0, 0),                                                         \
            "failed reporting request statistics");                            \
        LOG_IF_ERROR(                                                          \
            TRITONBACKEND_RequestRelease(                                      \
                REQUESTS[r], TRITONSERVER_REQUEST_RELEASE_ALL),                \
            "failed releasing request");                                       \
        REQUESTS[r] = nullptr;                                                 \
      }                                                                        \
      TRITONSERVER_ErrorDelete(farie_err_);                                    \
      return;                                                                  \
    }                                                                          \
  } while (false)

void CUDART_CB
TimestampCaptureCallback(void* data)
{
  SET_TIMESTAMP(*(reinterpret_cast<uint64_t*>(data)));
}

#else
#define FAIL_ALL_AND_RETURN_IF_ERROR(                                 \
    REQUESTS, REQUEST_COUNT, RESPONSES, S, LOG_MSG)                   \
  do {                                                                \
    TRITONSERVER_Error* farie_err_ = (S);                             \
    if (farie_err_ != nullptr) {                                      \
      for (uint32_t r = 0; r < REQUEST_COUNT; ++r) {                  \
        if (RESPONSES[r] != nullptr) {                                \
          LOG_IF_ERROR(                                               \
              TRITONBACKEND_ResponseSend(                             \
                  RESPONSES[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                  farie_err_),                                        \
              "failed to send TensorRT backend response");            \
          LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (LOG_MSG));             \
        }                                                             \
        LOG_IF_ERROR(                                                 \
            TRITONBACKEND_RequestRelease(                             \
                REQUESTS[r], TRITONSERVER_REQUEST_RELEASE_ALL),       \
            "failed releasing request");                              \
        REQUESTS[r] = nullptr;                                        \
      }                                                               \
      TRITONSERVER_ErrorDelete(farie_err_);                           \
      return;                                                         \
    }                                                                 \
  } while (false)

#endif  // TRITON_ENABLE_STATS

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // If the model configuration doesn't have an explicit model file
  // specified then use the default name.
  std::string cc_model_filename = (*state)->ArtifactFilename();
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.plan";
  }

  auto model_path = JoinPath(
      {model_state->RepositoryPath(), std::to_string(model_state->Version()),
       cc_model_filename});

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + model_path +
            "' for model instance '" + (*state)->Name() + "'");
  }

  (*state)->InitSemaphore();
  RETURN_IF_ERROR((*state)->InitStreamsAndEvents());
  RETURN_IF_ERROR(model_state->CreateEngine(
      (*state)->DeviceId(), (*state)->DLACoreId(), model_path,
      (*state)->EnginePtr()));

  // Create TRT API interface once able to obtain implicit batch info,
  // all TRT operations must be done after the interface is instantiated.
  if (UseTensorRTv1API((*state)->Engine())) {
    (*state)->interface_.reset(new TRTv1Interface(*state));
  } else {
    (*state)->interface_.reset(new TRTv3Interface(*state));
  }
  RETURN_IF_ERROR((*state)->InitOptimizationProfiles());
  RETURN_IF_ERROR((*state)->ValidateIO());
  RETURN_IF_ERROR((*state)->InitIOBindingBuffers());

  (*state)->completion_thread_ =
      std::thread(&ModelInstanceState::ProcessResponse, *state);

  // CUDA 10.1 starts to support CUDA graphs.
  // If enabled, build CUDA graphs with a set of graph specs.
#ifdef TRITON_ENABLE_CUDA_GRAPH
  if (model_state->UseCudaGraphs()) {
    RETURN_IF_ERROR((*state)->InitializeCudaGraph());
  }
#endif

  model_state->RegisterInstance((*state)->DeviceId(), *state);

  std::string profiles_desc;
  (*state)->GetConfiguredProfiles(&profiles_desc);
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Created instance ") + (*state)->Name() + " on GPU " +
       std::to_string((*state)->DeviceId()) + " with stream priority " +
       std::to_string((*state)->CudaStreamPriority()) +
       " and optimization profile" + profiles_desc)
          .c_str());

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : TensorRTModelInstance(model_state, triton_model_instance),
      total_bindings_(0), num_expected_bindings_(0),
      uses_implicit_state_(false), model_state_(model_state)
{
  // 'coalesce_request_input_' is set at backend level
  {
    TRITONBACKEND_Model* model;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelInstanceModel(triton_model_instance, &model));
    TRITONBACKEND_Backend* backend;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelBackend(model, &backend));
    void* state;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_BackendState(backend, &state));
    coalesce_request_input_ =
        reinterpret_cast<BackendConfiguration*>(state)->coalesce_request_input_;
  }

  if (Kind() != TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    throw triton::backend::BackendModelInstanceException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', TensorRT backend supports only GPU device")
            .c_str()));
  }

  signal_stream_ = nullptr;
  input_copy_stream_ = nullptr;
  output_copy_stream_ = nullptr;
  num_copy_streams_ = (model_state_->SeparateOutputStream()) ? 2 : 1;
  next_buffer_binding_set_ = 0;

  next_set_ = 0;
  for (size_t idx = 0; idx < EVENT_SET_COUNT; idx++) {
    events_[idx].input_ready_ = nullptr;
    events_[idx].ready_for_input_ = nullptr;
    events_[idx].output_ready_ = nullptr;
    events_[idx].ready_for_output_ = nullptr;
    events_[idx].compute_input_start_ = nullptr;
    events_[idx].compute_input_end_ = nullptr;
    events_[idx].compute_output_start_ = nullptr;
  }
  support_batching_ = (model_state_->MaxBatchSize() > 0);

  TRITONSERVER_Error* err =
      SupportsIntegratedZeroCopy(DeviceId(), &zero_copy_support_);
  if (err != nullptr) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
    err = nullptr;
    zero_copy_support_ = false;
  } else if (zero_copy_support_) {
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Zero copy optimization is enabled");
  } else {
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Zero copy optimization is disabled");
  }
}

ModelInstanceState::~ModelInstanceState()
{
  cudaSetDevice(DeviceId());
  for (auto& io_binding_infos : io_binding_infos_) {
    for (auto& io_binding_info : io_binding_infos) {
      if (!io_binding_info.IsDynamicShapeOutput() &&
          io_binding_info.GetBuffer() != nullptr) {
        cudaError_t err = cudaSuccess;
        if (io_binding_info.GetMemoryType() == TRITONSERVER_MEMORY_GPU) {
          err = cudaFree(io_binding_info.GetBuffer());
        } else {
          err = cudaFreeHost(io_binding_info.GetBuffer());
        }
        if (err != cudaSuccess) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("Failed to free allocated memory for '") + Name() +
               "': " + cudaGetErrorString(err))
                  .c_str());
        }
      }
    }
  }

  for (auto& trt_context : trt_contexts_) {
    for (const auto& cuda_graph_execs : trt_context.second.cuda_graph_execs_) {
      for (const auto& pr : cuda_graph_execs) {
        cudaError_t err = cudaGraphExecDestroy(pr.second.cuda_graph_exec_);
        if (err != cudaSuccess) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("Failed to destroy cuda graph exec: ") +
               +cudaGetErrorString(err))
                  .c_str());
        }
      }
    }
    trt_context.second.cuda_graph_execs_.clear();
  }

  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Failed to destroy cuda stream: ") +
           +cudaGetErrorString(err))
              .c_str());
    }
    stream_ = nullptr;
  }

  if (signal_stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(signal_stream_);
    if (err != cudaSuccess) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Failed to destroy cuda signal stream: ") +
           +cudaGetErrorString(err))
              .c_str());
    }
    signal_stream_ = nullptr;
  }

  if (input_copy_stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(input_copy_stream_);
    if (err != cudaSuccess) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Failed to destroy cuda input copy stream: ") +
           +cudaGetErrorString(err))
              .c_str());
    }
    input_copy_stream_ = nullptr;
  }

  if (output_copy_stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(output_copy_stream_);
    if (err != cudaSuccess) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Failed to destroy cuda output copy stream: ") +
           +cudaGetErrorString(err))
              .c_str());
    }
    output_copy_stream_ = nullptr;
  }

  DestroyEventSet();

  // Notify the completion thread to exit
  completion_queue_.Put(std::move(std::unique_ptr<Payload>()));
  if (completion_thread_.joinable()) {
    completion_thread_.join();
  }
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Issuing ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  Run(requests, request_count);

  bool run_failed = true;
  for (size_t i = 0; i < request_count; ++i) {
    if (requests[i] != nullptr) {
      run_failed = false;
      break;
    }
  }

  // On error, handle the response here instead of delegating to the
  // completion thread as the completion thread will wait on CUDA
  // events unconditionally, which can be ignored on error.
  if (run_failed) {
    // On inference error, place the slot back in the queue
    // immediately, as all work for the slot should be ignored.
    semaphore_->Release();
  } else {
    auto event_set_idx = next_set_;
    next_set_ = (event_set_idx + 1) % EVENT_SET_COUNT;
    payload_->requests_list_.reserve(payload_->request_count_);
    for (uint32_t i = 0; i < payload_->request_count_; i++) {
      payload_->requests_list_.push_back(payload_->requests_[i]);
    }
    payload_->requests_ = &payload_->requests_list_[0];
    // Put the details needed by the ProcessResponse thread on the
    // queue
    completion_queue_.Put(std::move(payload_));
    next_buffer_binding_set_ =
        (next_buffer_binding_set_ + 1) % num_copy_streams_;

    // Wait until the states are updated. Barrier is only
    // engaged when model has an implicit state.
    if (barrier_.get() != nullptr) {
      barrier_->get_future().wait();
    }
  }
}

void
ModelInstanceState::Run(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  NVTX_RANGE(nvtx_, "Run " + Name());

  // Should set a barrier so that instance execution can be blocked until
  // the state update is completed.
  if (uses_implicit_state_) {
    barrier_.reset(new std::promise<void>());
  }

  // Need to move the TRITONBACKEND_Request objects, as the lifetime
  // must be extended until ProcessResponse completes.
  payload_.reset(new Payload(next_set_, requests, request_count));
  SET_TIMESTAMP(payload_->compute_start_ns_);

  cudaSetDevice(DeviceId());
#ifdef TRITON_ENABLE_STATS
  {
    SET_TIMESTAMP(payload_->compute_start_ns_);
    cudaEventRecord(events_[next_set_].compute_input_start_, signal_stream_);
  }
#endif  // TRITON_ENABLE_STATS

  const int max_batch_size = StateForModel()->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  payload_->total_batch_size_ = 0;
  for (size_t i = 0; i < payload_->request_count_; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (payload_->requests_[i] == nullptr) {
      RequestsRespondWithError(
          payload_->requests_, payload_->request_count_,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to TensorRT backend for '" + Name() + "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs,
      // if the model support batching, the first dimension size is
      // batch size
      TRITONBACKEND_Input* input;
      auto err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        payload_->total_batch_size_ += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      payload_->total_batch_size_ += 1;
    }
  }

  // If there are no valid requests then no need to run the
  // inference. This should never happen unless called with an empty
  // 'requests' for some reason.
  if (payload_->total_batch_size_ == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((payload_->total_batch_size_ != 1) &&
      (payload_->total_batch_size_ > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        payload_->requests_, payload_->request_count_,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(payload_->total_batch_size_) +
                " for '" + Name() + "', max allowed is " +
                std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  std::map<int32_t, std::vector<int32_t>> request_shape_values;
  // Scheduler ensures all the requests have identical shape values so
  // use values from first shape tensor
  TRITONSERVER_Error* err = GetRequestShapeValues(
      payload_->total_batch_size_, payload_->requests_[0],
      &request_shape_values);
  if (err != nullptr) {
    RequestsRespondWithError(
        payload_->requests_, payload_->request_count_, err);
    return;
  }

  std::map<int, TensorRTContext>::iterator citr;
  err = GetMostOptimizedProfile(
      payload_->total_batch_size_, payload_->requests_,
      payload_->request_count_, request_shape_values, &citr);

  if (err != nullptr) {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, TRITONSERVER_ErrorMessage(err));
    TRITONSERVER_ErrorDelete(err);
    err = nullptr;
  }

  int binding_offset = citr->first * num_expected_bindings_;

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response pointer will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  payload_->responses_.reserve(payload_->request_count_);

  for (size_t i = 0; i < payload_->request_count_; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      payload_->responses_.emplace_back(response);
    } else {
      payload_->responses_.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // Calculate the set of event used with the current buffer set
  // in previous execution
  int prev_set = (EVENT_SET_COUNT -
                  (buffer_bindings_.size() % EVENT_SET_COUNT) + next_set_) %
                 EVENT_SET_COUNT;
  auto prev_input_ready_event = model_state_->EagerBatching()
                                    ? events_[prev_set].ready_for_input_
                                    : nullptr;
  std::vector<int64_t> input_dims{(int64_t)payload_->total_batch_size_};
  payload_->collector_.reset(new BackendInputCollector(
      payload_->requests_, payload_->request_count_, &payload_->responses_,
      model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
      input_copy_stream_, events_[next_set_].input_ready_,
      prev_input_ready_event, model_state_->GatherKernelBufferThreshold(),
      HostPolicyName().c_str(), zero_copy_support_, coalesce_request_input_));
  // For each input, concatenate input values from each request into
  // the corresponding binding.
  for (int io_index = 0; io_index < num_expected_bindings_; ++io_index) {
    auto& io_binding_info =
        io_binding_infos_[next_buffer_binding_set_][io_index];
    int binding_index = binding_offset + io_index;

    if (io_binding_info.IsDynamicShapeOutput()) {
      citr->second.context_->setOutputAllocator(
          io_binding_info.GetName().c_str(), io_binding_info.GetAllocator());
    }

    if (!engine_->bindingIsInput(binding_index)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(io_index);

    TRITONSERVER_Error* err = nullptr;
    // Set the shape binding if needed. If unable to set the shape
    // binding then fail all requests.
    if (engine_->isShapeBinding(binding_index)) {
      auto it = request_shape_values.find(io_index);
      if (it != request_shape_values.end()) {
        err = ValidateShapeValues(
            it->second, citr->second.min_shapes_[binding_index],
            citr->second.max_shapes_[binding_index],
            citr->second.nb_shape_values_, support_batching_);
      } else {
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to find shape values for shape input '") +
             name + "' in request for " + Name())
                .c_str());
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->request_count_, payload_->responses_,
            err, "missing shape values for the shape tensor");
      }
      if (err != nullptr) {
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->request_count_, payload_->responses_,
            err, "invalid shape values encountered for shape inputs");
      } else {
        // [FIXME] formalize it, the 'buffer_' may be set directly while forming
        // the shape value
        memcpy(
            io_binding_info.GetBuffer(), &(it->second[0]),
            sizeof(int32_t) * it->second.size());
        citr->second.context_->setInputTensorAddress(
            name.c_str(), io_binding_info.GetBuffer());
        // [FIXME] should be replaced by above, see another use of
        // setInputShapeBinding for detail
        citr->second.context_->setInputShapeBinding(
            binding_index,
            reinterpret_cast<int32_t*>(io_binding_info.GetBuffer()));
      }
    }

    // Skip the upcoming section if it is a shape tensor
    if (engine_->isShapeInferenceIO(name.c_str())) {
      continue;
    }

    // FIXME inefficient as looping in this way may iterate the same
    // source_input multiple times
    if (io_binding_info.GetBatchInput() != nullptr) {
      std::vector<int64_t> shape;
      const auto& batch_input = io_binding_info.GetBatchInput()->first;
      auto& allocated_memory = io_binding_info.GetBatchInput()->second;
      TRITONSERVER_MemoryType mem_type = allocated_memory->MemoryType();
      int64_t mem_type_id = allocated_memory->MemoryTypeId();
      char* input_buffer = allocated_memory->MemoryPtr();
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          payload_->collector_->BatchInputShape(batch_input, &shape),
          "error getting the batch input shape");
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          interface_->SetBindingDimensions(
              name, shape, citr->second, io_index, binding_index, &input_dims),
          "error setting the binding dimension");

      TRITONSERVER_DataType datatype = batch_input.DataType();
      size_t total_byte_size = GetByteSize(datatype, shape);

      const char* dst_buffer;
      size_t dst_buffer_byte_size;
      TRITONSERVER_MemoryType dst_memory_type;
      int64_t dst_memory_type_id;
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          payload_->collector_->ProcessBatchInput(
              batch_input, input_buffer, total_byte_size,
              {{mem_type, mem_type_id}}, &dst_buffer, &dst_buffer_byte_size,
              &dst_memory_type, &dst_memory_type_id),
          "error setting the batch input value");

      if ((batch_input.BatchInputKind() !=
           BatchInput::Kind::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE) &&
          (io_binding_info.GetMemoryType() == TRITONSERVER_MEMORY_GPU)) {
        bool cuda_used = false;

        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->request_count_, payload_->responses_,
            CopyBuffer(
                name, mem_type, mem_type_id, io_binding_info.GetMemoryType(),
                io_binding_info.GetMemoryTypeId(), total_byte_size,
                input_buffer, io_binding_info.GetBuffer(), input_copy_stream_,
                &cuda_used),
            "error copying the batch input buffer");
        if (cuda_used) {
          cudaEventRecord(events_[next_set_].input_ready_, input_copy_stream_);
        }
      }
    } else if (io_binding_info.IsBufferRagged()) {
      std::vector<int64_t> ragged_shape{0};
      TRITONSERVER_DataType datatype;
      for (size_t req_idx = 0; req_idx < payload_->request_count_; req_idx++) {
        TRITONBACKEND_Input* repr_input;
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->request_count_, payload_->responses_,
            TRITONBACKEND_RequestInput(
                payload_->requests_[req_idx], name.c_str(), &repr_input),
            (std::string("failed to obtain the input '") + name + "'").c_str());

        TRITONSERVER_DataType temp_dt;
        const int64_t* shape;
        uint32_t dims_count;
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->request_count_, payload_->responses_,
            TRITONBACKEND_InputProperties(
                repr_input, nullptr, &temp_dt, &shape, &dims_count, nullptr,
                nullptr),
            (std::string("failed to obtain the input properties for '") + name +
             "'")
                .c_str());

        ragged_shape[0] += backend::GetElementCount(shape, dims_count);
        if (req_idx == 0) {
          datatype = temp_dt;
        }
      }

      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          interface_->SetBindingDimensions(
              name, ragged_shape, citr->second, io_index, binding_index,
              &input_dims),
          "error setting the binding dimension");

      size_t total_byte_size = GetByteSize(datatype, ragged_shape);

      payload_->collector_->ProcessTensor(
          name.c_str(), static_cast<char*>(io_binding_info.GetBuffer()),
          total_byte_size, io_binding_info.GetMemoryType(),
          io_binding_info.GetMemoryTypeId());
    } else {
      TRITONBACKEND_Input* repr_input;
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          TRITONBACKEND_RequestInput(
              payload_->requests_[0], name.c_str(), &repr_input),
          (std::string("failed to obtain the representative input '") + name +
           "'")
              .c_str());

      // Get the shape of the input. The request has already checked
      // that the request shape is valid so don't need to do it here.
      TRITONSERVER_DataType datatype;
      const int64_t* shape;
      uint32_t dims_count;
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          TRITONBACKEND_InputProperties(
              repr_input, nullptr, &datatype, &shape, &dims_count, nullptr,
              nullptr),
          (std::string("failed to obtain the representative input "
                       "properties for '") +
           name + "'")
              .c_str());

      // The shape for the entire input batch, [total_batch_size, ...]
      std::vector<int64_t> batchn_shape;
      uint64_t dim_idx = 0;
      batchn_shape.reserve(dims_count);
      if (support_batching_) {
        if (!engine_->isShapeBinding(binding_index)) {
          batchn_shape.push_back(payload_->total_batch_size_);
          dim_idx = 1;
        }
      }
      while (dim_idx < dims_count) {
        batchn_shape.push_back(shape[dim_idx++]);
      }

      // Set the binding dimension so that output dimensions can be
      // obtained
      if (!engine_->isShapeBinding(binding_index)) {
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->request_count_, payload_->responses_,
            interface_->SetBindingDimensions(
                name, batchn_shape, citr->second, io_index, binding_index,
                &input_dims),
            "error setting the binding dimension");
      }

      size_t total_byte_size = 0;
      if (io_binding_info.GetFormat().is_linear_format_) {
        total_byte_size = GetByteSize(datatype, batchn_shape);
      } else {
        batchn_shape[io_binding_info.GetFormat().vectorized_dim_] +=
            (io_binding_info.GetFormat().components_per_element_ -
             (batchn_shape[io_binding_info.GetFormat().vectorized_dim_] %
              io_binding_info.GetFormat().components_per_element_));
        total_byte_size = GetByteSize(datatype, batchn_shape);
      }

      payload_->collector_->ProcessTensor(
          name.c_str(), static_cast<char*>(io_binding_info.GetBuffer()),
          total_byte_size, io_binding_info.GetMemoryType(),
          io_binding_info.GetMemoryTypeId());
    }
  }
  payload_->collector_->Finalize();

  const TensorRTContext::CudaGraph* cuda_graph = nullptr;
  bool found_exact = false;
  // FIXME closest_cuda_graph
  FindClosestCudaGraph(citr->second, input_dims, &cuda_graph, &found_exact);
  // [DLIS-4283] should re-visit below...
  // is below even necessary if we are going to launch a CUDA graph?
  // regular input buffer has been filled and batch input buffer has been set,
  // unless the below is something that must be changed based on selected graph
  if ((cuda_graph != nullptr) && !found_exact) {
    size_t input_idx = 0;
    for (int io_index = 0; io_index < num_expected_bindings_; ++io_index) {
      auto& io_binding_info =
          io_binding_infos_[next_buffer_binding_set_][io_index];
      int binding_index = binding_offset + io_index;
      if (!engine_->bindingIsInput(binding_index) ||
          engine_->isShapeBinding(binding_index)) {
        continue;
      }
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          interface_->SetBindingDimensions(
              "CUDA graph input", cuda_graph->input_dims_[input_idx],
              citr->second, io_index, binding_index, nullptr),
          "error setting the binding dimension");

      // Initialize additional entries in batch input
      if (io_binding_info.GetBatchInput() != nullptr) {
        const auto& batch_input = io_binding_info.GetBatchInput()->first;
        const size_t total_byte_size = GetByteSize(
            batch_input.DataType(), cuda_graph->input_dims_[input_idx]);

        auto& allocated_memory = io_binding_info.GetBatchInput()->second;
        TRITONSERVER_MemoryType mem_type = allocated_memory->MemoryType();
        int64_t mem_type_id = allocated_memory->MemoryTypeId();
        char* input_buffer = allocated_memory->MemoryPtr();

        const char* dst_buffer;
        size_t dst_buffer_byte_size;
        TRITONSERVER_MemoryType dst_memory_type;
        int64_t dst_memory_type_id;
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->request_count_, payload_->responses_,
            payload_->collector_->ProcessBatchInput(
                batch_input, input_buffer, total_byte_size,
                {{mem_type, mem_type_id}}, &dst_buffer, &dst_buffer_byte_size,
                &dst_memory_type, &dst_memory_type_id),
            "error setting the bath input value");

        if ((batch_input.BatchInputKind() !=
             BatchInput::Kind::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE) &&
            (io_binding_info.GetMemoryType() == TRITONSERVER_MEMORY_GPU)) {
          bool cuda_used = false;
          FAIL_ALL_AND_RETURN_IF_ERROR(
              payload_->requests_, payload_->request_count_,
              payload_->responses_,
              CopyBuffer(
                  "CUDA graph batch input", mem_type, mem_type_id,
                  io_binding_info.GetMemoryType(),
                  io_binding_info.GetMemoryTypeId(), total_byte_size,
                  input_buffer, io_binding_info.GetBuffer(), input_copy_stream_,
                  &cuda_used),
              "error copying the batch input buffer");
          if (cuda_used) {
            cudaEventRecord(
                events_[next_set_].input_ready_, input_copy_stream_);
          }
        }
      }
      input_idx++;
    }
  }

  // Ensure inputs are ready before execution.
  // Output buffers are guaranteed to be available at this point when
  // the execution and output copy are on the same stream.
  cudaStreamWaitEvent(stream_, events_[next_set_].input_ready_, 0);

#ifdef TRITON_ENABLE_STATS
  cudaStreamWaitEvent(signal_stream_, events_[next_set_].input_ready_, 0);
  cudaEventRecord(events_[next_set_].compute_input_end_, signal_stream_);
#endif  // TRITON_ENABLE_STATS

  // Wait for the output buffers to be available at this point when
  // the execution and output copy are on separate streams
  if (model_state_->SeparateOutputStream()) {
    cudaStreamWaitEvent(stream_, events_[next_set_].output_ready_, 0);
  }
  // Async execute the inference using a CUDA graph if available for
  // the batch-size, otherwise execution normally.
  if (cuda_graph != nullptr) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Context with profile ") + citr->second.profile_name_ +
         " [" + std::to_string(citr->first) + "] is launching CUDA graph for " +
         Name())
            .c_str());

    cudaError_t cuda_err =
        cudaGraphLaunch(cuda_graph->cuda_graph_exec_, stream_);
    if (cuda_err != cudaSuccess) {
      cudaStreamSynchronize(stream_);
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("unable to execute graph for inference ") + name_ +
               ": " + cudaGetErrorString(cuda_err))
                  .c_str()),
          "failed to run TRT inference");
    }
    // [DLIS-4283] [FIXME] should not need this as TRT adopted the new CUDA
    // graph API that exposed event activity
    // Event recorded during CUDA graph capture is not visible outside
    // of the graph, need to explicitly record it.
    cudaEventRecord(events_[next_set_].ready_for_input_, stream_);
  } else {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Context with profile ") + citr->second.profile_name_ +
         " [" + std::to_string(citr->first) + "] is being executed for " +
         Name())
            .c_str());

    if (!citr->second.context_->allInputDimensionsSpecified()) {
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to specify the dimensions of all input "
              "bindings"),
          "failed to run TRT inference");
    }
    if (!citr->second.context_->allInputShapesSpecified()) {
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to specify the values for all input shape "
              "tensors"),
          "failed to run TRT inference");
    }

    if (!interface_->Enqueue(citr->second.context_.get())) {
      cudaStreamSynchronize(stream_);
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload_->requests_, payload_->request_count_, payload_->responses_,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("unable to enqueue for inference ") + Name())
                  .c_str()),
          "failed to run TRT inference");
    }
  }

  cudaEventRecord(events_[next_set_].ready_for_output_, stream_);

#ifdef TRITON_ENABLE_STATS
  cudaStreamWaitEvent(signal_stream_, events_[next_set_].ready_for_output_, 0);
  cudaEventRecord(events_[next_set_].compute_output_start_, signal_stream_);
#endif  // TRITON_ENABLE_STATS

  // Collect the names of requested outputs. Do not include outputs
  // for requests that have already responded with an error.
  std::vector<std::set<std::string>> request_required_outputs(
      payload_->request_count_);
  for (size_t idx = 0; idx < payload_->request_count_; idx++) {
    const auto& request = payload_->requests_[idx];
    auto& response = payload_->responses_[idx];
    if (response != nullptr) {
      uint32_t output_count;
      RESPOND_AND_SET_NULL_IF_ERROR(
          &response, TRITONBACKEND_RequestOutputCount(request, &output_count));
      if (response != nullptr) {
        for (uint32_t output_idx = 0; output_idx < output_count; output_idx++) {
          const char* output_name;
          RESPOND_AND_SET_NULL_IF_ERROR(
              &response, TRITONBACKEND_RequestOutputName(
                             request, output_idx, &output_name));
          if (response != nullptr) {
            request_required_outputs[idx].insert(output_name);
          }
        }
      }
    }
  }

  // Wait for the inference to be completed before copying output if
  // output copy is on a separate stream
  if (model_state_->SeparateOutputStream()) {
    cudaStreamWaitEvent(
        output_copy_stream_, events_[next_set_].ready_for_output_, 0);
  }

  const auto output_stream =
      model_state_->SeparateOutputStream() ? output_copy_stream_ : stream_;

  // For each requested output, verify that the output can accept the
  // actual model output and then copy that output from the GPU
  payload_->responder_.reset(new BackendOutputResponder(
      payload_->requests_, payload_->request_count_, &payload_->responses_,
      model_state_->TritonMemoryManager(), model_state_->MaxBatchSize() > 0,
      model_state_->EnablePinnedOutput(), output_stream,
      events_[next_set_].output_ready_, zero_copy_support_));
  for (int io_index = 0; io_index < num_expected_bindings_; ++io_index) {
    auto& io_binding_info =
        io_binding_infos_[next_buffer_binding_set_][io_index];
    int binding_index = binding_offset + io_index;
    if (engine_->bindingIsInput(binding_index)) {
      continue;
    }

    const std::string& name = engine_->getBindingName(io_index);

    nvinfer1::Dims dims;
    dims = citr->second.context_->getBindingDimensions(binding_index);

    // Make sure each output is of the expected size and copy it into
    // the payload responses.
    bool cuda_copy = false;
    if (engine_->isShapeBinding(binding_index)) {
      // Custom handling for shape tensors
      // Obtain the shape value
      if (dims.nbDims != 0) {
        auto shape_value_ptr = reinterpret_cast<int32_t const*>(
            citr->second.context_->getTensorAddress(name.c_str()));

        // The first shape value must be equal to the total batch_size
        if (support_batching_ &&
            payload_->total_batch_size_ != (uint32_t)*shape_value_ptr) {
          FAIL_ALL_AND_RETURN_IF_ERROR(
              payload_->requests_, payload_->request_count_,
              payload_->responses_,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  (std::string("failed to retrieve the output shape values "
                               "from binding '") +
                   Name() + "'")
                      .c_str()),
              "failed to run TRT response");
        }

        std::vector<int64_t> batchn_shape;
        if (support_batching_) {
          batchn_shape.push_back(payload_->total_batch_size_);
          batchn_shape.push_back(dims.d[0] - 1);
        } else {
          batchn_shape.push_back(dims.d[0]);
        }

        for (size_t idx = 0; idx < payload_->responses_.size(); idx++) {
          auto& request = payload_->requests_[idx];
          auto& response = payload_->responses_[idx];

          if (support_batching_) {
            TRITONBACKEND_Input* input;
            TRITONBACKEND_RequestInputByIndex(request, 0, &input);
            const int64_t* shape;
            RESPOND_AND_SET_NULL_IF_ERROR(
                &response, TRITONBACKEND_InputProperties(
                               input, nullptr, nullptr, &shape, nullptr,
                               nullptr, nullptr));
            batchn_shape[0] = shape[0];
          }

          const size_t tensor_element_cnt =
              backend::GetElementCount(batchn_shape);

          TRITONSERVER_DataType dt = ConvertTrtTypeToDataType(
              engine_->getBindingDataType(binding_index));

          // Only need an response tensor for requested outputs.
          if ((response != nullptr) &&
              (request_required_outputs[idx].find(name) !=
               request_required_outputs[idx].end())) {
            TRITONBACKEND_Output* response_output = nullptr;
            RESPOND_AND_SET_NULL_IF_ERROR(
                &response, TRITONBACKEND_ResponseOutput(
                               response, &response_output, name.c_str(), dt,
                               batchn_shape.data(), batchn_shape.size()));
            cuda_copy |= SetOutputShapeTensorBuffer(
                shape_value_ptr, &response, response_output, tensor_element_cnt,
                batchn_shape[0], stream_);
          }
        }
      }
    } else if (io_binding_info.IsBufferRagged()) {
      // FIXME add correctness checking like below
      io_binding_info.SetBatchOutput(model_state_->FindBatchOutput(name));

      // Process the output tensors with pinned memory address if zero-copy is
      // supported, otherwise use device memory. Perform memory copies
      // asynchronously and wait for model execution.
      payload_->responder_->ProcessBatchOutput(
          name, *(io_binding_info.GetBatchOutput()),
          static_cast<const char*>(io_binding_info.GetBuffer()),
          io_binding_info.GetMemoryType(), io_binding_info.GetMemoryTypeId());
    } else {
      std::vector<int64_t> batchn_shape;

      if (engine_->hasImplicitBatchDimension() && support_batching_) {
        batchn_shape.push_back(payload_->total_batch_size_);
      }

      for (int i = 0; i < dims.nbDims; ++i) {
        batchn_shape.push_back(dims.d[i]);
      }

      TRITONSERVER_DataType dt =
          ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));

      // FIXME process reformat-free output, need to update output
      // process code to accept batch1_byte_size and request batch
      // size to break down output buffer properly.
      size_t batch1_byte_size = GetByteSize(dt, batchn_shape);
      if (support_batching_) {
        batch1_byte_size /= payload_->total_batch_size_;
      }

      if (io_binding_info.GetByteSize() <
              (batch1_byte_size * payload_->total_batch_size_) &&
          !io_binding_info.IsDynamicShapeOutput()) {
        FAIL_ALL_AND_RETURN_IF_ERROR(
            payload_->requests_, payload_->request_count_, payload_->responses_,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                (std::string("unexpected size for output '") + name +
                 "', byte-size " +
                 std::to_string(io_binding_info.GetByteSize()) +
                 " is less than " +
                 std::to_string(payload_->total_batch_size_) + " * " +
                 std::to_string(batch1_byte_size))
                    .c_str()),
            "failed to run TRT response");
      }

      if (io_binding_info.IsRequestedOutputTensor()) {
        // Process the output tensors with pinned memory address if zero-copy is
        // supported, otherwise use device memory. Perform memory copies
        // asynchronously and wait for model execution.
        payload_->responder_->ProcessTensor(
            name, dt, batchn_shape,
            static_cast<const char*>(io_binding_info.GetBuffer()),
            io_binding_info.GetMemoryType(), io_binding_info.GetMemoryTypeId());
      }

      if (io_binding_info.IsStateOutput()) {
        auto updated_states = payload_->responder_->ProcessStateTensor(
            name, dt, batchn_shape,
            static_cast<const char*>(io_binding_info.GetBuffer()),
            io_binding_info.GetMemoryType(), io_binding_info.GetMemoryTypeId());
        payload_->seq_states_.insert(
            payload_->seq_states_.end(), updated_states.begin(),
            updated_states.end());
      }
    }
  }
}

bool
ModelInstanceState::SetOutputShapeTensorBuffer(
    const int32_t* content, TRITONBACKEND_Response** response,
    TRITONBACKEND_Output* response_output, const size_t tensor_element_count,
    const int64_t batch_size, cudaStream_t stream)
{
  bool cuda_copy = false;

  const size_t expected_byte_size = tensor_element_count * sizeof(int32_t);

  // Allocate a buffer large enough to hold the serialized tensor.
  TRITONSERVER_MemoryType actual_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t actual_memory_type_id = 0;

  void* buffer;
  auto err = TRITONBACKEND_OutputBuffer(
      response_output, &buffer, expected_byte_size, &actual_memory_type,
      &actual_memory_type_id);
  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    return cuda_copy;
  }

  const size_t nb_shape_values = tensor_element_count / batch_size;

  // Copy the serialized tensor into the allocated buffer.
  bool cuda_used = false;
  size_t content_offset = support_batching_ ? 1 : 0;
  size_t buffer_offset = 0;
  for (int i = 0; i < batch_size; i++) {
    err = CopyBuffer(
        "Shape tensor output", TRITONSERVER_MEMORY_CPU /* src_memory_type */,
        0 /* src_memory_type_id */, actual_memory_type, actual_memory_type_id,
        nb_shape_values * sizeof(int32_t), (void*)(content + content_offset),
        (void*)((char*)buffer + buffer_offset), stream_, &cuda_used);
    cuda_copy |= cuda_used;
    buffer_offset += (nb_shape_values * sizeof(int32_t));
  }

  if (err != nullptr) {
    RESPOND_AND_SET_NULL_IF_ERROR(response, err);
    return cuda_copy;
  }

  return cuda_copy;
}

void
ModelInstanceState::ProcessResponse()
{
  while (true) {
    NVTX_RANGE(nvtx_, "ProcessResponse " + Name());
    auto payload = std::move(completion_queue_.Get());
    if (payload.get() == nullptr) {
      break;
    }
    auto& event_set = events_[payload->event_set_idx_];

    // The model execution associated with the current slot
    // has consumed the inputs. Put the slot back into the available
    // slots so that it can begin enqueuing new memcpys into the input
    // buffers
    cudaEventSynchronize(event_set.ready_for_input_);

    // This will be empty unless TRITONSERVER_RESET_BINDING_BUFFERS is set to 1
    for (auto& buffer_binding_pair : payload->buffer_input_binding_pairs_) {
      cudaMemsetAsync(
          buffer_binding_pair.first, 0, buffer_binding_pair.second,
          input_copy_stream_);
    }

    semaphore_->Release();
    NVTX_MARKER("plan_input_available");

    // Call Finalize() here to defer CUDA synchronization as much as
    // possible
    payload->responder_->Finalize();
    cudaEventSynchronize(event_set.output_ready_);
    NVTX_MARKER("plan_output_ready");

    // Update the states
    for (auto& state : payload->seq_states_) {
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload->requests_, payload->request_count_, payload->responses_,
          TRITONBACKEND_StateUpdate(state), "failed to update state");
    }
    // Signal the state update completion.
    if (barrier_.get() != nullptr) {
      barrier_->set_value();
    }

    // Compute ends when the output data copy is completed
    uint64_t compute_end_ns = 0;
#ifdef TRITON_ENABLE_STATS
    cudaEventSynchronize(event_set.compute_output_start_);
    float compute_infer = 0;
    LOG_IF_CUDA_ERROR(
        cudaEventElapsedTime(
            &compute_infer, event_set.compute_input_end_,
            event_set.compute_output_start_),
        "elapsed failed");

    float compute_input = 0;
    LOG_IF_CUDA_ERROR(
        cudaEventElapsedTime(
            &compute_input, event_set.compute_input_start_,
            event_set.compute_input_end_),
        "elapsed time failed.");

    payload->compute_input_end_ns_ =
        payload->compute_start_ns_ + (compute_input * 1e6);
    payload->compute_output_start_ns_ =
        payload->compute_input_end_ns_ + (compute_infer * 1e6);
    SET_TIMESTAMP(compute_end_ns);
#endif  // TRITON_ENABLE_STATS

    // Send all the responses that haven't already been sent because
    // of an earlier error. Note that the responses are not set to
    // nullptr here as we need that indication below to determine if
    // the request we successful or not.
    for (auto& response : payload->responses_) {
      if (response != nullptr) {
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
            "failed to send TensorRT backend response");
      }
    }

    // Report statistics for each request.
    for (uint32_t r = 0; r < payload->request_count_; ++r) {
      auto& request = payload->requests_[r];
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              TritonModelInstance(), request,
              (payload->responses_[r] != nullptr) /* success */,
              payload->compute_start_ns_, payload->compute_input_end_ns_,
              payload->compute_output_start_ns_, compute_end_ns),
          "failed reporting request statistics");

      LOG_IF_ERROR(
          TRITONBACKEND_RequestRelease(
              request, TRITONSERVER_REQUEST_RELEASE_ALL),
          "failed releasing request");
    }

    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), payload->total_batch_size_,
            payload->compute_start_ns_, payload->compute_input_end_ns_,
            payload->compute_output_start_ns_, compute_end_ns),
        "failed reporting batch request statistics");

    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
         " released " + std::to_string(payload->request_count_) + " requests")
            .c_str());
  }
}

TRITONSERVER_Error*
ModelInstanceState::GetRequestShapeValues(
    size_t total_batch_size, TRITONBACKEND_Request* request,
    std::map<int, std::vector<int32_t>>* request_shape_values)
{
  // Visit all the inputs and extract the shape values present in the
  // request
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(request, &input_count));
  for (uint32_t i = 0; i < input_count; i++) {
    TRITONBACKEND_Input* input;
    TRITONBACKEND_RequestInputByIndex(request, i, &input);
    const char* input_name;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint32_t dims_count;
    uint64_t byte_size;
    uint32_t buffer_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &datatype, &shape, &dims_count, &byte_size,
        &buffer_count));

    int io_index = engine_->getBindingIndex(input_name);
    if (engine_->isShapeBinding(io_index)) {
      auto it =
          request_shape_values->emplace(io_index, std::vector<int32_t>()).first;
      if (support_batching_) {
        it->second.push_back((int32_t)total_batch_size);
      }

      // For now being conservative and requiring that shape tensors
      // be in a single buffer on the CPU. We can handle more cases in
      // future if necessary.
      if (buffer_count != 1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("shape tensor for input '") + input_name +
             "' must be in single contiguous buffer on CPU")
                .c_str());
      }

      size_t data_byte_size;
      TRITONSERVER_MemoryType data_memory_type;
      int64_t data_memory_id;
      const char* data_buffer;
      RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
          input, 0 /* idx */, reinterpret_cast<const void**>(&data_buffer),
          &data_byte_size, &data_memory_type, &data_memory_id));

      if ((data_buffer == nullptr) ||
          (data_memory_type == TRITONSERVER_MEMORY_GPU)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("shape tensor for input '") + input_name +
             "' must be in single contiguous buffer on CPU")
                .c_str());
      }

      // Shape tensors datatype is INT32.
      int64_t element_cnt = backend::GetElementCount(shape, dims_count);
      if (support_batching_) {
        element_cnt /= shape[0];
      }
      const size_t expected_byte_size =
          element_cnt * GetByteSize(TRITONSERVER_TYPE_INT32, {1});

      bool includes_batch_shape_value = false;
      if (expected_byte_size != data_byte_size) {
        if (expected_byte_size == (data_byte_size - sizeof(int32_t))) {
          includes_batch_shape_value = true;
        } else {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("shape tensor for input '") + input_name +
               "' expected byte size is " + std::to_string(expected_byte_size) +
               " [ or " + std::to_string(expected_byte_size + sizeof(int32_t)) +
               " if input includes batch shape value] " + ", got " +
               std::to_string(data_byte_size))
                  .c_str());
        }
      }

      const int32_t* dims = reinterpret_cast<const int32_t*>(data_buffer);
      int64_t offset = includes_batch_shape_value ? 1 : 0;
      for (int64_t i = offset; i < element_cnt; ++i) {
        it->second.push_back(dims[i]);
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::GetMostOptimizedProfile(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    uint32_t request_count,
    const std::map<int, std::vector<int32_t>>& request_shape_values,
    std::map<int, TensorRTContext>::iterator* citr)
{
  // Returns the TensorRT context that uses profile with shortest
  // Manhattan distance in terms of input dimensions [TODO] traverse
  // it with more efficient data structure (i.e. K-D tree)
  *citr = trt_contexts_.begin();
  if (trt_contexts_.size() != 1) {
    int64_t shortest_distance = LLONG_MAX;
    for (auto cit = trt_contexts_.begin(); cit != trt_contexts_.end(); cit++) {
      int64_t current_distance = 0;
      EvaluateTensorRTContext(
          cit, total_batch_size, requests, request_count, request_shape_values,
          &current_distance);
      if (current_distance < shortest_distance) {
        *citr = cit;
        shortest_distance = current_distance;
      }
    }
    if (shortest_distance == LLONG_MAX) {
      std::string profiles_str;
      for (const auto& trt_context : trt_contexts_) {
        profiles_str +=
            (" " + trt_context.second.profile_name_ + "[" +
             std::to_string(trt_context.first) + "]");
      }
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("failed to find any Optimization Profile among [") +
           profiles_str +
           "] to support the "
           "requested dimensions (or shape values), proceeding with "
           "first "
           "profile.")
              .c_str());
    }
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("Optimization profile ") + (*citr)->second.profile_name_ +
       " [" + std::to_string((*citr)->first) + "] is selected for " + Name())
          .c_str());
  return nullptr;
}


TRITONSERVER_Error*
ModelInstanceState::EvaluateTensorRTContext(
    std::map<int, TensorRTContext>::iterator& citr, size_t total_batch_size,
    TRITONBACKEND_Request** requests, uint32_t request_count,
    const std::map<int, std::vector<int32_t>>& request_shape_values,
    int64_t* error_distance)
{
  *error_distance = 0;

  // Visit all the inputs and extract the shape values present in the
  // request
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  for (uint32_t i = 0; i < input_count; i++) {
    TRITONBACKEND_Input* input;
    TRITONBACKEND_RequestInputByIndex(requests[0], i, &input);
    const char* input_name;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, nullptr, &input_shape, &input_dims_count, nullptr,
        nullptr));
    std::vector<int64_t> input_shape_vec;
    for (uint32_t dim_idx = 0; dim_idx < input_dims_count; dim_idx++) {
      input_shape_vec.push_back(*(input_shape + dim_idx));
    }
    if (support_batching_) {
      input_shape_vec[0] = total_batch_size;
    }

    int io_index = engine_->getBindingIndex(input_name);
    auto& io_binding_info =
        io_binding_infos_[next_buffer_binding_set_][io_index];
    if (io_binding_info.IsBufferRagged()) {
      std::vector<int64_t> shape_vec{0};
      for (uint32_t req_idx = 0; req_idx < request_count; req_idx++) {
        TRITONBACKEND_Input* repr_input;
        RETURN_IF_ERROR(TRITONBACKEND_RequestInput(
            requests[req_idx], input_name, &repr_input));
        const int64_t* shape;
        uint32_t dims_count;
        RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
            repr_input, nullptr, nullptr, &shape, &dims_count, nullptr,
            nullptr));
        shape_vec[0] += backend::GetElementCount(shape, dims_count);
      }
      auto err = ValidateDimension(
          shape_vec, citr->second.min_dims_[io_index],
          citr->second.max_dims_[io_index], false);
      if (err != nullptr) {
        *error_distance = LLONG_MAX;
        TRITONSERVER_ErrorDelete(err);
        break;
      } else {
        const auto& opt_dims = citr->second.opt_dims_[io_index];
        *error_distance += std::abs(opt_dims.d[0] - shape_vec[0]);
      }
    } else {
      TRITONSERVER_Error* err;
      err = ValidateDimension(
          input_shape_vec, citr->second.min_dims_[io_index],
          citr->second.max_dims_[io_index], false);

      bool valid_bs = true;
      TRITONSERVER_Error* shape_err = nullptr;
      bool missing_shape_values = false;
      if (engine_->isShapeBinding(io_index)) {
        auto it = request_shape_values.find(io_index);
        if (it != request_shape_values.end()) {
          shape_err = ValidateShapeValues(
              it->second, citr->second.min_shapes_[io_index],
              citr->second.max_shapes_[io_index], citr->second.nb_shape_values_,
              support_batching_);
          valid_bs =
              (!support_batching_) || (((int32_t)total_batch_size >=
                                        *citr->second.min_shapes_[io_index]) &&
                                       ((int64_t)total_batch_size <=
                                        *citr->second.max_shapes_[io_index]));
        } else {
          missing_shape_values = true;
        }
      }

      if ((err != nullptr) || (shape_err != nullptr) || !valid_bs ||
          missing_shape_values) {
        *error_distance = LLONG_MAX;
        if (err != nullptr) {
          TRITONSERVER_ErrorDelete(err);
        }
        if (shape_err != nullptr) {
          TRITONSERVER_ErrorDelete(shape_err);
        }
        break;
      } else {
        const auto& opt_dims = citr->second.opt_dims_[io_index];
        *error_distance += std::abs(opt_dims.d[0] - (int64_t)total_batch_size);
        for (int idx = 1; idx < opt_dims.nbDims; idx++) {
          *error_distance +=
              std::abs(opt_dims.d[idx] - input_shape_vec[idx - 1]);
        }
        if (engine_->isShapeBinding(io_index)) {
          const auto* opt_shape_values = citr->second.opt_shapes_[io_index];
          *error_distance +=
              std::abs(*opt_shape_values - (int64_t)total_batch_size);
          auto it = request_shape_values.find(io_index);
          for (size_t idx = 1; idx < citr->second.nb_shape_values_; idx++) {
            *error_distance +=
                std::abs(*(opt_shape_values + idx) - it->second[idx - 1]);
          }
        }
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitStreamsAndEvents()
{
  // Set the device before preparing the context.
  auto cuerr = cudaSetDevice(DeviceId());
  if (cuerr != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, (std::string("unable to set device for ") +
                                      Name() + ": " + cudaGetErrorString(cuerr))
                                         .c_str());
  }

  // Create CUDA streams associated with the instance
  cuda_stream_priority_ = model_state_->GetCudaStreamPriority();

  // The stream created by default has set priority of 0. Destroy the
  // the default stream and create a new stream with requested
  // priority.
  // FIXME, This should be moved to backend repo to directly build
  // cuda stream with required priority.
  if (cuda_stream_priority_ != 0) {
    if (stream_ != nullptr) {
      cudaError_t err = cudaStreamDestroy(stream_);
      if (err != cudaSuccess) {
        TRITONSERVER_LogMessage(
            TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
            (std::string("~BackendModelInstance: ") + name_ +
             " failed to destroy cuda stream: " + cudaGetErrorString(err))
                .c_str());
      }
      stream_ = nullptr;
      RETURN_IF_ERROR(
          CreateCudaStream(DeviceId(), cuda_stream_priority_, &stream_));
    }
  }
#ifdef TRITON_ENABLE_STATS
  RETURN_IF_ERROR(
      CreateCudaStream(DeviceId(), cuda_stream_priority_, &signal_stream_));
#endif  // TRITON_ENABLE_STATS
  RETURN_IF_ERROR(
      CreateCudaStream(DeviceId(), cuda_stream_priority_, &input_copy_stream_));
  if (model_state_->SeparateOutputStream()) {
    RETURN_IF_ERROR(CreateCudaStream(
        DeviceId(), cuda_stream_priority_, &output_copy_stream_));
  }
  // Create CUDA events associated with the execution states
  RETURN_IF_ERROR(InitEventSet(model_state_->BusyWaitEvents()));

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitEventSet(bool busy_wait_events)
{
  unsigned int event_flags =
      (busy_wait_events ? cudaEventDefault : cudaEventBlockingSync) |
      cudaEventDisableTiming;

  for (size_t idx = 0; idx < EVENT_SET_COUNT; idx++) {
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " ready for input", event_flags,
        &events_[idx].ready_for_input_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " input ready", event_flags,
        &events_[idx].input_ready_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " ready for output", event_flags,
        &events_[idx].ready_for_output_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " output ready", event_flags,
        &events_[idx].output_ready_));
#ifdef TRITON_ENABLE_STATS
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " compute output start",
        cudaEventDefault, &events_[idx].compute_output_start_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " compute input start", cudaEventDefault,
        &events_[idx].compute_input_start_));
    RETURN_IF_ERROR(CreateCudaEvent(
        "Set " + std::to_string(idx) + " compute input end", cudaEventDefault,
        &events_[idx].compute_input_end_));
#endif  // TRITON_ENABLE_STATS
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::DestroyEventSet()
{
  for (size_t idx = 0; idx < EVENT_SET_COUNT; idx++) {
    if (events_[idx].ready_for_input_ != nullptr) {
      cudaEventDestroy(events_[idx].ready_for_input_);
    }
    if (events_[idx].input_ready_ != nullptr) {
      cudaEventDestroy(events_[idx].input_ready_);
    }
    if (events_[idx].ready_for_output_ != nullptr) {
      cudaEventDestroy(events_[idx].ready_for_output_);
    }
    if (events_[idx].output_ready_ != nullptr) {
      cudaEventDestroy(events_[idx].output_ready_);
    }
#ifdef TRITON_ENABLE_STATS
    if (events_[idx].compute_output_start_ != nullptr) {
      cudaEventDestroy(events_[idx].compute_output_start_);
    }
    if (events_[idx].compute_input_start_ != nullptr) {
      cudaEventDestroy(events_[idx].compute_input_start_);
    }
    if (events_[idx].compute_input_end_ != nullptr) {
      cudaEventDestroy(events_[idx].compute_input_end_);
    }
#endif
  }
  return nullptr;
}

void
ModelInstanceState::InitSemaphore()
{
  // If eager batching is set, we add to the semaphore resource count
  // which allows to start preparing next batch before the previous
  // batch has completed. The number of duplicates are limited by
  // number of event sets to prevent too many iterations are run
  // ahead and to avoid interference of the event communication in
  // the previous execution
  int sem_count = (model_state_->EagerBatching()) ? EVENT_SET_COUNT : 1;
  semaphore_.reset(new Semaphore(sem_count));
}

TRITONSERVER_Error*
ModelInstanceState::InitOptimizationProfiles()
{
  total_bindings_ = engine_->getNbBindings();
  const int total_profiles = engine_->getNbOptimizationProfiles();

  // TRT sets the optimization profile index to be 0 implicitly with
  // the first context creation. As currently triton supports one
  // context per engine, in order to set the specified profile_index,
  // another context is created and the previous context is destroyed.
  std::shared_ptr<nvinfer1::IExecutionContext> default_trt_context(
      engine_->createExecutionContext());
  if (default_trt_context == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to create TensorRT context: ") +
         model_state_->GetTensorRTLogger().LastErrorMsg())
            .c_str());
  }

  num_expected_bindings_ = total_bindings_ / total_profiles;

  std::vector<std::pair<std::string, int>> profile_name_index;
  // No optimization profile is set for this TensorRT plan
  if (ProfileNames().empty()) {
    profile_name_index.emplace_back("default", 0);
  } else {
    for (const auto& profile_name : ProfileNames()) {
      int profile_index = 0;
      RETURN_IF_ERROR(GetProfileIndex(profile_name, &profile_index));
      profile_name_index.emplace_back(profile_name, profile_index);
    }
  }

  // Create one TRT context for each specified profile
  for (const auto& name_index : profile_name_index) {
    const auto& profile_name = name_index.first;
    const int profile_index = name_index.second;
    auto res = trt_contexts_.emplace(
        profile_index, TensorRTContext(
                           profile_name, profile_index, num_expected_bindings_,
                           EVENT_SET_COUNT));
    if (!res.second) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (profile_name + " maps to profile index " +
           std::to_string(profile_index) + " which has been mapped by " +
           res.first->second.profile_name_ +
           ", existing optimization profile will be reused")
              .c_str());
      continue;
    }
    if (profile_index == 0) {
      res.first->second.context_ = std::move(default_trt_context);
    } else {
      res.first->second.context_.reset(engine_->createExecutionContext());
      if (res.first->second.context_ == nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to create TensorRT context: ") +
             model_state_->GetTensorRTLogger().LastErrorMsg())
                .c_str());
      }
      if (!res.first->second.context_->setOptimizationProfileAsync(
              profile_index, stream_)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("Can not set the specified optimization "
                         "profile ") +
             profile_name + "[" + std::to_string(profile_index) + "] for " +
             name_ + ". Expected optimization profile index range 0-" +
             std::to_string(engine_->getNbOptimizationProfiles() - 1))
                .c_str());
      }
      cudaStreamSynchronize(CudaStream());
    }

    // Store the profile dimensions for later initializing the input bindings
    for (int io_index = 0; io_index < num_expected_bindings_; io_index++) {
      const auto binding_index =
          profile_index * num_expected_bindings_ + io_index;
      if (engine_->bindingIsInput(binding_index)) {
        RETURN_IF_ERROR(
            GetProfileDimensions(io_index, profile_index, &res.first->second));
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ValidateIO()
{
  // Collect all the expected input and allowed output tensor names
  // and validate that the model configuration specifies only those.
  std::set<std::string> allowed_inputs, allowed_outputs, allowed_shape_tensors;
  for (int i = 0; i < num_expected_bindings_; ++i) {
    if (engine_->bindingIsInput(i)) {
      allowed_inputs.emplace(engine_->getBindingName(i));
    } else {
      allowed_outputs.emplace(engine_->getBindingName(i));
    }
    if (engine_->isExecutionBinding(i)) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Detected ") + engine_->getBindingName(i) +
           " as execution binding for " + Name())
              .c_str());
    }
    if (engine_->isShapeBinding(i)) {
      allowed_shape_tensors.emplace(engine_->getBindingName(i));
      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Detected ") + engine_->getBindingName(i) +
           " as shape binding for " + Name())
              .c_str());
    }
  }

  triton::common::TritonJson::Value config_inputs;
  RETURN_IF_ERROR(
      model_state_->ModelConfig().MemberAsArray("input", &config_inputs));
  if (allowed_inputs.size() < config_inputs.ArraySize()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unable to load model '" + model_state_->Name() +
            "', configuration expects " +
            std::to_string(config_inputs.ArraySize()) +
            " inputs, model provides at most " +
            std::to_string(allowed_inputs.size()))
            .c_str());
  }

  for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
    RETURN_IF_ERROR(CheckAllowedModelInput(io, allowed_inputs));
  }

  triton::common::TritonJson::Value config_outputs;
  RETURN_IF_ERROR(
      model_state_->ModelConfig().MemberAsArray("output", &config_outputs));
  for (size_t i = 0; i < config_outputs.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_outputs.IndexAsObject(i, &io));
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, allowed_outputs));
  }

  RETURN_IF_ERROR(ValidateIOHelper(
      config_inputs, allowed_shape_tensors, true /* is_input */));
  RETURN_IF_ERROR(ValidateIOHelper(
      config_outputs, allowed_shape_tensors, false /* is_input */));

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ValidateIOHelper(
    common::TritonJson::Value& ios,
    const std::set<std::string>& allowed_shape_tensors, const bool is_input)
{
  std::string type = is_input ? "input" : "output";
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

    std::string io_data_type;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_data_type));
    if (!ConvertDataTypeToTrtType(
             ModelConfigDataTypeToTritonServerDataType(io_data_type))
             .first) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype") + io_data_type + " for " + type +
           " '" + io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    }

    // Check the shape tensor specification
    if (allowed_shape_tensors.find(io_name) != allowed_shape_tensors.end()) {
      bool is_shape_tensor = false;
      RETURN_IF_ERROR(io.MemberAsBool("is_shape_tensor", &is_shape_tensor));
      if (!is_shape_tensor) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (type + " '" + io_name + "' for model '" + model_state_->Name() +
             "' is a shape tensor but the model configuration "
             "doesn't mark "
             "it as a shape tensor.")
                .c_str());
      }
    } else {
      bool is_shape_tensor = false;
      RETURN_IF_ERROR(io.MemberAsBool("is_shape_tensor", &is_shape_tensor));
      if (is_shape_tensor) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (type + " '" + io_name + "' for model '" + model_state_->Name() +
             "' is incorrectly marked as a shape tensor in the model "
             "configuration.")
                .c_str());
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitIOBindingBuffers()
{
  triton::common::TritonJson::Value config_inputs;
  RETURN_IF_ERROR(
      model_state_->ModelConfig().MemberAsArray("input", &config_inputs));
  triton::common::TritonJson::Value config_outputs;
  RETURN_IF_ERROR(
      model_state_->ModelConfig().MemberAsArray("output", &config_outputs));

  // Initialize the inputs and outputs. Make sure the model matches
  // what is in the configuration. Allocate memory for the maximum
  // possible batch size: min(engine maximum, config maximum)
  io_binding_infos_.push_back(
      std::vector<IOBindingInfo>(num_expected_bindings_));
  buffer_bindings_.push_back(std::vector<void*>(total_bindings_, nullptr));

  // Use an additional set of buffers if a separate stream is used for
  // output
  if (model_state_->SeparateOutputStream()) {
    io_binding_infos_.push_back(
        std::vector<IOBindingInfo>(num_expected_bindings_));
    buffer_bindings_.push_back(std::vector<void*>(total_bindings_, nullptr));
  }

  // Sequence State should be processed at the end.
  for (int s = 0; s < num_copy_streams_; s++) {
    next_buffer_binding_set_ = s;
    RETURN_IF_ERROR(InitializeConfigShapeInputBindings(config_inputs));
    RETURN_IF_ERROR(InitializeConfigExecuteInputBindings(config_inputs));
    RETURN_IF_ERROR(
        InitializeSequenceControlInputBindings(model_state_->ModelConfig()));
    RETURN_IF_ERROR(InitializeBatchInputBindings(model_state_->ModelConfig()));
    RETURN_IF_ERROR(
        InitializeSequenceStateInputBindings(model_state_->ModelConfig()));
  }

  for (const auto& trt_context : trt_contexts_) {
    if (!trt_context.second.context_->allInputDimensionsSpecified()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "failed to specify the dimensions of all input bindings");
    }
    if (!trt_context.second.context_->allInputShapesSpecified()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "failed to specify the values of all input shape tensors");
    }
  }

  // Validate the batch dimension against the implicit batch dimension
  // if available.
  if (engine_->hasImplicitBatchDimension() &&
      (model_state_->MaxBatchSize() > engine_->getMaxBatchSize())) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unexpected configuration maximum batch size ") +
         std::to_string(model_state_->MaxBatchSize()) + " for '" + Name() +
         "', model maximum is " + std::to_string(engine_->getMaxBatchSize()))
            .c_str());
  }

  // Batch output must be processed before other outputs and sequence state
  // should be processed at the end.
  for (int s = 0; s < num_copy_streams_; s++) {
    next_buffer_binding_set_ = s;
    RETURN_IF_ERROR(InitializeBatchOutputBindings(model_state_->ModelConfig()));
    RETURN_IF_ERROR(InitializeConfigShapeOutputBindings(config_outputs));
    RETURN_IF_ERROR(InitializeConfigExecuteOutputBindings(config_outputs));
    RETURN_IF_ERROR(
        InitializeSequenceStateOutputBindings(model_state_->ModelConfig()));
  }
  next_buffer_binding_set_ = 0;
  // Make sure every index which corresponds to an execution binding
  // is initialized.
  for (int s = 0; s < num_copy_streams_; ++s) {
    for (int i = 0; i < num_expected_bindings_; ++i) {
      if (!io_binding_infos_[s][i].IsBufferAllocated() &&
          engine_->isExecutionBinding(i)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("expected configuration for ") +
             std::string((engine_->bindingIsInput(i) ? "input" : "output")) +
             " '" + engine_->getBindingName(i) + "' for " + Name())
                .c_str());
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeConfigShapeInputBindings(
    common::TritonJson::Value& config_inputs)
{
  for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_data_type;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_data_type));
    common::TritonJson::Value model_config_dims;
    common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(reshape.MemberAsArray("shape", &model_config_dims));
    } else {
      RETURN_IF_ERROR(io.MemberAsArray("dims", &model_config_dims));
    }

    RETURN_IF_ERROR(InitializeShapeInputBinding(
        io_name, ModelConfigDataTypeToTritonServerDataType(io_data_type),
        model_config_dims));
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeConfigExecuteInputBindings(
    common::TritonJson::Value& config_inputs)
{
  for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_datatype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_datatype));
    common::TritonJson::Value model_config_dims;
    common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(reshape.MemberAsArray("shape", &model_config_dims));
    } else {
      RETURN_IF_ERROR(io.MemberAsArray("dims", &model_config_dims));
    }
    bool io_allow_ragged_batch = false;
    triton::common::TritonJson::Value allow_ragged_batch;
    if (io.Find("allow_ragged_batch", &allow_ragged_batch)) {
      RETURN_IF_ERROR(allow_ragged_batch.AsBool(&io_allow_ragged_batch));
    }

    RETURN_IF_ERROR(InitializeExecuteInputBinding(
        io_name, io_datatype, model_config_dims, false, io_allow_ragged_batch));
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeSequenceControlInputBindings(
    common::TritonJson::Value& config)
{
  common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    std::vector<std::string> boolean_kinds{
        "CONTROL_SEQUENCE_START", "CONTROL_SEQUENCE_END",
        "CONTROL_SEQUENCE_READY"};

    for (const auto& control_kind : boolean_kinds) {
      const bool required = false;

      std::string tensor_name;
      std::string tensor_datatype;
      RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
          sequence_batching, model_state_->Name(), control_kind, required,
          &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr,
          nullptr, nullptr));
      if (!tensor_name.empty()) {
        // Control tensors must have shape [1].
        common::TritonJson::Value dims{
            triton::common::TritonJson::ValueType::ARRAY};
        RETURN_IF_ERROR(dims.AppendInt(1));

        RETURN_IF_ERROR(InitializeExecuteInputBinding(
            tensor_name, tensor_datatype, dims, true));
      }
    }

    std::vector<std::string> typdef_kinds{"CONTROL_SEQUENCE_CORRID"};

    for (const auto& control_kind : typdef_kinds) {
      const bool required = false;

      std::string tensor_name;
      std::string tensor_datatype;
      RETURN_IF_ERROR(GetTypedSequenceControlProperties(
          sequence_batching, model_state_->Name(), control_kind, required,
          &tensor_name, &tensor_datatype));
      if (!tensor_name.empty()) {
        // Control tensors must have shape [1].
        common::TritonJson::Value dims{
            triton::common::TritonJson::ValueType::ARRAY};
        RETURN_IF_ERROR(dims.AppendInt(1));

        RETURN_IF_ERROR(InitializeExecuteInputBinding(
            tensor_name, tensor_datatype, dims, true));
      }
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeSequenceStateInputBindings(
    common::TritonJson::Value& config)
{
  common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value io;
        RETURN_IF_ERROR(states.IndexAsObject(i, &io));
        std::string input_name;
        RETURN_IF_ERROR(io.MemberAsString("input_name", &input_name));
        std::string io_datatype;
        RETURN_IF_ERROR(io.MemberAsString("data_type", &io_datatype));

        common::TritonJson::Value model_config_dims;
        RETURN_IF_ERROR(io.MemberAsArray("dims", &model_config_dims));

        RETURN_IF_ERROR(InitializeExecuteInputBinding(
            input_name, io_datatype, model_config_dims, false /* is_control */,
            false /* is_ragged */, true /* is_state */));
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeSequenceStateOutputBindings(
    common::TritonJson::Value& config)
{
  common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        uses_implicit_state_ = true;
        triton::common::TritonJson::Value io;
        RETURN_IF_ERROR(states.IndexAsObject(i, &io));
        std::string output_name;
        RETURN_IF_ERROR(io.MemberAsString("output_name", &output_name));
        std::string io_datatype;
        RETURN_IF_ERROR(io.MemberAsString("data_type", &io_datatype));

        common::TritonJson::Value model_config_dims;
        RETURN_IF_ERROR(io.MemberAsArray("dims", &model_config_dims));

        RETURN_IF_ERROR(InitializeExecuteOutputBinding(
            output_name, io_datatype, model_config_dims, true /* is_state */));
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeBatchInputBindings(
    common::TritonJson::Value& config)
{
  std::vector<BatchInput> batch_inputs;
  BatchInput::ParseFromModelConfig(config, &batch_inputs);
  for (const auto& batch_input : batch_inputs) {
    for (const auto& tensor_name : batch_input.TargetNames()) {
      TRITONSERVER_DataType tensor_datatype = batch_input.DataType();
      common::TritonJson::Value dims{
          triton::common::TritonJson::ValueType::ARRAY};
      // Different batch input expects different shape, note that
      // we are setting config input shape here so the batch dimension
      // is not included
      if (model_state_->MaxBatchSize() == 0) {
        // If the model doesn't support batching, the range of some
        // batch input kind is convergent to a fixed value, need to
        // specify the fixed value in such case.
        // Note that batch input is intended to be used with batching model,
        // the following is more for completeness.
        switch (batch_input.BatchInputKind()) {
          case BatchInput::Kind::BATCH_ELEMENT_COUNT:
          case BatchInput::Kind::BATCH_ACCUMULATED_ELEMENT_COUNT:
            RETURN_IF_ERROR(dims.AppendInt(1));
            break;
          case BatchInput::Kind::BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO:
            RETURN_IF_ERROR(dims.AppendInt(2));
            break;
          case BatchInput::Kind::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE:
            RETURN_IF_ERROR(dims.AppendInt(-1));
            break;
          case BatchInput::Kind::BATCH_ITEM_SHAPE:
          case BatchInput::Kind::BATCH_ITEM_SHAPE_FLATTEN: {
            // Compiler doesn't like switch case fall through,
            // add conditional handling
            if (batch_input.BatchInputKind() ==
                BatchInput::Kind::BATCH_ITEM_SHAPE) {
              RETURN_IF_ERROR(dims.AppendInt(1));
            }
            triton::common::TritonJson::Value inputs;
            RETURN_IF_ERROR(config.MemberAsArray("input", &inputs));
            for (size_t i = 0; i < inputs.ArraySize(); ++i) {
              triton::common::TritonJson::Value input;
              RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));
              std::string input_name;
              RETURN_IF_ERROR(input.MemberAsString("name", &input_name));
              if (input_name == batch_input.SourceInputs()[0]) {
                // Get input config shape
                common::TritonJson::Value model_config_dims;
                common::TritonJson::Value reshape;
                if (input.Find("reshape", &reshape)) {
                  RETURN_IF_ERROR(
                      reshape.MemberAsArray("shape", &model_config_dims));
                } else {
                  RETURN_IF_ERROR(
                      input.MemberAsArray("dims", &model_config_dims));
                }
                if (model_config_dims.ArraySize() != 0) {
                  RETURN_IF_ERROR(
                      dims.AppendInt(model_config_dims.ArraySize()));
                }
                break;
              }
            }
            break;
          }
          default:
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string(
                     "batch input type '" + batch_input.BatchInputKindString() +
                     "' is not supported"))
                    .c_str());
        }
      } else {
        // For most type 'dims' will be empty as the full shape
        // of the batch input is [-1] which will be covered by
        // batch dimension.
        switch (batch_input.BatchInputKind()) {
          case BatchInput::Kind::BATCH_ELEMENT_COUNT:
          case BatchInput::Kind::BATCH_ACCUMULATED_ELEMENT_COUNT:
          case BatchInput::Kind::BATCH_ACCUMULATED_ELEMENT_COUNT_WITH_ZERO:
          case BatchInput::Kind::BATCH_MAX_ELEMENT_COUNT_AS_SHAPE:
          case BatchInput::Kind::BATCH_ITEM_SHAPE_FLATTEN:
            break;
          case BatchInput::Kind::BATCH_ITEM_SHAPE: {
            triton::common::TritonJson::Value inputs;
            RETURN_IF_ERROR(config.MemberAsArray("input", &inputs));
            for (size_t i = 0; i < inputs.ArraySize(); ++i) {
              triton::common::TritonJson::Value input;
              RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));
              std::string input_name;
              RETURN_IF_ERROR(input.MemberAsString("name", &input_name));
              if (input_name == batch_input.SourceInputs()[0]) {
                // Get input config shape
                common::TritonJson::Value model_config_dims;
                common::TritonJson::Value reshape;
                if (input.Find("reshape", &reshape)) {
                  RETURN_IF_ERROR(
                      reshape.MemberAsArray("shape", &model_config_dims));
                } else {
                  RETURN_IF_ERROR(
                      input.MemberAsArray("dims", &model_config_dims));
                }
                if (model_config_dims.ArraySize() != 0) {
                  RETURN_IF_ERROR(
                      dims.AppendInt(model_config_dims.ArraySize()));
                }
                break;
              }
            }
            break;
          }
          default:
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string(
                     "batch input type '" + batch_input.BatchInputKindString() +
                     "' is not supported"))
                    .c_str());
        }
      }
      int io_index = engine_->getBindingIndex(tensor_name.c_str());
      auto& io_binding_info =
          io_binding_infos_[next_buffer_binding_set_][io_index];
      io_binding_info.SetName(tensor_name);
      // Special handling hint for InitializeExecuteInputBinding()
      io_binding_info.SetBatchInput(batch_input);

      std::string data_type_str("TYPE_");
      data_type_str.append(TRITONSERVER_DataTypeString(tensor_datatype));
      RETURN_IF_ERROR(InitializeExecuteInputBinding(
          tensor_name, data_type_str, dims, false));


      BackendMemory* bm;
      if (io_binding_info.GetMemoryType() != TRITONSERVER_MEMORY_GPU) {
        // zero-copy is used so the input buffer is direct-writable
        BackendMemory::Create(
            model_state_->TritonMemoryManager(),
            BackendMemory::AllocationType::CPU_PINNED_POOL,
            0 /* memory_type_id */, io_binding_info.GetBuffer(),
            io_binding_info.GetByteSize(), &bm);
      } else {
        BackendMemory::Create(
            model_state_->TritonMemoryManager(),
            {BackendMemory::AllocationType::CPU_PINNED_POOL},
            0 /* memory_type_id */, io_binding_info.GetByteSize(), &bm);
      }
      io_binding_info.GetBatchInput()->second.reset(bm);
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeBatchOutputBindings(
    common::TritonJson::Value& config)
{
  std::vector<BatchOutput> batch_outputs;
  BatchOutput::ParseFromModelConfig(config, &batch_outputs);
  for (const auto& io : batch_outputs) {
    for (const auto& name : io.TargetNames()) {
      // FIXME Currently not handling the case that batch output is
      // shape tensor
      int io_index = engine_->getBindingIndex(name.c_str());
      auto& io_binding_info =
          io_binding_infos_[next_buffer_binding_set_][io_index];
      io_binding_info.SetName(name);
      if (engine_->isShapeBinding(io_index)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string(
                 "batch output '" + name + "' can not be shape binding"))
                .c_str());
      }

      // Whether the output needs to be scattered based on input
      if (io.BatchOutputKind() !=
          BatchOutput::Kind::BATCH_SCATTER_WITH_INPUT_SHAPE) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("batch output kind other than "
                         "BATCH_SCATTER_WITH_INPUT_SHAPE is not "
                         "supported for ") +
             name)
                .c_str());
      }
      // Set hints to for InitializeBatchOutputBindings()
      io_binding_info.SetIsBufferRagged(true);
      io_binding_info.SetIoShapeMapping(
          std::make_pair(io.SourceInputs()[0], std::vector<int64_t>()));
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeConfigShapeOutputBindings(
    common::TritonJson::Value& config_output)
{
  for (size_t i = 0; i < config_output.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_output.IndexAsObject(i, &io));

    // the maximum byte sizes across all profiles
    int64_t max_byte_size = 0;

    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

    bool is_shape_tensor = false;
    RETURN_IF_ERROR(io.MemberAsBool("is_shape_tensor", &is_shape_tensor));
    // Skip if this output is not a shape tensor
    if (!is_shape_tensor) {
      continue;
    }

    int io_index = engine_->getBindingIndex(io_name.c_str());
    auto& io_binding_info =
        io_binding_infos_[next_buffer_binding_set_][io_index];
    io_binding_info.SetName(io_name);
    for (auto& trt_context : trt_contexts_) {
      auto& profile_index = trt_context.first;
      auto& context = trt_context.second;
      int binding_index = num_expected_bindings_ * profile_index + io_index;
      if (binding_index < 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_NOT_FOUND,
            (std::string("output '") + io_name + "' not found for " + Name())
                .c_str());
      }

      if (io_binding_info.IsBufferAllocated()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("output '") + io_name +
             "'  has already appeared as an input or output for " + Name())
                .c_str());
      }

      if (engine_->bindingIsInput(binding_index)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("output '") + io_name +
             "' is expected to be an output in model for " + Name())
                .c_str());
      }

      std::string io_data_type;
      RETURN_IF_ERROR(io.MemberAsString("data_type", &io_data_type));

      if (io_data_type.compare("TYPE_INT32") != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unexpected datatype '") + io_data_type +
             " in model configuration for shape output '" + io_name +
             "', expecting TYPE_INT32 for " + Name())
                .c_str());
      }

      TRITONSERVER_DataType dt =
          ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
      TRITONSERVER_DataType config_dt =
          ModelConfigDataTypeToTritonServerDataType(io_data_type);
      if ((dt == TRITONSERVER_TYPE_INVALID) || (dt != config_dt)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unexpected datatype TYPE_") +
             TRITONSERVER_DataTypeString(dt) + " for inference output '" +
             io_name + "', expecting TYPE_" +
             TRITONSERVER_DataTypeString(config_dt) + " for " + Name())
                .c_str());
      }

      // [DLIS-4283] Note that the initialize-binding functions are
      // configuring the binding infos, may be group in separate class?
      RETURN_IF_ERROR(
          interface_->SetFormat(binding_index, &io_binding_info.GetFormat()));

      common::TritonJson::Value model_config_dims;
      common::TritonJson::Value reshape;
      if (io.Find("reshape", &reshape)) {
        RETURN_IF_ERROR(reshape.MemberAsArray("shape", &model_config_dims));
      } else {
        RETURN_IF_ERROR(io.MemberAsArray("dims", &model_config_dims));
      }

      nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
      if (ContainsWildcard(engine_dims)) {
        context.is_dynamic_per_binding_[io_index] = true;
        io_binding_info.SetIsDynamicShapeOutput(true);
      }

      RETURN_IF_ERROR(CompareShapeDimsSupported(
          Name(), io_name, engine_dims, model_config_dims, support_batching_));

      if (!io_binding_info.IsDynamicShapeOutput()) {
        const nvinfer1::Dims output_dim =
            context.context_->getBindingDimensions(binding_index);
        std::vector<int64_t> dim_vec;
        DimsToDimVec(output_dim, &dim_vec);
        int64_t byte_size = GetByteSize(dt, dim_vec);

        max_byte_size = std::max(max_byte_size, byte_size);
      }
    }

    if (!io_binding_info.IsDynamicShapeOutput()) {
      // [DLIS-4283] review below comment
      // Allocate CUDA memory. Use cudaHostAlloc if zero copy
      // supported. For static output tensors, we rely on
      // buffer_bindings_ being non-nullptr to indicate that the
      // buffer has been correctly initialized so even for zero-sized
      // tensors always allocate something.
      void* buffer = nullptr;
      cudaError_t err = cudaSuccess;
      // [DLIS-4283] by knowing that shape tensor should be on host, ideally
      // should perform allocation based on getTensorLocation(), in such way
      // the handling can be unified between shape tensor and execution tensor
      err = cudaHostAlloc(
          &buffer, std::max((int64_t)1, max_byte_size), cudaHostAllocDefault);
      if (err != cudaSuccess) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to allocate memory for output '") + io_name +
             "' for " + Name() + ": " + cudaGetErrorString(err))
                .c_str());
      }

      io_binding_info.SetByteSize(max_byte_size);
      io_binding_info.SetBuffer(buffer);
      io_binding_info.SetDeviceBuffer(buffer);
    } else {
      auto allocator = std::make_unique<OutputAllocator>(zero_copy_support_);
      io_binding_info.SetBuffer(std::move(allocator));
    }
    io_binding_info.SetMemoryType(TRITONSERVER_MEMORY_CPU_PINNED);
    io_binding_info.SetMemoryTypeId(0);

    // [DLIS-4283] below is v1 handling and v1 doesn't support shape tensor,
    // can be removed.
    // Set buffer bindings of all optimization profile since buffer
    // is allocated
    for (auto& trt_context : trt_contexts_) {
      auto binding_index =
          num_expected_bindings_ * trt_context.first + io_index;
      buffer_bindings_[next_buffer_binding_set_][binding_index] =
          io_binding_info.GetDeviceBuffer();
      if (!io_binding_info.IsDynamicShapeOutput()) {
        // [DLIS-4283] revisit below, note that 'device_buffer_' is actually
        // not on device for shape tensor, the name can be misleading, perhaps
        // something like 'trt_enqueue_buffer'
        RETURN_ERROR_IF_FALSE(
            trt_context.second.context_->setTensorAddress(
                io_name.c_str(), io_binding_info.GetDeviceBuffer()),
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("'") + io_name +
             "' does not map to an input or output tensor"));
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeConfigExecuteOutputBindings(
    common::TritonJson::Value& config_output)
{
  for (size_t i = 0; i < config_output.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_output.IndexAsObject(i, &io));

    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

    std::string io_datatype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_datatype));

    bool is_shape_tensor = false;
    RETURN_IF_ERROR(io.MemberAsBool("is_shape_tensor", &is_shape_tensor));

    common::TritonJson::Value model_config_dims;
    common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(reshape.MemberAsArray("shape", &model_config_dims));
    } else {
      RETURN_IF_ERROR(io.MemberAsArray("dims", &model_config_dims));
    }

    // Skip if the output is specified to be a shape tensor
    if (is_shape_tensor) {
      continue;
    }

    RETURN_IF_ERROR(InitializeExecuteOutputBinding(
        io_name, io_datatype, model_config_dims));
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeExecuteInputBinding(
    const std::string& input_name, const std::string& input_datatype,
    common::TritonJson::Value& input_dims, const bool is_control,
    const bool is_ragged, const bool is_state)
{
  // the maximum byte sizes across all profiles
  int64_t max_byte_size = 0;
  int io_index = engine_->getBindingIndex(input_name.c_str());
  auto& io_binding_info = io_binding_infos_[next_buffer_binding_set_][io_index];
  io_binding_info.SetName(input_name);

  if (io_binding_info.IsBufferAllocated() && is_state) {
    // The input bindings for the given state input is already allocated,
    // hence, no need to proceed further.
    return nullptr;
  }

  for (auto& trt_context : trt_contexts_) {
    auto& profile_index = trt_context.first;
    auto& context = trt_context.second;
    int binding_index = num_expected_bindings_ * profile_index + io_index;
    if (io_index < 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          (std::string("input '") + input_name + "' not found for " + Name())
              .c_str());
    }

    // Skip if shape binding is encountered
    if (engine_->isShapeBinding(binding_index)) {
      return nullptr;
    }

    if (io_binding_info.IsBufferAllocated()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("input '") + input_name +
           "'  has already appeared as an input or output for " + Name())
              .c_str());
    }


    if (!engine_->bindingIsInput(binding_index)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("input '") + input_name +
           "' is expected to be an input in model for " + Name())
              .c_str());
    }

    TRITONSERVER_DataType dt =
        ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
    TRITONSERVER_DataType config_dt =
        ModelConfigDataTypeToTritonServerDataType(input_datatype);
    if ((dt == TRITONSERVER_TYPE_INVALID) || (dt != config_dt)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unexpected datatype TYPE_") +
           TRITONSERVER_DataTypeString(dt) + " for inference input '" +
           input_name + "', expecting TYPE_" +
           TRITONSERVER_DataTypeString(config_dt) + " for " + Name())
              .c_str());
    }

    RETURN_IF_ERROR(
        interface_->SetFormat(binding_index, &io_binding_info.GetFormat()));

    // Detect whether dynamic or not
    nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
    if (ContainsWildcard(engine_dims)) {
      context.is_dynamic_per_binding_[io_index] = true;
    }

    if (!(is_control && context.is_dynamic_per_binding_[io_index])) {
      if (!is_ragged) {
        RETURN_IF_ERROR(CompareDimsSupported(
            StateForModel()->Name(), input_name, engine_dims, input_dims,
            support_batching_, (!engine_->hasImplicitBatchDimension()),
            false /* compare_exact */));
      } else {
        // For ragged input, the input will be concatenated and
        // flatten, so expecting engine dims to be one dimensional.
        int64_t input_dims_0;
        RETURN_IF_ERROR(input_dims.IndexAsInt(0, &input_dims_0));
        if ((engine_dims.nbDims != 1) || (engine_dims.d[0] != input_dims_0)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("model '") + model_state_->Name() + "', tensor '" +
               input_name +
               "': for the model to support ragged input, the engine "
               "shape"
               " is: " +
               DimsDebugString(engine_dims) +
               " while the model config shape is: " +
               DimsJsonToString(input_dims))
                  .c_str());
        }
      }
    } else {
      TRITONSERVER_Error* err =
          ValidateControlDimsDynamic(engine_dims, support_batching_);
      if (err != nullptr) {
        TRITONSERVER_Error* full_err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unexpected shape ") + DimsDebugString(engine_dims) +
             " for control input '" + input_name + "' for model " +
             model_state_->Name() + ": " + TRITONSERVER_ErrorMessage(err))
                .c_str());
        TRITONSERVER_ErrorDelete(err);
        return full_err;
      }
    }

    std::vector<int64_t> full_config_dim;
    if (is_ragged) {
      // if ragged, full shape is defined to be [-1]
      full_config_dim = {-1};
    } else if (io_binding_info.GetBatchInput() != nullptr) {
      // if batch input, "config shape" has been populated in 'input_dims',
      // but if batching is supported, batch dimension doesn't necessary
      // limited by config max batch size
      RETURN_IF_ERROR(DimsJsonToDimVec(input_dims, &full_config_dim));
      if (model_state_->MaxBatchSize() > 0) {
        full_config_dim.insert(full_config_dim.begin(), -1);
      }
    } else {
      RETURN_IF_ERROR(DimsJsonToDimVec(input_dims, &full_config_dim));
      if (model_state_->MaxBatchSize() > 0) {
        full_config_dim.insert(
            full_config_dim.begin(), model_state_->MaxBatchSize());
      }
    }

    std::vector<int64_t> maximum_dims;
    RETURN_IF_ERROR(interface_->ConfigureInputDimensions(
        &context, io_index, binding_index, full_config_dim, &maximum_dims));

    int64_t byte_size = 0;
    if (io_binding_info.GetFormat().is_linear_format_) {
      byte_size = GetByteSize(dt, maximum_dims);
    } else {
      maximum_dims[io_binding_info.GetFormat().vectorized_dim_] +=
          (io_binding_info.GetFormat().components_per_element_ -
           (maximum_dims[io_binding_info.GetFormat().vectorized_dim_] %
            io_binding_info.GetFormat().components_per_element_));
      byte_size = GetByteSize(dt, maximum_dims);
    }

    if (byte_size == -1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to calculate size for input '") + input_name +
           "' for " + Name())
              .c_str());
    }
    max_byte_size = std::max(max_byte_size, byte_size);
  }

  // Allocate CUDA memory. Use cudaHostAlloc if zero copy supported.
  // We rely on buffer_bindings_ being non-nullptr to indicate that
  // the buffer has been correctly initialized so even for zero-sized
  // tensors always allocate something.
  void* buffer = nullptr;
  cudaError_t err = cudaSuccess;
  if (zero_copy_support_) {
    err = cudaHostAlloc(
        &buffer, std::max((int64_t)1, max_byte_size), cudaHostAllocMapped);
  } else {
    err = cudaMalloc(&buffer, std::max((int64_t)1, max_byte_size));
  }
  if (err != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to allocate memory for input '") + input_name +
         "' for " + Name() + ": " + cudaGetErrorString(err))
            .c_str());
  }
  err = cudaMemset((uint8_t*)buffer, 0, max_byte_size);
  if (err != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to set memory for input '") + input_name +
         "' for " + Name() + ": " + cudaGetErrorString(err))
            .c_str());
  }

  io_binding_info.SetByteSize(max_byte_size);
  io_binding_info.SetBuffer(buffer);
  io_binding_info.SetDeviceBuffer(buffer);
  io_binding_info.SetIsBufferRagged(is_ragged);
  if (zero_copy_support_) {
    io_binding_info.SetMemoryType(TRITONSERVER_MEMORY_CPU_PINNED);
    io_binding_info.SetMemoryTypeId(0);
    err = cudaHostGetDevicePointer(
        io_binding_info.GetDeviceBufferAddr(), io_binding_info.GetBuffer(), 0);
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to get mapped device address for input '") +
           input_name + "' for " + Name() + ": " + cudaGetErrorString(err))
              .c_str());
    }
  } else {
    io_binding_info.SetMemoryType(TRITONSERVER_MEMORY_GPU);
    io_binding_info.SetMemoryTypeId(DeviceId());
  }
  if (io_binding_info.IsBufferRagged() &&
      !io_binding_info.GetFormat().is_linear_format_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unexpected allow-ragged for non-linear input '") +
         input_name + "' for " + Name())
            .c_str());
  }

  // Set buffer bindings of all optimization profile since buffer is
  // allocated
  for (auto& trt_context : trt_contexts_) {
    auto binding_index = num_expected_bindings_ * trt_context.first + io_index;
    buffer_bindings_[next_buffer_binding_set_][binding_index] =
        io_binding_info.GetDeviceBuffer();
    RETURN_ERROR_IF_FALSE(
        trt_context.second.context_->setTensorAddress(
            input_name.c_str(), io_binding_info.GetDeviceBuffer()),
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("'") + input_name +
         "' does not map to an input or output tensor"));
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeExecuteOutputBinding(
    const std::string& output_name, const std::string& output_datatype,
    common::TritonJson::Value& output_dims, bool is_state)
{
  // the maximum byte sizes across all profiles
  int64_t max_byte_size = 0;

  int io_index = engine_->getBindingIndex(output_name.c_str());

  auto& io_binding_info = io_binding_infos_[next_buffer_binding_set_][io_index];
  io_binding_info.SetName(output_name);

  // State output is initialized before the requested output tensor.
  if (is_state) {
    io_binding_info.SetIsStateOutput(true);
  } else {
    io_binding_info.SetIsRequestedOutputTensor(true);
  }

  if (io_binding_info.IsBufferAllocated() && is_state) {
    // The input bindings for the given state input is already allocated,
    // hence, no need to proceed further.
    return nullptr;
  }

  // Check whether the output shape is data-dependent.
  for (auto& trt_context : trt_contexts_) {
    if (ContainsWildcard(
            trt_context.second.context_->getTensorShape(output_name.c_str()))) {
      io_binding_info.SetIsDynamicShapeOutput(true);
      break;
    }
  }

  for (auto& trt_context : trt_contexts_) {
    auto& profile_index = trt_context.first;
    auto& context = trt_context.second;
    int binding_index = num_expected_bindings_ * profile_index + io_index;

    if (binding_index < 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          (std::string("output '") + output_name + "' not found for " + Name())
              .c_str());
    }

    if (engine_->bindingIsInput(binding_index)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("output '") + output_name +
           "' is expected to be an input in model for " + Name())
              .c_str());
    }

    if (io_binding_info.IsBufferAllocated()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("output '") + output_name +
           "'  has already appeared as an input or output for " + Name())
              .c_str());
    }

    nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
    if (ContainsWildcard(engine_dims)) {
      context.is_dynamic_per_binding_[io_index] = true;
    }

    TRITONSERVER_DataType dt =
        ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
    TRITONSERVER_DataType config_dt =
        ModelConfigDataTypeToTritonServerDataType(output_datatype);
    if ((dt == TRITONSERVER_TYPE_INVALID) || (dt != config_dt)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unexpected datatype TYPE_") +
           TRITONSERVER_DataTypeString(dt) + " for inference output '" +
           output_name + "', expecting TYPE_" +
           TRITONSERVER_DataTypeString(config_dt) + " for " + Name())
              .c_str());
    }

    RETURN_IF_ERROR(
        interface_->SetFormat(binding_index, &io_binding_info.GetFormat()));

    // Skip 'batch_output' validation as it is not exact match to
    // model dims
    if (!io_binding_info.IsBufferRagged()) {
      RETURN_IF_ERROR(CompareDimsSupported(
          StateForModel()->Name(), output_name, engine_dims, output_dims,
          support_batching_, (!engine_->hasImplicitBatchDimension()),
          false /* compare_exact */));
    }

    if (io_binding_info.IsBufferRagged() &&
        !io_binding_info.GetFormat().is_linear_format_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unexpected allow-ragged for non-linear output '") +
           output_name + "' for " + Name())
              .c_str());
    }
    if (!io_binding_info.IsDynamicShapeOutput()) {
      int64_t byte_size = interface_->GetFullByteSize(
          context.context_.get(), output_name, binding_index);
      if (byte_size == -1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to allocate memory for output '") +
             output_name + "' for " + Name())
                .c_str());
      }
      max_byte_size = std::max(max_byte_size, byte_size);
    }
  }

  cudaError_t err = cudaSuccess;

  if (!io_binding_info.IsDynamicShapeOutput()) {
    // Allocate CUDA memory. Use cudaHostAlloc if zero copy supported.
    // For static output tensors, we rely on buffer_bindings_ being
    // non-nullptr to indicate that the buffer has been correctly initialized
    // so even for zero-sized tensors always allocate something.
    void* buffer = nullptr;
    if (zero_copy_support_) {
      err = cudaHostAlloc(
          &buffer, std::max((int64_t)1, max_byte_size), cudaHostAllocMapped);
    } else {
      err = cudaMalloc(&buffer, std::max((int64_t)1, max_byte_size));
    }

    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to allocate memory for output '") + output_name +
           "' for " + Name() + ": " + cudaGetErrorString(err))
              .c_str());
    }
    io_binding_info.SetByteSize(max_byte_size);
    io_binding_info.SetBuffer(buffer);
    io_binding_info.SetDeviceBuffer(buffer);
  } else {
    auto allocator = std::make_unique<OutputAllocator>(zero_copy_support_);
    for (auto& trt_context : trt_contexts_) {
      trt_context.second.context_->setOutputAllocator(
          output_name.c_str(), allocator.get());
    }
    io_binding_info.SetBuffer(std::move(allocator));
  }

  // Whether the output needs to be scattered based on input
  if (io_binding_info.IsBufferRagged()) {
    std::vector<int64_t> output_shape;
    if (support_batching_) {
      output_shape.push_back(-1);
    }
    for (size_t i = 0; i < output_dims.ArraySize(); i++) {
      int64_t dim;
      RETURN_IF_ERROR(output_dims.IndexAsInt(i, &dim));
      output_shape.push_back(dim);
    }
    io_binding_info.GetIoShapeMapping().second = output_shape;
  }
  if (zero_copy_support_) {
    io_binding_info.SetMemoryType(TRITONSERVER_MEMORY_CPU_PINNED);
    io_binding_info.SetMemoryTypeId(0);
    if (!io_binding_info.IsDynamicShapeOutput()) {
      err = cudaHostGetDevicePointer(
          io_binding_info.GetDeviceBufferAddr(), io_binding_info.GetBuffer(),
          0);
      if (err != cudaSuccess) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to get mapped device address for output '") +
             output_name + " for " + Name() + ": " + cudaGetErrorString(err))
                .c_str());
      }
    }
  } else {
    io_binding_info.SetMemoryType(TRITONSERVER_MEMORY_GPU);
    io_binding_info.SetMemoryTypeId(DeviceId());
  }

  // Set buffer bindings of all optimization profile since buffer is
  // allocated
  for (auto& trt_context : trt_contexts_) {
    auto binding_index = num_expected_bindings_ * trt_context.first + io_index;
    buffer_bindings_[next_buffer_binding_set_][binding_index] =
        io_binding_info.GetDeviceBuffer();
    if (!io_binding_info.IsDynamicShapeOutput()) {
      RETURN_ERROR_IF_FALSE(
          trt_context.second.context_->setTensorAddress(
              output_name.c_str(), io_binding_info.GetDeviceBuffer()),
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("'") + output_name +
           "' does not map to an input or output tensor"));
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeShapeInputBinding(
    const std::string& input_name, const TRITONSERVER_DataType input_datatype,
    common::TritonJson::Value& model_config_dims)
{
  // the maximum byte sizes across all profiles
  int64_t max_byte_size = 0;
  int io_index = engine_->getBindingIndex(input_name.c_str());

  auto& io_binding_info = io_binding_infos_[next_buffer_binding_set_][io_index];
  io_binding_info.SetName(input_name);
  for (auto& trt_context : trt_contexts_) {
    auto& profile_index = trt_context.first;
    auto& context = trt_context.second;
    int binding_index = num_expected_bindings_ * profile_index + io_index;
    if (io_index < 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          (std::string("input '") + input_name + "' not found for " + Name())
              .c_str());
    }

    if (io_binding_info.IsBufferAllocated()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("input '") + input_name +
           "'  has already appeared as an input or output for " + Name())
              .c_str());
    }

    if (!engine_->bindingIsInput(binding_index)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("input '") + input_name +
           "' is expected to be an input in model for " + Name())
              .c_str());
    }

    // Skip if the binding is not a shape tensor
    if (!engine_->isShapeBinding(binding_index)) {
      return nullptr;
    }

    if (input_datatype != TRITONSERVER_TYPE_INT32) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unexpected datatype TYPE_") +
           TRITONSERVER_DataTypeString(input_datatype) +
           "  in model configuration for shape input '" + input_name +
           "', expecting TYPE_INT32 for " + Name())
              .c_str());
    }

    TRITONSERVER_DataType dt =
        ConvertTrtTypeToDataType(engine_->getBindingDataType(binding_index));
    if ((dt == TRITONSERVER_TYPE_INVALID) || (dt != input_datatype)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unexpected datatype TYPE_") +
           TRITONSERVER_DataTypeString(dt) + " in engine for shape input '" +
           input_name + "', expecting TYPE_" +
           TRITONSERVER_DataTypeString(input_datatype) + " for " + Name())
              .c_str());
    }

    RETURN_IF_ERROR(
        interface_->SetFormat(binding_index, &io_binding_info.GetFormat()));

    nvinfer1::Dims engine_dims = engine_->getBindingDimensions(binding_index);
    if (ContainsWildcard(engine_dims)) {
      context.is_dynamic_per_binding_[io_index] = true;
    }

    RETURN_IF_ERROR(CompareShapeDimsSupported(
        name_, input_name, engine_dims, model_config_dims, support_batching_));

    if (!context.context_->setBindingDimensions(
            binding_index, context.max_dims_[io_index])) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("trt failed to set binding dimension to ") +
           DimsDebugString(context.max_dims_[io_index]) + " for input '" +
           input_name + "' for " + Name())
              .c_str());
    }

    context.nb_shape_values_ = (context.max_dims_[io_index].nbDims == 0)
                                   ? 1
                                   : context.max_dims_[io_index].d[0];
    context.max_shapes_[io_index] = engine_->getProfileShapeValues(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kMAX);
    context.min_shapes_[io_index] = engine_->getProfileShapeValues(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
    context.opt_shapes_[io_index] = engine_->getProfileShapeValues(
        binding_index, profile_index, nvinfer1::OptProfileSelector::kOPT);

    // Set shape tensor address to buffer that contains max allowed value so
    // later shape inference will return max output shape / size for
    // pre-allocation.
    if (!context.context_->setInputTensorAddress(
            input_name.c_str(), context.max_shapes_[io_index])) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("trt failed to set the input shape binding for '") +
           input_name + "' for " + Name())
              .c_str());
    }
    // [FIXME] setInputShapeBinding() is deprecated and
    // setInputTensorAddress() above alone should be sufficient. There is bug
    // in TRT that getMaxOutputSize() won't work without
    // setInputShapeBinding() for shape tensor, ETA fix in 8.5.2
    if (!context.context_->setInputShapeBinding(
            binding_index, context.max_shapes_[io_index])) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("trt failed to set the input shape binding for '") +
           input_name + "' for " + Name())
              .c_str());
    }

    if (engine_->isExecutionBinding(binding_index)) {
      int64_t byte_size = 0;
      if (io_binding_info.GetFormat().is_linear_format_) {
        std::vector<int64_t> dim_vec;
        DimsToDimVec(
            context.context_->getBindingDimensions(binding_index), &dim_vec);
        byte_size = GetByteSize(dt, dim_vec);
      } else {
        auto component_count =
            GetElementCount(context.context_->getStrides(binding_index));
        component_count *=
            engine_->getBindingComponentsPerElement(binding_index);
        byte_size = component_count *
                    engine_->getBindingBytesPerComponent(binding_index);
      }
      max_byte_size = std::max(max_byte_size, byte_size);
    }
  }

  if (max_byte_size != 0) {
    // Allocate CUDA memory. Use cudaHostAlloc if zero copy supported.
    // We rely on buffer_bindings_ being non-nullptr to indicate that
    // the buffer has been correctly initialized so even for zero-sized
    // tensors always allocate something.
    void* buffer = nullptr;
    cudaError_t err = cudaSuccess;
    // [DLIS-4283] by knowing that shape tensor should be on host, ideally
    // should perform allocation based on getTensorLocation(), in such way
    // the handling can be unified between shape tensor and execution tensor
    auto cuda_flag =
        zero_copy_support_ ? cudaHostAllocMapped : cudaHostAllocDefault;
    err =
        cudaHostAlloc(&buffer, std::max((int64_t)1, max_byte_size), cuda_flag);
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to allocate memory for input '") + input_name +
           "' for " + Name() + ": " + cudaGetErrorString(err))
              .c_str());
    }

    io_binding_info.SetByteSize(max_byte_size);
    io_binding_info.SetBuffer(buffer);
    io_binding_info.SetDeviceBuffer(buffer);
    io_binding_info.SetMemoryType(TRITONSERVER_MEMORY_CPU_PINNED);
    io_binding_info.SetMemoryTypeId(0);

    // Different from other tensors that setTensorAdress() will be set to the
    // allocated buffer at the end, we keep it to be the 'max_shapes_' set
    // above because the buffer content will affect the output shape
    // inference. We rely on Enqueue() function to make sure the correct
    // buffer is set during runtime.

    // [DLIS-4283] below is v1 handling and v1 doesn't support shape tensor,
    // can be removed.
    // Set buffer bindings of all optimization profile since buffer is
    // allocated
    for (auto& trt_context : trt_contexts_) {
      auto binding_index =
          num_expected_bindings_ * trt_context.first + io_index;
      buffer_bindings_[next_buffer_binding_set_][binding_index] =
          io_binding_info.GetDeviceBuffer();
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::GetProfileDimensions(
    const int io_index, const int profile_index, TensorRTContext* context)
{
  int binding_index = (profile_index * num_expected_bindings_) + io_index;
  context->max_dims_[io_index] = engine_->getProfileDimensions(
      binding_index, profile_index, nvinfer1::OptProfileSelector::kMAX);
  context->min_dims_[io_index] = engine_->getProfileDimensions(
      binding_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
  context->opt_dims_[io_index] = engine_->getProfileDimensions(
      binding_index, profile_index, nvinfer1::OptProfileSelector::kOPT);
  return nullptr;
}

void
ModelInstanceState::GetConfiguredProfiles(std::string* profiles_desc)
{
  profiles_desc->clear();
  for (const auto& trt_context : trt_contexts_) {
    (*profiles_desc) +=
        (" " + trt_context.second.profile_name_ + "[" +
         std::to_string(trt_context.first) + "];");
  }
}

void
ModelInstanceState::FindClosestCudaGraph(
    const TensorRTContext& trt_context,
    const std::vector<int64_t>& cuda_graph_key,
    const TensorRTContext::CudaGraph** cuda_graph, bool* found_exact)
{
  *cuda_graph = nullptr;
  auto itr =
      trt_context.cuda_graph_execs_[next_set_].lower_bound(cuda_graph_key);
  if (itr != trt_context.cuda_graph_execs_[next_set_].end()) {
    *found_exact = (itr->first == cuda_graph_key);
    if (*found_exact) {
      *cuda_graph = &itr->second;
      return;
    } else if (allow_inexact_match_) {
      // For vector as key, returned lower bound may not satisfy
      // requirements that all dims must be >= actual dims
      for (; itr != trt_context.cuda_graph_execs_[next_set_].end(); itr++) {
        bool found = true;
        for (size_t key_idx = 0; key_idx < cuda_graph_key.size(); key_idx++) {
          if ((cuda_graph_key[key_idx] > itr->first[key_idx]) ||
              (cuda_graph_key[key_idx] <
               itr->second.lower_bound_key_[key_idx])) {
            found = false;
            break;
          }
        }
        if (found) {
          *cuda_graph = &itr->second;
          return;
        }
      }
    }
  }
  return;
}

// CUDA 10.1 starts to support CUDA graphs.
#ifdef TRITON_ENABLE_CUDA_GRAPH
TRITONSERVER_Error*
ModelInstanceState::InitializeCudaGraph()
{
  std::vector<GraphSpec> graph_specs;
  RETURN_IF_ERROR(InitializeGraphSpecs(&graph_specs, &allow_inexact_match_));

  // CUDA graph will be captured for every TRT contexts as CUDA graph
  // is merely capturing GPU activities for a given execution.
  for (auto& graph_spec : graph_specs) {
    for (auto& trt_context : trt_contexts_) {
      graph_spec.captured_ =
          interface_->BuildCudaGraph(&(trt_context.second), graph_spec);
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeGraphSpecs(
    std::vector<GraphSpec>* graph_specs, bool* allow_inexact_match)
{
  *allow_inexact_match = false;
  graph_specs->clear();

  if (model_state_->GraphSpecs().ArraySize() == 0) {
    // No graph spec is provided, use default specs
    // Graphs are most likely to help for small batch sizes so by
    // default build for batch sizes 1, 2, 3, 4, 6, 8, 12, 16,
    // 'max_batch_size'. If preferred batch size is specified, then
    // the batch sizes will be 1, preferred batch sizes,
    // 'max_batch_size'.
    std::set<int> cuda_graph_batch_sizes;
    if (model_state_->MaxBatchSize() == 0) {
      cuda_graph_batch_sizes = {0};
    } else {
      cuda_graph_batch_sizes = {1};
    }

    common::TritonJson::Value dynamic_batching;
    common::TritonJson::Value sequence_batching;
    common::TritonJson::Value oldest;
    if (model_state_->ModelConfig().Find(
            "dynamic_batching", &dynamic_batching)) {
      common::TritonJson::Value pbs;
      RETURN_IF_ERROR(
          dynamic_batching.MemberAsArray("preferred_batch_size", &pbs));
      for (size_t i = 0; i < pbs.ArraySize(); i++) {
        int64_t bs;
        RETURN_IF_ERROR(pbs.IndexAsInt(i, &bs));
        cuda_graph_batch_sizes.emplace(bs);
      }
    } else if (
        model_state_->ModelConfig().Find(
            "sequence_batching", &sequence_batching) &&
        sequence_batching.Find("oldest", &oldest)) {
      common::TritonJson::Value pbs;
      RETURN_IF_ERROR(oldest.MemberAsArray("preferred_batch_size", &pbs));
      for (size_t i = 0; i < pbs.ArraySize(); i++) {
        int64_t bs;
        RETURN_IF_ERROR(pbs.IndexAsInt(i, &bs));
        cuda_graph_batch_sizes.emplace(bs);
      }
    } else {
      cuda_graph_batch_sizes = {1, 2, 3, 4, 6, 8, 12, 16};
      if (model_state_->MaxBatchSize() == 0) {
        cuda_graph_batch_sizes.emplace(0);
      }
    }
    if (model_state_->MaxBatchSize() > 0) {
      cuda_graph_batch_sizes.emplace(model_state_->MaxBatchSize());
    }

    for (const auto bs : cuda_graph_batch_sizes) {
      if (bs <= model_state_->MaxBatchSize()) {
        graph_specs->emplace_back();
        graph_specs->back().batch_size_ = bs;
        graph_specs->back().lower_bound_batch_size_ = bs;
      }
    }
  } else {
    for (size_t i = 0; i < model_state_->GraphSpecs().ArraySize(); i++) {
      common::TritonJson::Value config_spec;
      RETURN_IF_ERROR(
          model_state_->GraphSpecs().IndexAsObject(i, &config_spec));
      graph_specs->emplace_back();
      auto& graph_spec = graph_specs->back();
      RETURN_IF_ERROR(
          config_spec.MemberAsInt("batch_size", &graph_spec.batch_size_));
      common::TritonJson::Value inputs;
      if (config_spec.Find("input", &inputs)) {
        std::vector<std::string> input_names;
        RETURN_IF_ERROR(inputs.Members(&input_names));
        for (const auto& input_name : input_names) {
          common::TritonJson::Value input;
          RETURN_IF_ERROR(inputs.MemberAsObject(input_name.c_str(), &input));
          std::vector<int64_t> input_shape;
          common::TritonJson::Value dims;
          RETURN_IF_ERROR(input.MemberAsArray("dim", &dims));
          for (size_t i = 0; i < dims.ArraySize(); i++) {
            std::string dim;
            RETURN_IF_ERROR(dims.IndexAsString(i, &dim));
            input_shape.push_back(std::stoi(dim));
          }
          graph_spec.shapes_[input_name] = std::move(input_shape);
        }
      }

      common::TritonJson::Value lower_bound_spec;
      if (config_spec.Find("graph_lower_bound", &lower_bound_spec)) {
        *allow_inexact_match = true;
        RETURN_IF_ERROR(lower_bound_spec.MemberAsInt(
            "batch_size", &graph_spec.lower_bound_batch_size_));
        common::TritonJson::Value inputs;
        if (lower_bound_spec.Find("input", &inputs)) {
          std::vector<std::string> input_names;
          RETURN_IF_ERROR(inputs.Members(&input_names));
          for (const auto& input_name : input_names) {
            common::TritonJson::Value input;
            RETURN_IF_ERROR(inputs.MemberAsObject(input_name.c_str(), &input));
            std::vector<int64_t> input_shape;
            common::TritonJson::Value dims;
            RETURN_IF_ERROR(input.MemberAsArray("dim", &dims));
            for (size_t i = 0; i < dims.ArraySize(); i++) {
              std::string dim;
              RETURN_IF_ERROR(dims.IndexAsString(i, &dim));
              input_shape.push_back(std::stoi(dim));
            }
            graph_spec.lower_bound_shapes_[input_name] = std::move(input_shape);
          }
        }
      } else {
        graph_spec.lower_bound_batch_size_ = graph_spec.batch_size_;
        graph_spec.lower_bound_shapes_ = graph_spec.shapes_;
      }
    }
  }
  for (const auto& graph_spec : *graph_specs) {
    RETURN_IF_ERROR(ValidateGraphSpec(graph_spec));
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ValidateGraphSpec(const GraphSpec& graph_spec)
{
  if (model_state_->MaxBatchSize() == 0) {
    if ((graph_spec.batch_size_ != 0) ||
        (graph_spec.lower_bound_batch_size_ != 0)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "graph spec expects 'batch_size' to be 0 if "
          "'max_batch_size' is 0");
    }
  } else if (
      ((graph_spec.batch_size_ > model_state_->MaxBatchSize()) ||
       (graph_spec.batch_size_ < 1)) ||
      ((graph_spec.lower_bound_batch_size_ > model_state_->MaxBatchSize()) ||
       (graph_spec.lower_bound_batch_size_ < 1))) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("graph spec expects 'batch_size' to be >= 1 and <= ") +
         std::to_string(model_state_->MaxBatchSize()))
            .c_str());
  }
  if (graph_spec.lower_bound_batch_size_ > graph_spec.batch_size_) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "graph lower bound spec expects 'batch_size' to be <= graph "
        "spec "
        "'batch_size'");
  }
  for (const auto& input : graph_spec.shapes_) {
    const auto lit = graph_spec.lower_bound_shapes_.find(input.first);
    if (lit == graph_spec.lower_bound_shapes_.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("graph lower bound spec expects shape for input '") +
           input.first + "'")
              .c_str());
    } else {
      if (lit->second.size() != input.second.size()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("graph lower bound spec expects to have '") +
             std::to_string(input.second.size()) + " dimensions, got " +
             std::to_string(lit->second.size()))
                .c_str());
      }
      for (size_t idx = 0; idx < input.second.size(); idx++) {
        if ((lit->second[idx] < 0) || (input.second[idx] < 0)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("graph spec expects input '") + input.first +
               "' to have dimension >= 0")
                  .c_str());
        }
        if (lit->second[idx] > input.second[idx]) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("graph lower bound spec expects input '") +
               input.first +
               "' to have dimension <= " + std::to_string(input.second[idx]))
                  .c_str());
        }
      }
    }
  }
  return nullptr;
}

bool
TRTv1Interface::BuildCudaGraph(
    TensorRTContext* trt_context, const GraphSpec& graph_spec)
{
  // 1 is special case as non-batching model has 'max_batch_size == 0'
  int batch_size = (graph_spec.batch_size_ == 0) ? 1 : graph_spec.batch_size_;
  std::vector<int64_t> cuda_graph_key{batch_size};
  auto cuda_graph = TensorRTContext::CudaGraph();
  int lower_bound_batch_size = (graph_spec.lower_bound_batch_size_ == 0)
                                   ? 1
                                   : graph_spec.lower_bound_batch_size_;
  cuda_graph.lower_bound_key_ = {lower_bound_batch_size};
  for (int io_index = 0; io_index < instance_->num_expected_bindings_;
       ++io_index) {
    // FIXME handle shape tensor properly, for now if model uses shape
    // tensor then cuda graph is not captured
    if (instance_->engine_->isShapeBinding(io_index)) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("Detected shape tensor, CUDA graph is not "
                       "captured for ") +
           instance_->Name())
              .c_str());
      return false;
    }
  }

  // Enqueue to TRT to setup resources properly BEFORE capturing CUDA
  // graph
  for (int s = 0; s < instance_->num_copy_streams_; s++) {
    if (!trt_context->context_->enqueue(
            batch_size, instance_->buffer_bindings_[s].data(),
            instance_->CudaStream(), nullptr)) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("unable to record CUDA graph for ") + instance_->Name())
              .c_str());
      return false;
    }
    cudaStreamSynchronize(instance_->CudaStream());
  }

  bool captured = true;
  for (int set_idx = 0; set_idx < EVENT_SET_COUNT; set_idx++) {
    // The same spec has been captured
    if (trt_context->cuda_graph_execs_[set_idx].find(cuda_graph_key) !=
        trt_context->cuda_graph_execs_[set_idx].end()) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("Detected duplicated CUDA graph specification for ") +
           instance_->Name() + ", skipping the duplicated specification")
              .c_str());

      return true;
    }
    // Use second set of buffers to capture cuda graph if
    // double-buffering
    auto buffer_binding_index = set_idx % instance_->num_copy_streams_;
    cudaGraph_t graph;
    // Using cudaStreamCaptureModeThreadLocal mode to confine the graph
    // capture to this thread and avoid interference from other potentially
    // unsafe cuda calls.
    auto cuerr = cudaStreamBeginCapture(
        instance_->CudaStream(), cudaStreamCaptureModeThreadLocal);
    if (cuerr != cudaSuccess) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("unable to start CUDA graph for ") + instance_->Name() +
           ": " + cudaGetErrorString(cuerr))
              .c_str());
      captured = false;
    } else {
      auto context = trt_context->context_;
      if (!context->enqueue(
              batch_size,
              instance_->buffer_bindings_[buffer_binding_index].data(),
              instance_->CudaStream(),
              &instance_->events_[set_idx].ready_for_input_)) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("unable to record CUDA graph for ") +
             instance_->Name())
                .c_str());
        captured = false;
      }

      cuerr = cudaStreamEndCapture(instance_->CudaStream(), &graph);
      if (captured == false) {
        if (cuerr != cudaErrorStreamCaptureInvalidated) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("stream capture is not invalidated for ") +
               instance_->Name() + ": " + cudaGetErrorString(cuerr))
                  .c_str());
        }
        // There has been an error during graph capture. Below call resets the
        // sticky error from the cuda runtime.
        cudaGetLastError();
        // Verify if the  error has been cleared successfully.
        auto cuerr2 = cudaGetLastError();
        if (cuerr2 != cudaSuccess) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("unable to clear cuda runtime error for ") +
               instance_->Name() + ": " + cudaGetErrorString(cuerr2))
                  .c_str());
        }
      }
      if (cuerr != cudaSuccess) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("unable to finish CUDA graph for ") +
             instance_->Name() + ": " + cudaGetErrorString(cuerr))
                .c_str());
        captured = false;
      }

      if (captured) {
        cudaGraphExec_t graph_exec;
#if CUDART_VERSION >= 12000
        cuerr = cudaGraphInstantiate(&graph_exec, graph, 0);
#else
        cuerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
#endif
        if (cuerr != cudaSuccess) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("unable to instantiate CUDA graph for ") +
               instance_->Name() + ": " + cudaGetErrorString(cuerr))
                  .c_str());
          captured = false;
        } else {
          cuda_graph.cuda_graph_exec_ = graph_exec;

          trt_context->cuda_graph_execs_[set_idx].insert(
              std::make_pair(cuda_graph_key, cuda_graph));
        }
        cuerr = cudaGraphDestroy(graph);
        if (cuerr != cudaSuccess) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("unable to destroy graph for ") + instance_->Name() +
               ": " + cudaGetErrorString(cuerr))
                  .c_str());
          captured = false;
        }
      }
    }
  }

  if (captured) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("captured CUDA graph for ") + instance_->Name() +
         ", batch size " + std::to_string(batch_size))
            .c_str());
  }

  return captured;
}

bool
TRTv3Interface::BuildCudaGraph(
    TensorRTContext* trt_context, const GraphSpec& graph_spec)
{
  // FIXME handle shape tensor properly, for now if model uses shape
  // tensor then cuda graph is not captured
  for (int i = 0; i < instance_->num_expected_bindings_; ++i) {
    if (instance_->engine_->isShapeBinding(i)) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("Detected shape tensor, CUDA graph is not "
                       "captured for") +
           instance_->Name())
              .c_str());
      return false;
    }
  }

  std::vector<int64_t> cuda_graph_key;
  auto cuda_graph = TensorRTContext::CudaGraph();
  auto err =
      SetCudaGraphShape(trt_context, graph_spec, &cuda_graph_key, &cuda_graph);
  if (err != nullptr) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_ERROR,
        (std::string("Failed to set cuda graph shape for ") +
         instance_->Name() + TRITONSERVER_ErrorMessage(err))
            .c_str());
    TRITONSERVER_ErrorDelete(err);
    return false;
  }

  // [DLIS-4283] keeping current event / buffer idx for later recovery,
  // however do we need to restore them?
  int curr_buffer_binding_set = instance_->next_buffer_binding_set_;
  size_t curr_event_set_idx = instance_->next_set_;

  // Enqueue to TRT to setup resources properly BEFORE capturing CUDA
  // graph, so there hasn't been association with input consumed event yet.
  // [DLIS-4283] shouldn't need to repeat for all sets of bindings, reason
  // being that the resource setup is more likely to be related to the
  // execution context and not the binding buffers
  for (int s = 0; s < instance_->num_copy_streams_; s++) {
    // Use helper function to setTensorAddress if different buffer binding
    // should be used.
    instance_->next_buffer_binding_set_ = s;
    if (!Enqueue(trt_context->context_.get())) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("unable to record CUDA graph for ") + instance_->Name())
              .c_str());
      return false;
    }
    cudaStreamSynchronize(instance_->CudaStream());
  }

  bool captured = true;

  for (int set_idx = 0; set_idx < EVENT_SET_COUNT; set_idx++) {
    cudaGraph_t graph;
    instance_->next_buffer_binding_set_ =
        set_idx % instance_->num_copy_streams_;
    instance_->next_set_ = set_idx;
    // Using cudaStreamCaptureModeThreadLocal mode to confine the graph
    // capture to this thread and avoid interference from other potentially
    // unsafe cuda calls.
    auto cuerr = cudaStreamBeginCapture(
        instance_->CudaStream(), cudaStreamCaptureModeThreadLocal);
    if (cuerr != cudaSuccess) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("unable to start CUDA graph for ") + instance_->Name() +
           ": " + cudaGetErrorString(cuerr))
              .c_str());
      captured = false;
    } else {
      auto context = trt_context->context_;
      if (!Enqueue(context.get())) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("unable to record CUDA graph for ") +
             instance_->Name())
                .c_str());
        captured = false;
      }

      cuerr = cudaStreamEndCapture(instance_->CudaStream(), &graph);
      if (captured == false) {
        if (cuerr != cudaErrorStreamCaptureInvalidated) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("stream capture is not invalidated for ") +
               instance_->Name() + ": " + cudaGetErrorString(cuerr))
                  .c_str());
        }
        // There has been an error during graph capture. Below call resets the
        // sticky error from the cuda runtime.
        cudaGetLastError();
        // Verify if the  error has been cleared successfully.
        auto cuerr2 = cudaGetLastError();
        if (cuerr2 != cudaSuccess) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("unable to clear cuda runtime error for ") +
               instance_->Name() + ": " + cudaGetErrorString(cuerr2))
                  .c_str());
        }
      }
      if (cuerr != cudaSuccess) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("unable to finish CUDA graph for ") +
             instance_->Name() + ": " + cudaGetErrorString(cuerr))
                .c_str());
        captured = false;
      }

      if (captured) {
        cudaGraphExec_t graph_exec;
#if CUDART_VERSION >= 12000
        cuerr = cudaGraphInstantiate(&graph_exec, graph, 0);
#else
        cuerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
#endif
        if (cuerr != cudaSuccess) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("unable to instantiate CUDA graph for ") +
               instance_->Name() + ": " + cudaGetErrorString(cuerr))
                  .c_str());
          captured = false;
        } else {
          cuda_graph.cuda_graph_exec_ = graph_exec;
          trt_context->cuda_graph_execs_[set_idx].insert(
              std::make_pair(cuda_graph_key, cuda_graph));
        }
        cuerr = cudaGraphDestroy(graph);
        if (cuerr != cudaSuccess) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_ERROR,
              (std::string("unable to destroy graph for ") + instance_->Name() +
               ": " + cudaGetErrorString(cuerr))
                  .c_str());
          captured = false;
        }
      }
    }
  }
  instance_->next_buffer_binding_set_ = curr_buffer_binding_set;
  instance_->next_set_ = curr_event_set_idx;

  if (captured) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("captured CUDA graph for ") + instance_->Name() +
         ", batch size " + std::to_string(graph_spec.batch_size_))
            .c_str());
  }

  return captured;
}

TRITONSERVER_Error*
TRTv3Interface::SetCudaGraphShape(
    TensorRTContext* trt_context, const GraphSpec& graph_spec,
    std::vector<int64_t>* cuda_graph_key,
    TensorRTContext::CudaGraph* cuda_graph)
{
  int batch_size = graph_spec.batch_size_;
  int binding_offset =
      trt_context->profile_idx_ * instance_->num_expected_bindings_;
  *cuda_graph_key = std::vector<int64_t>{batch_size};
  auto& lower_bound_key = cuda_graph->lower_bound_key_;
  lower_bound_key.push_back(
      (graph_spec.lower_bound_batch_size_ == 0)
          ? 1
          : graph_spec.lower_bound_batch_size_);
  for (int io_index = 0; io_index < instance_->num_expected_bindings_;
       io_index++) {
    auto& io_binding_info = instance_->io_binding_infos_[0][io_index];
    auto binding_index = binding_offset + io_index;
    if (!instance_->engine_->bindingIsInput(binding_index)) {
      continue;
    }
    // Empty shapes indicates the graph spec is added by default,
    // for default graph spec, opt dims are used.
    if (graph_spec.shapes_.empty()) {
      auto shape = trt_context->opt_dims_[io_index];
      if (batch_size != 0) {
        shape.d[0] = batch_size;
      }
      if (!trt_context->context_->setBindingDimensions(binding_index, shape)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("trt failed to set binding dimension to ") +
             DimsDebugString(shape) + " for binding " +
             std::to_string(binding_index) + " for " + instance_->Name())
                .c_str());
      }
      std::vector<int64_t> dims;
      DimsToDimVec(shape, &dims);
      cuda_graph->input_dims_.emplace_back(dims);
      cuda_graph_key->insert(cuda_graph_key->end(), dims.begin(), dims.end());
      lower_bound_key.insert(lower_bound_key.end(), dims.begin(), dims.end());
    } else {
      const std::string& name = instance_->engine_->getBindingName(io_index);
      auto it = graph_spec.shapes_.find(name);
      if (it != graph_spec.shapes_.end()) {
        // For ragged / batch input, assume the shape in graph spec is proper
        // shape after ragged.
        if (io_binding_info.IsBufferRagged() ||
            (io_binding_info.GetBatchInput() != nullptr)) {
          cuda_graph->input_dims_.emplace_back();
        } else {
          cuda_graph->input_dims_.emplace_back();
          if (batch_size != 0) {
            cuda_graph->input_dims_.back().push_back(batch_size);
          }
          lower_bound_key.push_back(lower_bound_key[0]);
        }
        auto& shape = cuda_graph->input_dims_.back();
        shape.insert(shape.end(), it->second.begin(), it->second.end());
        nvinfer1::Dims trt_shape;
        DimVecToDims(shape, &trt_shape);
        if (!trt_context->context_->setBindingDimensions(
                binding_index, trt_shape)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("trt failed to set binding dimension to ") +
               DimsDebugString(trt_shape) + " for binding " +
               std::to_string(binding_index) + " for " + instance_->Name())
                  .c_str());
        }
        cuda_graph_key->insert(
            cuda_graph_key->end(), shape.begin(), shape.end());
        auto lit = graph_spec.lower_bound_shapes_.find(name);
        lower_bound_key.insert(
            lower_bound_key.end(), lit->second.begin(), lit->second.end());
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("trt failed to set binding dimension for "
                         "unknown input '") +
             name + "' for " + instance_->Name())
                .c_str());
      }
    }
  }
  return nullptr;
}

#endif  // TRITON_ENABLE_CUDA_GRAPH

bool
TRTv1Interface::Enqueue(nvinfer1::IExecutionContext* context)
{
  return context->enqueue(
      instance_->payload_->total_batch_size_,
      instance_->buffer_bindings_[instance_->next_buffer_binding_set_].data(),
      instance_->stream_,
      &instance_->events_[instance_->next_set_].ready_for_input_);
}

int64_t
TRTv1Interface::GetFullByteSize(
    nvinfer1::IExecutionContext* context, const std::string& tensor_name,
    int32_t binding_index)
{
  auto dt = ConvertTrtTypeToDataType(
      instance_->engine_->getBindingDataType(binding_index));
  const nvinfer1::Dims output_dim =
      context->getBindingDimensions(binding_index);
  std::vector<int64_t> dim_vec;
  DimsToDimVec(output_dim, &dim_vec);

  std::vector<int64_t> dim_vec_with_mbs;
  if (instance_->support_batching_) {
    dim_vec_with_mbs.push_back(instance_->model_state_->MaxBatchSize());
  }
  dim_vec_with_mbs.insert(
      dim_vec_with_mbs.end(), dim_vec.begin(), dim_vec.end());
  return GetByteSize(dt, dim_vec_with_mbs);
}

TRITONSERVER_Error*
TRTv1Interface::SetFormat(int binding_index, TensorFormat* format)
{
  format->is_linear_format_ =
      (instance_->engine_->getBindingFormat(binding_index) ==
       nvinfer1::TensorFormat::kLINEAR);
  if (!format->is_linear_format_) {
    format->vectorized_dim_ =
        instance_->engine_->getBindingVectorizedDim(binding_index);
    format->components_per_element_ =
        instance_->engine_->getBindingComponentsPerElement(binding_index);
    if (format->vectorized_dim_ == -1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unexpected vectorized dim is -1 for non-linear "
                       "tensor '") +
           instance_->engine_->getBindingName(binding_index) + "' for " +
           instance_->Name())
              .c_str());
    }
    // +1 for implicit batch dimension to take full shape dim as reference
    if (instance_->support_batching_) {
      format->vectorized_dim_++;
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
TRTv1Interface::ConfigureInputDimensions(
    TensorRTContext* context, int io_index, int binding_index,
    std::vector<int64_t> full_config_dims, std::vector<int64_t>* maximum_dims)
{
  // Below is more of a sanity check, v1 only support fixed shape tensor so
  // config shape must be exact match of engine shape.

  // v1 slightly different that TRT dim doesn't contain batch dimension
  TRITONSERVER_Error* err = ValidateDimension(
      full_config_dims, context->min_dims_[io_index],
      context->max_dims_[io_index],
      instance_->support_batching_ /* skip_first_dimension */);
  if (err != nullptr) {
    TRITONSERVER_Error* full_err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("model configuration specified invalid shape for "
                     "input '") +
         instance_->engine_->getBindingName(io_index) + "' for model " +
         instance_->model_state_->Name() +
         ". Error details: " + TRITONSERVER_ErrorMessage(err))
            .c_str());

    TRITONSERVER_ErrorDelete(err);
    return full_err;
  }
  *maximum_dims = full_config_dims;
  return nullptr;
}

TRITONSERVER_Error*
TRTv1Interface::SetBindingDimensions(
    const std::string& input_name, const std::vector<int64_t>& shape,
    const TensorRTContext& trt_context, const size_t io_index,
    const size_t binding_index, std::vector<int64_t>* cuda_graph_key)
{
  // NO-OP, v1 engine doesn't support dynamic shape, so input shape won't
  // need to be set at runtime, for the same reason 'cuda_graph_key' won't
  // be changed at runtime as long as the batch size has set.
  return nullptr;
}

bool
TRTv3Interface::Enqueue(nvinfer1::IExecutionContext* context)
{
  if (SetTensorAddress(context)) {
    if (context->setInputConsumedEvent(
            instance_->events_[instance_->next_set_].ready_for_input_)) {
      return context->enqueueV3(instance_->stream_);
    }
  }
  return false;
}

bool
TRTv3Interface::SetTensorAddress(nvinfer1::IExecutionContext* context)
{
  const auto& io_binding_info =
      instance_->io_binding_infos_[instance_->next_buffer_binding_set_];
  for (const auto& info : io_binding_info) {
    if (!context->setTensorAddress(
            info.GetName().c_str(), info.GetDeviceBuffer())) {
      return false;
    }
  }
  return true;
}

int64_t
TRTv3Interface::GetFullByteSize(
    nvinfer1::IExecutionContext* context, const std::string& tensor_name,
    int32_t binding_index)
{
  return context->getMaxOutputSize(tensor_name.c_str());
}

TRITONSERVER_Error*
TRTv3Interface::SetFormat(int binding_index, TensorFormat* format)
{
  format->is_linear_format_ =
      (instance_->engine_->getBindingFormat(binding_index) ==
       nvinfer1::TensorFormat::kLINEAR);
  if (!format->is_linear_format_) {
    format->vectorized_dim_ =
        instance_->engine_->getBindingVectorizedDim(binding_index);
    format->components_per_element_ =
        instance_->engine_->getBindingComponentsPerElement(binding_index);
    if (format->vectorized_dim_ == -1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unexpected vectorized dim is -1 for non-linear "
                       "tensor '") +
           instance_->engine_->getBindingName(binding_index) + "' for " +
           instance_->Name())
              .c_str());
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
TRTv3Interface::ConfigureInputDimensions(
    TensorRTContext* context, int io_index, int binding_index,
    std::vector<int64_t> full_config_dims, std::vector<int64_t>* maximum_dims)
{
  // [FIXME] WAR: max batch size needs to be special handled as it is really
  // a dynamic shape, setting it to fixed value will interfere with below
  // dimension checkings
  const auto batch_dim = full_config_dims[0];
  if (instance_->support_batching_) {
    full_config_dims[0] = -1;
  }
  TRITONSERVER_Error* err = ValidateDimension(
      full_config_dims, context->min_dims_[io_index],
      context->max_dims_[io_index], false /* skip_first_dimension */);
  if (err != nullptr) {
    TRITONSERVER_Error* full_err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("model configuration specified invalid shape for "
                     "input '") +
         instance_->engine_->getBindingName(io_index) + "' for model " +
         instance_->model_state_->Name() +
         ". Error details: " + TRITONSERVER_ErrorMessage(err))
            .c_str());

    TRITONSERVER_ErrorDelete(err);
    return full_err;
  }

  // Only "prune" maximum dims for the input given context max dim and
  // config shape, user may provide fixed config value even for dynamic dim,
  // to reduce the size of allocation.
  RETURN_IF_ERROR(MaximumDims(
      context->max_dims_[io_index], full_config_dims, maximum_dims));
  // WAR, see [FIXME] above
  if (instance_->support_batching_ && (batch_dim != -1)) {
    (*maximum_dims)[0] = std::min((*maximum_dims)[0], batch_dim);
  }
  DimVecToDims(*maximum_dims, &context->max_dims_[io_index]);

  if (!context->context_->setBindingDimensions(
          binding_index, context->max_dims_[io_index])) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("trt failed to set binding dimension to ") +
         DimsDebugString(context->max_dims_[io_index]) + " for input '" +
         instance_->engine_->getBindingName(io_index) + "' for " +
         instance_->Name())
            .c_str());
  }

  return nullptr;
}

TRITONSERVER_Error*
TRTv3Interface::MaximumDims(
    const nvinfer1::Dims& max_profile_dims, const std::vector<int64_t>& dims,
    std::vector<int64_t>* max_dims)
{
  if (max_profile_dims.nbDims != (int32_t)dims.size()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("can not maximize dimension ") +
         backend::ShapeToString(dims) + " to " +
         DimsDebugString(max_profile_dims) + " due to  incompatibility.")
            .c_str());
  }

  for (uint64_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == WILDCARD_DIM) {
      max_dims->emplace_back(max_profile_dims.d[i]);
    } else {
      if (dims[i] <= max_profile_dims.d[i]) {
        max_dims->emplace_back(dims[i]);
      } else {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("can not maximize dimension ") +
             backend::ShapeToString(dims) + " to " +
             DimsDebugString(max_profile_dims) + " due to incompatibility.")
                .c_str());
      }
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
TRTv3Interface::SetBindingDimensions(
    const std::string& input_name, const std::vector<int64_t>& shape,
    const TensorRTContext& trt_context, const size_t io_index,
    const size_t binding_index, std::vector<int64_t>* cuda_graph_key)
{
  if (cuda_graph_key != nullptr) {
    cuda_graph_key->insert(cuda_graph_key->end(), shape.begin(), shape.end());
  }

  nvinfer1::Dims this_dim;
  // Set the binding dimension so that output dimensions can be
  // obtained
  if (!DimVecToDims(shape, &this_dim)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to create dims object for ") +
         ShapeToString(shape) + " for input '" + input_name + "' for " +
         instance_->name_ + ".")
            .c_str());
  }
  TRITONSERVER_Error* err = ValidateDimension(
      this_dim, trt_context.min_dims_[io_index],
      trt_context.max_dims_[io_index]);
  if (err != nullptr) {
    TRITONSERVER_Error* full_err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("request specifies invalid shape for input '") +
         input_name + "' for " + instance_->name_ +
         ". Error details: " + TRITONSERVER_ErrorMessage(err))
            .c_str());
    TRITONSERVER_ErrorDelete(err);
    return full_err;
  }

  if (!trt_context.is_dynamic_per_binding_[io_index]) {
    // No need to set dimension for the binding that does not include
    // dynamic shape.
    return nullptr;
  }

  if (!trt_context.context_->setBindingDimensions(binding_index, this_dim)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("trt failed to set binding dimension to ") +
         DimsDebugString(this_dim) + " for input '" + input_name + "' for " +
         instance_->Name())
            .c_str());
  }

  return nullptr;
}
}}}  // namespace triton::backend::tensorrt
