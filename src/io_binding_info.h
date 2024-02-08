// Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>

#include "output_allocator.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace tensorrt {

struct TensorFormat {
  bool is_linear_format_{true};
  int vectorized_dim_{-1};
  int components_per_element_{1};
};

using BatchInputData = std::pair<BatchInput, std::unique_ptr<BackendMemory>>;

class IOBindingInfo {
 private:
  std::string name_{};
  uint64_t byte_size_{0};
  void* buffer_{nullptr};
  void* device_buffer_{nullptr};
  TRITONSERVER_MemoryType memory_type_{TRITONSERVER_MEMORY_GPU};
  int64_t memory_type_id_{0};
  bool is_buffer_ragged_{false};
  TensorFormat format_{};
  const BatchOutput* batch_output_{nullptr};
  std::shared_ptr<BatchInputData> batch_input_{nullptr};
  std::pair<std::string, std::vector<int64_t>> io_shape_mapping_{};
  bool is_state_output_{false};
  bool is_requested_output_tensor_{false};
  bool is_dynamic_shape_output_{false};
  std::unique_ptr<OutputAllocator> allocator_{nullptr};

 public:
  void SetName(const std::string& name);
  const std::string& GetName() const;

  void SetByteSize(uint64_t byte_size);
  uint64_t GetByteSize() const;

  void SetBuffer(void* buffer);
  void SetBuffer(std::unique_ptr<OutputAllocator> allocator);
  void* GetBuffer() const;

  void SetDeviceBuffer(void* device_buffer);
  void* GetDeviceBuffer() const;
  void** GetDeviceBufferAddr();

  void SetMemoryType(TRITONSERVER_MemoryType memory_type);
  TRITONSERVER_MemoryType GetMemoryType() const;

  void SetMemoryTypeId(int64_t memory_type_id);
  int64_t GetMemoryTypeId() const;

  void SetIsBufferRagged(bool is_buffer_ragged);
  bool IsBufferRagged() const;

  void SetFormat(const TensorFormat& format);
  TensorFormat& GetFormat();

  void SetBatchOutput(const BatchOutput* batch_output);
  const BatchOutput* GetBatchOutput() const;

  void SetBatchInput(const BatchInput& batch_input);
  const std::shared_ptr<BatchInputData>& GetBatchInput() const;

  void SetIoShapeMapping(
      const std::pair<std::string, std::vector<int64_t>>& io_shape_mapping);
  std::pair<std::string, std::vector<int64_t>>& GetIoShapeMapping();

  void SetIsStateOutput(bool is_state_output);
  bool IsStateOutput() const;

  void SetIsRequestedOutputTensor(bool is_requested_output_tensor);
  bool IsRequestedOutputTensor() const;

  void SetIsDynamicShapeOutput(bool is_dynamic_shape_output);
  bool IsDynamicShapeOutput() const;

  bool IsBufferAllocated() const;

  OutputAllocator* GetAllocator();
};

}}}  // namespace triton::backend::tensorrt
