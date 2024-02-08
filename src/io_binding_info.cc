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

#include "io_binding_info.h"

namespace triton { namespace backend { namespace tensorrt {

void
IOBindingInfo::SetName(const std::string& name)
{
  name_ = name;
}

const std::string&
IOBindingInfo::GetName() const
{
  return name_;
}

void
IOBindingInfo::SetByteSize(uint64_t byte_size)
{
  byte_size_ = byte_size;
}

uint64_t
IOBindingInfo::GetByteSize() const
{
  return byte_size_;
}

void
IOBindingInfo::SetBuffer(void* buffer)
{
  buffer_ = buffer;
}

void
IOBindingInfo::SetBuffer(std::unique_ptr<OutputAllocator> allocator)
{
  allocator_ = std::move(allocator);
  is_dynamic_shape_output_ = true;
}

void*
IOBindingInfo::GetBuffer() const
{
  if (is_dynamic_shape_output_) {
    if (allocator_ == nullptr) {
      return nullptr;
    } else {
      return allocator_->getBuffer();
    }
  } else {
    return buffer_;
  }
}

void
IOBindingInfo::SetDeviceBuffer(void* device_buffer)
{
  device_buffer_ = device_buffer;
}

void*
IOBindingInfo::GetDeviceBuffer() const
{
  if (is_dynamic_shape_output_) {
    if (allocator_ == nullptr) {
      return nullptr;
    } else {
      return allocator_->getBuffer();
    }
  } else {
    return device_buffer_;
  }
}

void**
IOBindingInfo::GetDeviceBufferAddr()
{
  if (is_dynamic_shape_output_) {
    return allocator_->getBufferAddr();
  } else {
    return &device_buffer_;
  }
}

void
IOBindingInfo::SetMemoryType(TRITONSERVER_MemoryType memory_type)
{
  memory_type_ = memory_type;
}

TRITONSERVER_MemoryType
IOBindingInfo::GetMemoryType() const
{
  return memory_type_;
}

void
IOBindingInfo::SetMemoryTypeId(int64_t memory_type_id)
{
  memory_type_id_ = memory_type_id;
}

int64_t
IOBindingInfo::GetMemoryTypeId() const
{
  return memory_type_id_;
}

void
IOBindingInfo::SetIsBufferRagged(bool is_buffer_ragged)
{
  is_buffer_ragged_ = is_buffer_ragged;
}

bool
IOBindingInfo::IsBufferRagged() const
{
  return is_buffer_ragged_;
}

void
IOBindingInfo::SetFormat(const TensorFormat& format)
{
  format_ = format;
}

TensorFormat&
IOBindingInfo::GetFormat()
{
  return format_;
}

void
IOBindingInfo::SetBatchOutput(const BatchOutput* batch_output)
{
  batch_output_ = batch_output;
}

const BatchOutput*
IOBindingInfo::GetBatchOutput() const
{
  return batch_output_;
}

void
IOBindingInfo::SetBatchInput(const BatchInput& batch_input)
{
  batch_input_.reset(new BatchInputData(batch_input, nullptr));
}

const std::shared_ptr<BatchInputData>&
IOBindingInfo::GetBatchInput() const
{
  return batch_input_;
}

void
IOBindingInfo::SetIoShapeMapping(
    const std::pair<std::string, std::vector<int64_t>>& io_shape_mapping)
{
  io_shape_mapping_ = io_shape_mapping;
}

std::pair<std::string, std::vector<int64_t>>&
IOBindingInfo::GetIoShapeMapping()
{
  return io_shape_mapping_;
}

void
IOBindingInfo::SetIsStateOutput(bool is_state_output)
{
  is_state_output_ = is_state_output;
}

bool
IOBindingInfo::IsStateOutput() const
{
  return is_state_output_;
}

void
IOBindingInfo::SetIsRequestedOutputTensor(bool is_requested_output_tensor)
{
  is_requested_output_tensor_ = is_requested_output_tensor;
}

bool
IOBindingInfo::IsRequestedOutputTensor() const
{
  return is_requested_output_tensor_;
}

void
IOBindingInfo::SetIsDynamicShapeOutput(bool is_dynamic_shape_output)
{
  is_dynamic_shape_output_ = is_dynamic_shape_output;
}

bool
IOBindingInfo::IsDynamicShapeOutput() const
{
  return is_dynamic_shape_output_;
}

bool
IOBindingInfo::IsBufferAllocated() const
{
  return (buffer_ != nullptr) || (allocator_ != nullptr);
}

OutputAllocator*
IOBindingInfo::GetAllocator()
{
  if (allocator_) {
    return allocator_.get();
  } else {
    return nullptr;
  }
}

}}}  // namespace triton::backend::tensorrt
