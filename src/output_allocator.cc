// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "output_allocator.h"

namespace triton { namespace backend { namespace tensorrt {

void*
OutputAllocator::reallocateOutput(
    char const* tensor_name, void* current_memory, uint64_t size,
    uint64_t alignment) noexcept
{
  // When the requested output size is larger than the current size,
  // free the allocated memory and attempt to allocate the larger size
  // for the underlying output buffer.
  if (size > output_size_) {
    cudaFree(output_ptr_);
    output_ptr_ = nullptr;
    output_size_ = 0;
    if (zero_copy_support_) {
      cudaHostAlloc(&output_ptr_, size, cudaHostAllocMapped);
      // If zero copy support is enabled, need to set the buffer to the device
      // pointer.
      void* device_buffer;
      auto err = cudaHostGetDevicePointer(&device_buffer, &output_ptr_, 0);
      if (err == cudaSuccess) {
        output_ptr_ = device_buffer;
      }
    } else {
      cudaMalloc(&output_ptr_, size);
    }

    // If the memory allocation fails, output_ptr_=nullptr and engine
    // gracefully fails.
    if (output_ptr_ != nullptr) {
      output_size_ = size;
    }
  }
  return output_ptr_;
}

void
OutputAllocator::notifyShape(
    char const* tensor_name, nvinfer1::Dims const& dims) noexcept
{
  output_dims_ = dims;
}

OutputAllocator::~OutputAllocator()
{
  cudaFree(output_ptr_);
}

}}}  // namespace triton::backend::tensorrt
