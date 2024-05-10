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

#pragma once

#include <NvInfer.h>
#include <malloc.h>

namespace triton { namespace backend { namespace tensorrt {

class OutputAllocator : public nvinfer1::IOutputAllocator {
  // This class extends nvinfer1::IOutputAllocator and its functions
  // reallocateOutput and notifyShape. For consistency, all of its
  // functions use camel case.
 public:
  OutputAllocator(bool zero_copy_support)
      : zero_copy_support_(zero_copy_support)
  {
  }
  // Allocates output dimensions
  void* reallocateOutput(
      char const* tensor_name, void* current_memory, uint64_t size,
      uint64_t alignment) noexcept override;

  // Updates output dimensions
  void notifyShape(
      char const* tensor_name, nvinfer1::Dims const& dims) noexcept override;

  void* getBuffer() { return output_ptr_; };
  void** getBufferAddr() { return &output_ptr_; };

  ~OutputAllocator() override;

 private:
  // Saved dimensions of the output tensor
  nvinfer1::Dims output_dims_{};

  // Pointer to output, nullptr if memory could not be allocated
  void* output_ptr_{nullptr};

  // Size of allocation pointed to by output
  uint64_t output_size_{0};

  // Boolean flag indicating if zero copy support is enabled
  bool zero_copy_support_{false};
};

}}}  // namespace triton::backend::tensorrt
