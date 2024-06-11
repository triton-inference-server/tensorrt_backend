// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace tensorrt {

enum class ShapeTensorDataType { INT32, INT64 };

class ShapeTensor {
 public:
  ShapeTensor()
      : size_(0), element_cnt_(0), datatype_(ShapeTensorDataType::INT32),
        data_(nullptr)
  {
  }

  TRITONSERVER_Error* SetDataFromBuffer(
      const char* data, const TRITONSERVER_DataType datatype,
      const size_t element_cnt, const bool support_batching,
      const size_t total_batch_size);

  TRITONSERVER_Error* SetDataFromShapeValues(
      const int32_t* shape_values, const TRITONSERVER_DataType datatype,
      const size_t element_cnt);

  int64_t GetDistance(
      const ShapeTensor& other, const int64_t total_batch_size) const;
  const char* GetDataTypeString() const;

  size_t GetSize() const { return size_; }
  size_t GetElementCount() const { return element_cnt_; }
  ShapeTensorDataType GetDataType() const { return datatype_; }
  const void* GetData() const { return static_cast<const void*>(data_.get()); }

 private:
  size_t size_;
  size_t element_cnt_;
  ShapeTensorDataType datatype_;
  std::unique_ptr<char[]> data_;
};

}}}  // namespace triton::backend::tensorrt
