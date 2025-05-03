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

#include "shape_tensor.h"

namespace triton { namespace backend { namespace tensorrt {

TRITONSERVER_Error*
ShapeTensor::SetDataFromBuffer(
    const char* data_buffer, size_t data_byte_size,
    TRITONSERVER_DataType datatype, size_t nb_shape_values,
    const char* input_name, bool support_batching, size_t total_batch_size)
{
  if (data_buffer == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Null data pointer received for Shape tensor");
  }

  if (datatype == TRITONSERVER_DataType::TRITONSERVER_TYPE_INT32) {
    datatype_ = ShapeTensorDataType::INT32;
  } else if (datatype == TRITONSERVER_DataType::TRITONSERVER_TYPE_INT64) {
    datatype_ = ShapeTensorDataType::INT64;
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Unsupported data type received for Shape tensor");
  }

  nb_shape_values_ = nb_shape_values;
  if (support_batching) {
    nb_shape_values_++;  // Account for batch size
  }
  const size_t datatype_size = TRITONSERVER_DataTypeByteSize(datatype);
  size_ = nb_shape_values_ * datatype_size;

  TRITONSERVER_Error* err =
      ValidateDataByteSize(data_byte_size, input_name, datatype_size);
  if (err != nullptr) {
    return err;
  }

  data_ = std::make_unique<char[]>(size_);
  if (support_batching) {
    if (datatype_ == ShapeTensorDataType::INT32) {
      *reinterpret_cast<int32_t*>(data_.get()) =
          static_cast<int32_t>(total_batch_size);
    } else if (datatype_ == ShapeTensorDataType::INT64) {
      *reinterpret_cast<int64_t*>(data_.get()) =
          static_cast<int64_t>(total_batch_size);
    }
    if (size_ < datatype_size) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "Unexpected integer underflow while calculating shape tensor size.");
    }
    std::memcpy(
        data_.get() + datatype_size, data_buffer, size_ - datatype_size);
  } else {
    std::memcpy(data_.get(), data_buffer, size_);
  }

  return nullptr;
}

TRITONSERVER_Error*
ShapeTensor::SetDataFromShapeValues(
    const int32_t* shape_values, TRITONSERVER_DataType datatype,
    size_t nb_shape_values)
{
  nb_shape_values_ = nb_shape_values;
  const size_t datatype_size = TRITONSERVER_DataTypeByteSize(datatype);
  size_ = nb_shape_values_ * datatype_size;

  if (datatype == TRITONSERVER_DataType::TRITONSERVER_TYPE_INT32) {
    datatype_ = ShapeTensorDataType::INT32;
    data_.reset(new char[size_]);
    int32_t* data_ptr = reinterpret_cast<int32_t*>(data_.get());
    std::memcpy(data_ptr, shape_values, size_);
  } else if (datatype == TRITONSERVER_DataType::TRITONSERVER_TYPE_INT64) {
    datatype_ = ShapeTensorDataType::INT64;
    data_.reset(new char[size_]);
    int64_t* data_ptr = reinterpret_cast<int64_t*>(data_.get());
    for (size_t i = 0; i < nb_shape_values_; ++i) {
      data_ptr[i] = static_cast<int64_t>(shape_values[i]);
    }
  } else {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Unsupported data type received for Shape tensor");
  }

  return nullptr;
}

int64_t
ShapeTensor::GetDistance(
    const ShapeTensor& other, int64_t total_batch_size) const
{
  int64_t distance = 0;
  if (datatype_ == ShapeTensorDataType::INT32) {
    const auto* shape_values = reinterpret_cast<const int32_t*>(data_.get());
    const auto* opt_shape_values =
        reinterpret_cast<const int32_t*>(other.GetData());
    distance += std::abs(*opt_shape_values - total_batch_size);
    for (size_t idx = 1; idx < other.GetNbShapeValues(); idx++) {
      distance += std::abs(*(opt_shape_values + idx) - shape_values[idx - 1]);
    }
  } else {
    const auto* shape_values = reinterpret_cast<const int64_t*>(data_.get());
    const auto* opt_shape_values =
        reinterpret_cast<const int64_t*>(other.GetData());
    distance += std::abs(*opt_shape_values - total_batch_size);
    for (size_t idx = 1; idx < other.GetNbShapeValues(); idx++) {
      distance += std::abs(*(opt_shape_values + idx) - shape_values[idx - 1]);
    }
  }
  return distance;
}

const char*
ShapeTensor::GetDataTypeString() const
{
  switch (datatype_) {
    case ShapeTensorDataType::INT32:
      return "INT32";
    case ShapeTensorDataType::INT64:
      return "INT64";
    default:
      break;
  }
  return nullptr;
}

TRITONSERVER_Error*
ShapeTensor::ValidateDataByteSize(
    size_t expected_byte_size, const char* input_name,
    size_t datatype_size) const
{
  if (expected_byte_size != (size_ - datatype_size) &&
      (expected_byte_size != size_)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("shape tensor for input '") + input_name +
         "' expected byte size is " + std::to_string(expected_byte_size) +
         " [ or " + std::to_string(size_) +
         " if input includes batch shape value] " + ", got " +
         std::to_string(expected_byte_size))
            .c_str());
  }
  return nullptr;
}

}}}  // namespace triton::backend::tensorrt
