// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_utils.h"

#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace tensorrt {

TRITONSERVER_DataType
ConvertTrtTypeToDataType(nvinfer1::DataType trt_type)
{
  switch (trt_type) {
    case nvinfer1::DataType::kFLOAT:
      return TRITONSERVER_TYPE_FP32;
    case nvinfer1::DataType::kHALF:
      return TRITONSERVER_TYPE_FP16;
    case nvinfer1::DataType::kINT8:
      return TRITONSERVER_TYPE_INT8;
    case nvinfer1::DataType::kINT32:
      return TRITONSERVER_TYPE_INT32;
    case nvinfer1::DataType::kBOOL:
      return TRITONSERVER_TYPE_BOOL;
  }

  return TRITONSERVER_TYPE_INVALID;
}

std::string
ConvertTrtTypeToConfigDataType(nvinfer1::DataType trt_type)
{
  switch (trt_type) {
    case nvinfer1::DataType::kFLOAT:
      return "TYPE_FP32";
    case nvinfer1::DataType::kHALF:
      return "TYPE_FP16";
    case nvinfer1::DataType::kINT8:
      return "TYPE_INT8";
    case nvinfer1::DataType::kINT32:
      return "TYPE_INT32";
    case nvinfer1::DataType::kBOOL:
      return "TYPE_BOOL";
  }

  return "TYPE_INVALID";
}

bool
UseTensorRTv2API(const std::shared_ptr<nvinfer1::ICudaEngine>& engine)
{
  // In order to use TensorRT V2 API, engine must contain
  // an explicit batch dimension. Detecting the presence of
  // an implicit batch dimension to detect whether or not
  // to use the TensorRT V2 API.
  return !engine->hasImplicitBatchDimension();
}

TRITONSERVER_Error*
GetProfileIndex(const std::string& profile_name, int* profile_index)
{
  if (profile_name.empty()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, "profile name must not be empty");
  }

  try {
    *profile_index = stoi(profile_name);
  }
  catch (const std::invalid_argument& ia) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to parse '") + profile_name + "': " + ia.what())
            .c_str());
  }

  return nullptr;
}

std::pair<bool, nvinfer1::DataType>
ConvertDataTypeToTrtType(const TRITONSERVER_DataType& dtype)
{
  nvinfer1::DataType trt_type = nvinfer1::DataType::kFLOAT;
  switch (dtype) {
    case TRITONSERVER_TYPE_FP32:
      trt_type = nvinfer1::DataType::kFLOAT;
      break;
    case TRITONSERVER_TYPE_FP16:
      trt_type = nvinfer1::DataType::kHALF;
      break;
    case TRITONSERVER_TYPE_INT8:
      trt_type = nvinfer1::DataType::kINT8;
      break;
    case TRITONSERVER_TYPE_INT32:
      trt_type = nvinfer1::DataType::kINT32;
      break;
    case TRITONSERVER_TYPE_BOOL:
      trt_type = nvinfer1::DataType::kBOOL;
      break;
    default:
      return std::make_pair(false, trt_type);
  }
  return std::make_pair(true, trt_type);
}

bool
CompareDims(const nvinfer1::Dims& model_dims, const std::vector<int64_t>& dims)
{
  if (model_dims.nbDims != (int32_t)dims.size()) {
    return false;
  }

  for (int i = 0; i < model_dims.nbDims; ++i) {
    if (model_dims.d[i] != dims[i]) {
      return false;
    }
  }

  return true;
}

bool
CompareDims(const nvinfer1::Dims& ldims, const nvinfer1::Dims& rdims)
{
  if (ldims.nbDims != rdims.nbDims) {
    return false;
  }

  for (int i = 0; i < ldims.nbDims; ++i) {
    if (ldims.d[i] != rdims.d[i]) {
      return false;
    }
  }

  return true;
}

TRITONSERVER_Error*
CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const nvinfer1::Dims& model_dims, common::TritonJson::Value& dims,
    const bool supports_batching, const bool contains_explicit_batch,
    const bool compare_exact)
{
  std::vector<int64_t> dims_vec;
  for (size_t i = 0; i < dims.ArraySize(); i++) {
    int64_t dim;
    RETURN_IF_ERROR(dims.IndexAsInt(i, &dim));
    dims_vec.push_back(dim);
  }

  RETURN_IF_ERROR(CompareDimsSupported(
      model_name, tensor_name, model_dims, dims_vec, supports_batching,
      contains_explicit_batch, compare_exact));
  return nullptr;
}

TRITONSERVER_Error*
CompareDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const nvinfer1::Dims& model_dims, const std::vector<int64_t>& dims,
    const bool supports_batching, const bool contains_explicit_batch,
    const bool compare_exact)
{
  // If the model configuration expects batching support in the model,
  // then the first dimension must be -1.
  if (supports_batching && contains_explicit_batch) {
    if ((model_dims.nbDims == 0)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("model '") + model_name + "', tensor '" + tensor_name +
           "': for the model to support batching the shape should have at "
           "least 1 dimension; but shape expected by the model is " +
           DimsDebugString(model_dims))
              .c_str());
    }

    std::vector<int64_t> full_dims;
    full_dims.emplace_back(WILDCARD_DIM);
    full_dims.insert(full_dims.end(), dims.begin(), dims.end());

    bool succ = (model_dims.nbDims == (int32_t)full_dims.size());
    if (succ) {
      for (uint64_t i = 0; i < full_dims.size(); ++i) {
        const int64_t model_dim = model_dims.d[i];
        if (compare_exact || ((model_dim != -1) && (full_dims[i] != -1))) {
          succ &= (model_dim == full_dims[i]);
        }
      }
    }

    if (!succ) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("model '") + model_name + "', tensor '" + tensor_name +
           "': the model expects " + std::to_string(model_dims.nbDims) +
           " dimensions (shape " + DimsDebugString(model_dims) +
           ") but the model configuration specifies " +
           std::to_string(full_dims.size()) +
           " dimensions (an initial batch dimension because max_batch_size "
           "> 0 followed by the explicit tensor shape, making complete "
           "shape " +
           backend::ShapeToString(full_dims) + ")")
              .c_str());
    }
  } else {
    // ! supports_batching
    bool succ = (model_dims.nbDims == (int32_t)dims.size());
    if (succ) {
      for (uint64_t i = 0; i < dims.size(); ++i) {
        const int64_t model_dim = model_dims.d[i];
        if (compare_exact || ((model_dim != -1) && (dims[i] != -1))) {
          succ &= (model_dim == dims[i]);
        }
      }
    }

    if (!succ) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("model '") + model_name + "', tensor '" + tensor_name +
           "': the model expects " + std::to_string(model_dims.nbDims) +
           " dimensions (shape " + DimsDebugString(model_dims) +
           ") but the model configuration specifies " +
           std::to_string(dims.size()) + " dimensions (shape " +
           backend::ShapeToString(dims) + ")")
              .c_str());
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
CompareShapeDimsSupported(
    const std::string& model_name, const std::string& tensor_name,
    const nvinfer1::Dims& model_dims, common::TritonJson::Value& dims,
    const bool supports_batching)
{
  std::vector<int64_t> dims_vec;
  for (size_t i = 0; i < dims.ArraySize(); i++) {
    int64_t dim;
    RETURN_IF_ERROR(dims.IndexAsInt(i, &dim));
    dims_vec.push_back(dim);
  }

  const int batch_offset = supports_batching ? 1 : 0;
  bool not_supported = false;
  if ((int32_t)dims_vec.size() != model_dims.nbDims) {
    not_supported = true;
  } else if (model_dims.nbDims == 1) {
    if (((dims_vec[0] + batch_offset) != model_dims.d[0]) ||
        (dims_vec[0] == WILDCARD_DIM)) {
      not_supported = true;
    }
  } else if (model_dims.nbDims > 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to load model '") + model_name +
         "', shape binding '" + tensor_name +
         "' can only be 0-d or 1-D tensor, got " + DimsDebugString(model_dims))
            .c_str());
  }


  if (not_supported) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to load model '") + model_name + "', binding '" +
         tensor_name + "' shape expected by framework " +
         DimsDebugString(model_dims) +
         " doesn't match model configuration shape " +
         backend::ShapeToString(dims_vec))
            .c_str());
  }

  return nullptr;
}

TRITONSERVER_Error*
MaximumDims(
    const nvinfer1::Dims& max_profile_dims, const std::vector<int64_t>& dims,
    const bool support_batching, const int max_batch_size,
    std::vector<int64_t>* max_dims)
{
  const int nonbatch_start_idx = (support_batching ? 1 : 0);
  if (max_profile_dims.nbDims != (int32_t)(dims.size() + nonbatch_start_idx)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("can not maximize dimension ") +
         backend::ShapeToString(dims) + " to " +
         DimsDebugString(max_profile_dims) + " due to  incompatibility.")
            .c_str());
  }

  if (support_batching) {
    int this_batch_size = max_batch_size > max_profile_dims.d[0]
                              ? max_profile_dims.d[0]
                              : max_batch_size;
    max_dims->emplace_back(this_batch_size);
  }

  for (uint64_t i = 0; i < dims.size(); ++i) {
    if (dims[i] == WILDCARD_DIM) {
      max_dims->emplace_back(max_profile_dims.d[i + nonbatch_start_idx]);
    } else {
      if (dims[i] <= max_profile_dims.d[i + nonbatch_start_idx]) {
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
ValidateDimension(
    const nvinfer1::Dims& this_dims, const nvinfer1::Dims& min_dims,
    const nvinfer1::Dims& max_dims, const bool skip_first_dimension)
{
  const int nonbatch_start_idx = (skip_first_dimension ? 1 : 0);
  if ((this_dims.nbDims + nonbatch_start_idx) != max_dims.nbDims) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("model expected ") +
         std::to_string(max_dims.nbDims - nonbatch_start_idx) +
         " dimensions but received " + std::to_string(this_dims.nbDims) +
         " dimensions")
            .c_str());
  }

  for (int i = 0; i < this_dims.nbDims; i++) {
    if (this_dims.d[i] < min_dims.d[i + nonbatch_start_idx] ||
        this_dims.d[i] > max_dims.d[i + nonbatch_start_idx]) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("model expected the shape of dimension ") +
           std::to_string(i) + " to be between " +
           std::to_string(min_dims.d[i + nonbatch_start_idx]) + " and " +
           std::to_string(max_dims.d[i + nonbatch_start_idx]) +
           " but received " + std::to_string(this_dims.d[i]))
              .c_str());
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ValidateControlDimsDynamic(
    const nvinfer1::Dims& dims, const bool support_batching)
{
  int expected_first_shape = (support_batching ? -1 : 1);
  if (dims.d[0] != expected_first_shape) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("expects the first dimension to be ") +
         std::to_string(expected_first_shape) + " but the model specified " +
         std::to_string(dims.d[0]))
            .c_str());
  }
  for (int i = 1; i < dims.nbDims; i++) {
    if (dims.d[i] != 1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("expects all dimensions (conditionally first) to be 1 "
                       "but the model specified shape ") +
           DimsDebugString(dims))
              .c_str());
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ValidateShapeValues(
    const std::vector<int32_t>& request_shape_values,
    const int32_t* min_shape_values, const int32_t* max_shape_values,
    size_t nb_shape_values, const bool support_batching)
{
  if (request_shape_values.size() != nb_shape_values) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string(
             "mismatch between the number of shape values. Expecting ") +
         std::to_string(nb_shape_values) + ". Got " +
         std::to_string(request_shape_values.size()))
            .c_str());
  }

  for (size_t i = 0; i < nb_shape_values; i++) {
    if (request_shape_values[i] < *(min_shape_values + i) ||
        request_shape_values[i] > *(max_shape_values + i)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("The shape value at index ") + std::to_string(i) +
           " is expected to be in range from " +
           std::to_string(*(min_shape_values + i)) + " to " +
           std::to_string(*(max_shape_values + i)) +
           ", Got: " + std::to_string(request_shape_values[i]))
              .c_str());
    }
  }
  return nullptr;
}

void
DimsToDimVec(const nvinfer1::Dims& model_dims, std::vector<int64_t>* dims)
{
  dims->clear();
  for (int i = 0; i < model_dims.nbDims; ++i) {
    dims->emplace_back(model_dims.d[i]);
  }
}

TRITONSERVER_Error*
DimsJsonToDimVec(
    common::TritonJson::Value& dims_json, std::vector<int64_t>* dims)
{
  dims->clear();
  for (size_t i = 0; i < dims_json.ArraySize(); i++) {
    int64_t dim;
    RETURN_IF_ERROR(dims_json.IndexAsInt(i, &dim));
    dims->push_back(dim);
  }
  return nullptr;
}


bool
DimVecToDims(const std::vector<int64_t>& dim_vec, nvinfer1::Dims* dims)
{
  if (dim_vec.size() > dims->MAX_DIMS) {
    return false;
  } else {
    dims->nbDims = dim_vec.size();
    for (int i = 0; i < dims->nbDims; ++i) {
      dims->d[i] = (int)dim_vec[i];
    }
  }
  return true;
}

int64_t
GetElementCount(const nvinfer1::Dims& dims)
{
  int64_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    count *= dims.d[i];
  }
  return count;
}

bool
ContainsWildcard(const nvinfer1::Dims& dims)
{
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] == WILDCARD_DIM) {
      return true;
    }
  }
  return false;
}

bool
ContainsWildcardAtExplicitBatchDim(const nvinfer1::Dims& dims)
{
  if (dims.d[0] == WILDCARD_DIM) {
    return true;
  }
  return false;
}


const std::string
DimsDebugString(const nvinfer1::Dims& dims)
{
  std::vector<int64_t> dims_vec;
  DimsToDimVec(dims, &dims_vec);
  return ShapeToString(dims_vec);
}

const std::string
DimsJsonToString(common::TritonJson::Value& dims)
{
  std::vector<int64_t> dims_vec;
  auto err = DimsJsonToDimVec(dims, &dims_vec);
  if (err != nullptr) {
    TRITONSERVER_ErrorDelete(err);
    return std::string("UNPARSABLE");
  }
  return ShapeToString(dims_vec);
}

}}}  // namespace triton::backend::tensorrt
