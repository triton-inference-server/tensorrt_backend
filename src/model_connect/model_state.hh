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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "triton/backend/backend_model.h"

namespace triton::backend::tensorrt::model_connect {

// One named tensor. The model configuration records the name and data type.
// TYPE_STRING is stored as TYPE_BYTES. A request copy records the shape and
// host bytes. A text result may instead borrow external, which external_keep
// owns: for TYPE_BYTES that is the string and external_size is its character
// count, and for a numeric type external_size is the byte count.
struct NamedTensor {
  std::string name;
  TRITONSERVER_DataType datatype{TRITONSERVER_TYPE_INVALID};
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
  const void* external{nullptr};
  uint64_t external_size{0};
  std::shared_ptr<void> external_keep;
};

// A TensorRT Model Connect bundle. The bundle header records the family and
// the task. The task selects the implementation. Inputs and outputs are the
// tensors named in the model configuration. Requests are not combined.
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  const std::string& BundlePath() const { return bundle_path_; }
  const std::string& BundleTask() const { return bundle_task_; }
  const std::string& BundleFamily() const { return bundle_family_; }
  const std::vector<NamedTensor>& Inputs() const { return inputs_; }
  const std::vector<NamedTensor>& Outputs() const { return outputs_; }

 private:
  explicit ModelState(TRITONBACKEND_Model* triton_model);

  // Read the bundle header and record the configured tensors.
  TRITONSERVER_Error* ValidateModelConfig();
  TRITONSERVER_Error* ValidateTensors();

  std::string bundle_path_;
  std::string bundle_task_;
  std::string bundle_family_;
  std::vector<NamedTensor> inputs_;
  std::vector<NamedTensor> outputs_;
};

}  // namespace triton::backend::tensorrt::model_connect
