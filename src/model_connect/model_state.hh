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
#include "trtmc/types.h"

namespace triton::backend::tensorrt::model_connect {

// One named tensor. A model configuration entry sets only name and datatype.
// A request copy or a packed output also sets shape and bytes. A text result
// borrows its data from the task result instead of copying it into bytes.
struct NamedTensor {
  std::string name;
  TRITONSERVER_DataType datatype{TRITONSERVER_TYPE_INVALID};
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
  // Borrowed TYPE_BYTES output, one view per element with no length prefix.
  std::vector<trtmc_string_view> strings;
  // Borrowed numeric host data, used instead of bytes when external_keep is
  // set. external_size is its byte count.
  const void* external{nullptr};
  uint64_t external_size{0};
  // Owns the result that strings or external point into until the response is
  // written. For a numeric type, non-null means the data is in external, not
  // bytes.
  std::shared_ptr<void> external_keep;
};

// Additional parameters in model configuration.
struct Parameters {
  uint64_t kv_cache_size_bytes{0};
  double kv_cache_memory_ratio{0};

  // Unsupported parameters
  // bool kv_cache_hydration_enabled{false};
  // uint64_t kv_cache_max_entries{0};
  // double kv_cache_ttl_seconds{0};
  // double kv_cache_lease_seconds{0};
};

// A TensorRT Model Connect bundle. The bundle header records the family and
// the task. The task selects the implementation. Inputs and outputs are the
// tensors named in the model configuration. structured_ranking with
// max_batch_size greater than 0 scores the requests of one execute together.
// text_continuation runs them in one batch_text_continuation call.
// text_generation runs one request at a time.
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  const std::string& BundlePath() const { return bundle_path_; }
  const std::string& BundleTask() const { return bundle_task_; }
  const std::string& BundleFamily() const { return bundle_family_; }
  const struct Parameters& Parameters() const { return params; }
  const std::vector<NamedTensor>& Inputs() const { return inputs_; }
  const std::vector<NamedTensor>& Outputs() const { return outputs_; }

 private:
  explicit ModelState(TRITONBACKEND_Model* triton_model);

  // Read the bundle header and record the configured tensors.
  TRITONSERVER_Error* ValidateModelConfig();
  TRITONSERVER_Error* ValidateTensors();
  TRITONSERVER_Error* ParseParameters();

  std::string bundle_path_;
  std::string bundle_task_;
  std::string bundle_family_;
  struct Parameters params;
  std::vector<NamedTensor> inputs_;
  std::vector<NamedTensor> outputs_;
};

}  // namespace triton::backend::tensorrt::model_connect
