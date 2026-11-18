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

#include "model_connect/ranking_instance_state.hh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "model_connect/scoped_defer.hh"
#include "model_connect/trtmc_utils.hh"
#include "triton/backend/backend_common.h"

namespace triton::backend::tensorrt::model_connect {

namespace {

// Moves host inputs into score() and moves the scored tensors back out.
TRITONSERVER_Error*
RunRankingTask(
    trtmc::IStructuredRanking* ranking, std::vector<NamedTensor>& inputs,
    std::vector<NamedTensor>* outputs)
{
  // Move the host bytes in. score() returns its own buffers.
  std::vector<trtmc::FeatureTensor> features;
  features.reserve(inputs.size());
  for (NamedTensor& input : inputs) {
    trtmc::FeatureTensor feature;
    feature.name = std::move(input.name);
    feature.shape = std::move(input.shape);
    feature.bytes = std::move(input.bytes);
    // ValidateInput accepted this datatype.
    NumericType(input.datatype, &feature.dtype);
    features.push_back(std::move(feature));
  }
  std::vector<trtmc::FeatureTensor> scored = ranking->score(features);
  outputs->reserve(scored.size());
  for (trtmc::FeatureTensor& feature : scored) {
    NamedTensor output;
    output.name = std::move(feature.name);
    output.shape = std::move(feature.shape);
    output.bytes = std::move(feature.bytes);
    if (!TritonType(feature.dtype, &output.datatype)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unexpected datatype for inference output '") +
           output.name + "'")
              .c_str());
    }
    outputs->push_back(std::move(output));
  }
  return nullptr;
}

}  // namespace

struct RankingInstanceState::Metrics {
  struct TritonMetric {
    TRITONSERVER_MetricFamily* family{nullptr};
    TRITONSERVER_Metric* metric{nullptr};
    std::function<double()> read;
  };

  std::vector<TritonMetric> metrics;
  std::atomic<bool> stop{false};
  std::mutex wait_mu;
  std::condition_variable cv;
  std::thread publisher;

  ~Metrics()
  {
    {
      std::lock_guard<std::mutex> lock(wait_mu);
      stop.store(true);
    }
    cv.notify_all();
    if (publisher.joinable()) {
      publisher.join();
    }
    for (TritonMetric& metric : metrics) {
      if (metric.metric != nullptr) {
        LOG_IF_ERROR(
            TRITONSERVER_MetricDelete(metric.metric),
            "failed to delete model_connect metric");
        metric.metric = nullptr;
      }
    }
    for (TritonMetric& metric : metrics) {
      if (metric.family != nullptr) {
        LOG_IF_ERROR(
            TRITONSERVER_MetricFamilyDelete(metric.family),
            "failed to delete model_connect metric family");
        metric.family = nullptr;
      }
    }
  }
};

RankingInstanceState::RankingInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : ModelInstanceState(model_state, triton_model_instance)
{
}

RankingInstanceState::~RankingInstanceState() = default;

TRITONSERVER_Error*
RankingInstanceState::Load()
{
  RETURN_IF_ERROR(LoadSdkTask(&task_));
  ranking_ = dynamic_cast<trtmc::IStructuredRanking*>(task_.get());
  if (ranking_ == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "bundle task structured_ranking did not load IStructuredRanking");
  }
  return StartMetrics();
}

// The configuration allows only numeric inputs and Triton checks each request
// against it, so another type here is an internal error.
TRITONSERVER_Error*
RankingInstanceState::ValidateInput(
    TRITONBACKEND_Input* input, const char* name,
    const TRITONSERVER_DataType datatype, const int64_t* shape,
    const uint32_t dims) const
{
  trtmc::DType dtype = trtmc::DType::kFloat32;
  if (!NumericType(datatype, &dtype)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unexpected datatype TYPE_") +
         TRITONSERVER_DataTypeString(datatype) + " for inference input '" +
         (name != nullptr ? name : "") + "'")
            .c_str());
  }
  return nullptr;
}

TRITONSERVER_Error*
RankingInstanceState::Execute(
    std::vector<TRITONBACKEND_Request*>& requests,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  return RunRequest(
      requests, responses,
      [this](
          std::vector<NamedTensor>& inputs, std::vector<NamedTensor>* outputs) {
        return RunRankingTask(ranking_, inputs, outputs);
      });
}

TRITONSERVER_Error*
RankingInstanceState::StartMetrics()
{
  metrics_ = std::make_unique<Metrics>();

  const std::string& model = model_state_->Name();
  const std::string version = std::to_string(model_state_->Version());
  const std::string device = std::to_string(DeviceId());
  const std::string& instance = Name();
  std::vector<const TRITONSERVER_Parameter*> labels{
      TRITONSERVER_ParameterNew(
          "model", TRITONSERVER_PARAMETER_STRING, model.c_str()),
      TRITONSERVER_ParameterNew(
          "version", TRITONSERVER_PARAMETER_STRING, version.c_str()),
      TRITONSERVER_ParameterNew(
          "device", TRITONSERVER_PARAMETER_STRING, device.c_str()),
      TRITONSERVER_ParameterNew(
          "instance", TRITONSERVER_PARAMETER_STRING, instance.c_str())};
  ScopedDefer delete_labels([&labels] {
    for (const TRITONSERVER_Parameter* label : labels) {
      TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(label));
    }
    labels.clear();
  });
  for (const TRITONSERVER_Parameter* label : labels) {
    if (label == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "failed to create model_connect metric label");
    }
  }

  // Each entry is one published metric. Append here to add another.
  struct MetricSpec {
    const char* name;
    const char* description;
    double (*read)(const trtmc::CacheStats&);
  };

  // Shared definitions. Each instance creates its own gauges from this
  // table and reads that instance's cache statistics.
  static constexpr MetricSpec kMetrics[] = {
      {"hstu_kv_cache_supported", "1 when this instance has a managed KV cache",
       [](const trtmc::CacheStats& s) { return s.supported ? 1.0 : 0.0; }},
      {"hstu_kv_cache_budget_bytes", "KV cache budget in bytes",
       [](const trtmc::CacheStats& stats) {
         return static_cast<double>(stats.budget_bytes);
       }},
      {"hstu_kv_cache_reserved_bytes", "Resident KV cache storage in bytes",
       [](const trtmc::CacheStats& stats) {
         return static_cast<double>(stats.reserved_bytes);
       }},
      {"hstu_kv_cache_used_bytes",
       "KV cache bytes active for the current request",
       [](const trtmc::CacheStats& stats) {
         return static_cast<double>(stats.used_bytes);
       }},
      {"hstu_kv_cache_peak_bytes",
       "Maximum active KV cache bytes since this instance loaded",
       [](const trtmc::CacheStats& stats) {
         return static_cast<double>(stats.peak_bytes);
       }},
      {"hstu_kv_cache_active_sequences",
       "Sequences currently using the KV cache",
       [](const trtmc::CacheStats& stats) {
         return static_cast<double>(stats.active_sequences);
       }},
      {"hstu_kv_cache_allocation_failures", "KV cache allocation failures",
       [](const trtmc::CacheStats& stats) {
         return static_cast<double>(stats.allocation_failures);
       }},
  };

  bool first = true;
  for (const MetricSpec& spec : kMetrics) {
    metrics_->metrics.push_back(Metrics::TritonMetric{});
    Metrics::TritonMetric& metric = metrics_->metrics.back();
    metric.read = [ranking = ranking_, read = spec.read]() {
      return read(ranking->cache_stats());
    };
    TRITONSERVER_Error* err = TRITONSERVER_MetricFamilyNew(
        &metric.family, TRITONSERVER_METRIC_KIND_GAUGE, spec.name,
        spec.description);
    if (err != nullptr) {
      if (TRITONSERVER_ErrorCode(err) == TRITONSERVER_ERROR_UNSUPPORTED &&
          first) {
        // Metrics are not enabled in Triton.
        metrics_->metrics.clear();
        TRITONSERVER_ErrorDelete(err);
        return nullptr;
      }
      return err;
    }
    first = false;
    err = TRITONSERVER_MetricNew(
        &metric.metric, metric.family, labels.data(), labels.size());
    if (err != nullptr) {
      return err;
    }
  }

  try {
    metrics_->publisher = std::thread([this] { PollMetrics(); });
  }
  catch (const std::exception& ex) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to start model_connect metrics: ") + ex.what())
            .c_str());
  }
  return nullptr;
}

void
RankingInstanceState::PollMetrics()
{
  if (metrics_ == nullptr || metrics_->metrics.empty()) {
    return;
  }
  while (true) {
    for (const Metrics::TritonMetric& metric : metrics_->metrics) {
      if (metric.metric == nullptr) {
        continue;
      }
      LOG_IF_ERROR(
          TRITONSERVER_MetricSet(metric.metric, metric.read()),
          "failed to update model_connect metric");
    }
    std::unique_lock<std::mutex> lock(metrics_->wait_mu);
    if (metrics_->cv.wait_for(lock, std::chrono::milliseconds(10), [this] {
          return metrics_->stop.load();
        })) {
      return;
    }
  }
}

}  // namespace triton::backend::tensorrt::model_connect
