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

// Offline TP weight-shard + engine builder (TRT-28040 follow-on).
// Builds a Megatron MLP (Y = (X*W1)*W2) and serializes:
//   model_sd.plan         - full weights, single device, no collective (reference)
//   model.plan.rank{r}    - rank r's weight shards + AllReduce (true tensor parallel)
// W1 is column-parallel (split on output dim F); W2 is row-parallel (split on
// input dim F) with a trailing AllReduce, so each rank holds ~1/world of the
// weights. Same deterministic weights across SD and the TP shards.
//
//   nvcc -std=c++17 -ccbin g++ -w build_tp_engines.cpp -o build_tp_engines \
//        -I$TRT/include -L$TRT/lib -lnvinfer
//   LD_LIBRARY_PATH=$TRT/lib ./build_tp_engines <world> <out_dir>
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
using namespace nvinfer1;

#define PREVIEW_MD 1  // TRT 10.16 needs the preview flag; remove for TRT 11.0

class Logger : public ILogger {
  void log(Severity s, const char* m) noexcept override
  { if (s <= Severity::kERROR) fprintf(stderr, "[TRT] %s\n", m); }
} gLogger;

constexpr int M = 2048, K = 4096, F = 8192, N = 4096;  // seq, hidden, inter, out
static std::vector<float> W1, W2;  // K*F, F*N (deterministic)

static void initWeights() {
  W1.resize((size_t)K * F); W2.resize((size_t)F * N);
  for (size_t i = 0; i < W1.size(); ++i) W1[i] = 0.01f * ((int)(i % 17) - 8) / std::sqrt((float)K);
  for (size_t i = 0; i < W2.size(); ++i) W2[i] = 0.01f * ((int)(i % 13) - 6) / std::sqrt((float)F);
}

static std::vector<__half> toHalf(const float* p, size_t n) {
  std::vector<__half> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = __float2half(p[i]);
  return h;
}

static void writeFile(const std::string& path, IHostMemory* m) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(m->data()), m->size());
  printf("wrote %s (%zu bytes)\n", path.c_str(), m->size());
}

// Build an MLP engine. If world==1: full W1[K,F], W2[F,N], no collective.
// Else: rank's column shard W1[:, r*Fr:(r+1)*Fr], row shard W2[r*Fr:(r+1)*Fr, :],
// + AllReduce(SUM, nbRanks=world).
static std::vector<char> buildEngine(int world, int rank) {
  const int Fr = (world == 1) ? F : F / world;
  const int f0 = (world == 1) ? 0 : rank * Fr;
  std::vector<float> w1(K * Fr), w2(Fr * N);
  for (int k = 0; k < K; ++k)
    for (int f = 0; f < Fr; ++f) w1[k * Fr + f] = W1[(size_t)k * F + (f0 + f)];
  for (int f = 0; f < Fr; ++f)
    for (int n = 0; n < N; ++n) w2[(size_t)f * N + n] = W2[(size_t)(f0 + f) * N + n];

  std::unique_ptr<IBuilder> builder(createInferBuilder(gLogger));
  std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(
      1U << (uint32_t)NetworkDefinitionCreationFlag::kSTRONGLY_TYPED));
  ITensor* x = net->addInput("X", DataType::kFLOAT, Dims2{M, K});
  auto* c1 = net->addConstant(Dims2{K, Fr}, Weights{DataType::kFLOAT, w1.data(), (int64_t)w1.size()});
  auto* h = net->addMatrixMultiply(*x, MatrixOperation::kNONE, *c1->getOutput(0), MatrixOperation::kNONE);
  auto* c2 = net->addConstant(Dims2{Fr, N}, Weights{DataType::kFLOAT, w2.data(), (int64_t)w2.size()});
  auto* p = net->addMatrixMultiply(*h->getOutput(0), MatrixOperation::kNONE, *c2->getOutput(0), MatrixOperation::kNONE);
  ITensor* y = p->getOutput(0);
  if (world > 1) {
    auto* coll = net->addDistCollective(*y, CollectiveOperation::kALL_REDUCE, ReduceOperation::kSUM, -1, nullptr, 0);
    coll->setNbRanks(world);
    y = coll->getOutput(0);
  }
  y->setName("Y");
  net->markOutput(*y);

  std::unique_ptr<IBuilderConfig> cfg(builder->createBuilderConfig());
#if PREVIEW_MD
  if (world > 1) cfg->setPreviewFeature(PreviewFeature::kMULTIDEVICE_RUNTIME_10_16, true);
#endif
  std::unique_ptr<IHostMemory> ser(builder->buildSerializedNetwork(*net, *cfg));
  if (!ser) { fprintf(stderr, "build failed world=%d rank=%d\n", world, rank); std::abort(); }
  const char* d = static_cast<const char*>(ser->data());
  std::vector<char> v(d, d + ser->size());
  return v;
}

int main(int argc, char** argv) {
  int world = (argc > 1) ? atoi(argv[1]) : 2;
  std::string dir = (argc > 2) ? argv[2] : ".";
  initWeights();
  // Single-device reference (full weights, no collective)
  { auto e = buildEngine(1, 0);
    std::ofstream f(dir + "/model_sd.plan", std::ios::binary); f.write(e.data(), e.size());
    printf("wrote %s/model_sd.plan (%zu bytes, full weights)\n", dir.c_str(), e.size()); }
  // Per-rank TP engines
  for (int r = 0; r < world; ++r) {
    auto e = buildEngine(world, r);
    std::ofstream f(dir + "/model.plan.rank" + std::to_string(r), std::ios::binary);
    f.write(e.data(), e.size());
    printf("wrote %s/model.plan.rank%d (%zu bytes, weight shard)\n", dir.c_str(), r, e.size());
  }
  printf("dims: X[%d,%d] -> Y[%d,%d], F=%d split across %d ranks (Fr=%d)\n", M, K, M, N, F, world, F/world);
  return 0;
}
