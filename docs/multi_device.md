# TensorRT Multi-Device (multi-GPU) support — TRT-28040

This backend can run a single TensorRT engine sharded across multiple GPUs using
TensorRT's Multi-Device feature (NCCL `DistCollective` operations), so one Triton
model instance drives N GPUs.

## How it works

A Multi-Device instance is a single `KIND_MODEL` instance that owns N GPUs:

- At load (`InitMultiDevice`): create N in-process NCCL communicators with
  `ncclCommInitAll`; create one `IExecutionContext` per rank (rank 0 reuses the
  instance's existing engine/context, ranks 1..N-1 get their own engine on their
  GPU); attach the communicator to every rank **concurrently** with
  `IExecutionContext::setCommunicator()` (a sequential loop deadlocks, since the
  call is a cross-rank handshake).
- At inference (`EnqueueMultiDevice`): replicate rank-0's input tensors to every
  rank, then issue `enqueueV3` on all ranks concurrently so the in-engine
  collectives rendezvous; rank 0's output is returned. Input replication uses a
  direct `cudaMemcpyPeer` when peer access is available (opt-in via
  `TRT_MD_USE_PEER`, see Notes) and otherwise a pinned host bounce buffer.

The single-GPU `KIND_GPU` path is unchanged and remains the default.

## Model configuration

```protobuf
backend: "tensorrt"
max_batch_size: 0
instance_group [ { kind: KIND_MODEL  count: 1 } ]
parameters [
  { key: "enable_multi_device" value: { string_value: "true" } },
  { key: "multi_device_gpus"   value: { string_value: "0,1" } }
]
```

The engine is sharded offline (e.g. `polygraphy multi-device shard`) and the same
plan is deserialized on every rank; the rank is distinguished at runtime by the
communicator. Requires TensorRT >= 11.0 (Multi-Device GA) and NCCL; the backend
must be built with `TRITON_ENABLE_TENSORRT_MULTI_DEVICE=ON`.

## Validation (TRT 11.0)

A context-parallel self-attention model (4096 hidden, 32 heads, FP16), sharded
with `polygraphy multi-device shard`, served as `attn_cp` (2-GPU, KIND_MODEL) and
compared against the unsharded `attn_sd` (1-GPU, KIND_GPU) on the same input.

Accuracy (2-GPU CP vs 1-GPU, identical input):

| seq    | result            |
|--------|-------------------|
| 8192   | bit-identical (max_abs_diff 0.0) |
| 16384  | correct, rel_max ~6e-3 (FP16 split-reduction) |
| 32768  | correct |
| 65536  | correct |

Latency, 1x vs 2x NVIDIA B200 (NVLink, NV18 / ~900 GB/s; batch=1, single
request). Numbers are noisy run-to-run (±10%):

| seq / model            | 1-GPU      | 2-GPU CP   | speedup (observed range) |
|------------------------|------------|------------|--------------------------|
| 16384 (hidden 4096)    | ~480–724 ms| ~495–598 ms| ~0.96–1.21x              |
| 32768 (hidden 4096)    | ~1160 ms   | ~1100 ms   | ~1.03–1.05x              |
| 65536 (hidden 4096)    | ~2224 ms   | ~2240 ms   | ~0.99–1.03x (cold ~2x)   |
| 8192  (hidden 8192)    | ~593–728 ms| ~543–791 ms| ~0.92–1.11x              |

Single-attention-layer models at batch=1 are overhead/latency-bound on B200
(per-request input replication, collective launch, kernel-launch overhead), so
the multi-GPU speedup is hidden by noise — that is why the table above is
neutral-to-modest with high variance.

**Multi-layer model (compute-bound).** Stacking 6 attention blocks (so compute
dominates the fixed per-request overhead) gives a consistent speedup on 2x B200
over NVLink at seq 32768:

| run | 1-GPU    | 2-GPU CP | speedup |
|-----|----------|----------|---------|
| 1   | 1388 ms  | 1202 ms  | 1.15x   |
| 2   | 1349 ms  | 1185 ms  | 1.14x   |
| 3   | 1209 ms  | 1102 ms  | 1.10x   |
| 4   | 1187 ms  | 1127 ms  | 1.05x   |

All runs faster, ~1.1x average, accuracy correct (rel_err 4.7e-3). The gain grows
with model depth as the fixed per-request overhead amortizes; a full transformer
(dozens of layers, larger batch) is expected to approach the ideal ~2x.

(Generator: `create_onnx_multilayer.py --layers N` builds the stacked model and a
matching per-layer CP hint.)

## Weight-sharded tensor parallelism (per-rank engines)

The default MD path loads the **same** engine on every rank (context/activation
parallel — weights replicated). For **true Megatron tensor parallelism**, set
`multi_device_per_rank_engines: "true"` and place a **distinct engine per rank**
in the version dir: `model.plan.rank0`, `model.plan.rank1`, ... Each engine holds
only that rank's weight shard (column-parallel GEMM → no comm; row-parallel GEMM
→ trailing `DistCollective` AllReduce), giving ~1/N per-GPU weight memory. (Keep a
`model.plan` symlink to `model.plan.rank0` so Triton's repository check passes.)

```protobuf
instance_group [ { kind: KIND_MODEL count: 1 } ]
parameters [
  { key: "enable_multi_device"            value: { string_value: "true" } },
  { key: "multi_device_gpus"              value: { string_value: "0,1" } },
  { key: "multi_device_per_rank_engines"  value: { string_value: "true" } }
]
```

Rank r loads `model.plan.rank{r}` (`Create()` for rank 0, `InitMultiDevice()` for
the rest); the execute path is unchanged (full input replicated to every rank,
output from rank 0 after the AllReduce). Validated on 2x A30 with a Megatron MLP:
2-GPU weight-TP output matches the single-GPU reference (rel_err ~9e-5), with each
rank's engine ~half the full-weight engine size (134 MB vs 268 MB). Engines were
built with `docs/build_tp_engines.cpp` (no polygraphy). TRT MD supports distinct
per-rank engines sharing one communicator with a matching collective (validated
standalone before wiring the backend).

## Notes / known limitations

- Direct `cudaMemcpyPeer` of TensorRT IO buffers did not transfer correctly in the
  threaded backend context on the tested systems (likely pooled/virtual device
  memory), so input replication defaults to a pinned host bounce buffer. The P2P
  path is opt-in via the `TRT_MD_USE_PEER` env var pending a fix.
- Single optimization profile per MD model (one communicator per context).
- CUDA graphs are not used on the MD path.
