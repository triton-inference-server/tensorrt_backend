<!--
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue for a suspected security vulnerability.**

To report a potential security vulnerability in any NVIDIA product, use one of
the following channels:

1. **NVIDIA Vulnerability Disclosure Program** (preferred):
   <https://www.nvidia.com/en-us/security/>
2. **Email** [psirt@nvidia.com](mailto:psirt@nvidia.com). Please encrypt the
   message with NVIDIA's public PGP key:
   <https://www.nvidia.com/en-us/security/pgp-key>
3. **GitHub Private Vulnerability Reporting** through the Security tab of this
   repository, where enabled.

**OEM partners should contact their NVIDIA Customer Program Manager.**

Please include:

1. Product name and version or branch that contains the vulnerability
2. Type of vulnerability (for example code execution, denial of service,
   memory corruption)
3. Step-by-step instructions to reproduce the issue
4. Proof-of-concept or exploit code, if available
5. Potential impact, including how an attacker could exploit the issue

NVIDIA PSIRT acknowledges reports, assesses them, coordinates a fix and
disclosure timeline with the reporter, and publishes security bulletins at
<https://www.nvidia.com/en-us/security/>.

## Security Architecture and Context

**Project:** the Triton Inference Server backend for NVIDIA TensorRT. It is a
C++ shared library (`libtriton_tensorrt.so`) that is loaded in-process by
Triton Inference Server and executes serialized TensorRT engines (plan files)
on NVIDIA GPUs through the TensorRT C++ API.

**Software classification:** Library (a Triton backend plugin). It exposes no
network listener, command-line tool or authentication mechanism of its own.

**Primary security responsibility:** safely deserializing and executing model
engines supplied by the Triton model repository, and moving request and
response tensors between Triton-managed memory and GPU memory.

**Key interfaces and boundaries:**

- The Triton backend API (`TRITONBACKEND_*`), through which the server passes
  model configuration, backend configuration and inference requests.
- Model repository content: the serialized engine, the model configuration and
  any TensorRT plugin libraries.
- Backend command-line options set by the server operator
  (`--backend-config=tensorrt,...`), including `plugins` and
  `version-compatible`.
- The TensorRT, CUDA and (optionally) NCCL runtime libraries, which run in the
  same process.

Authentication, authorization, transport security and request validation at
the network edge are the responsibility of Triton Inference Server and the
deployment environment, not of this backend.

**Repository Exposure Classification:** Public.

**Service Exposure Classification:** Internal-Sensitive (low confidence). The
backend runs inside a serving process whose exposure is decided by the
deployer. This is an informal classification for this document and not an
official NVIDIA label.

## Threat Model

1. **Malicious or tampered model engine:** an attacker who can write to the
   model repository supplies a crafted serialized engine. Deserialization in
   `src/loader.cc` (`deserializeCudaEngine`) parses untrusted binary data in
   the server process and may trigger memory-safety faults in TensorRT.
2. **Code execution through version-compatible engines:** when the
   `version-compatible` option is enabled, the backend allows engine host code
   (`setEngineHostCodeAllowed` in `src/loader.cc`). A model then carries a lean
   runtime that is deserialized and executed in-process, so a malicious model
   can run arbitrary code with the server's privileges.
3. **Malicious TensorRT plugin library:** libraries named by the `plugins`
   backend option are opened with `dlopen` (`src/shared_library.cc`,
   `src/tensorrt.cc`). A library at an attacker-controlled path or in a
   writable location executes code at load time.
4. **Malformed inference requests:** crafted tensor shapes, shape-tensor values
   or optimization-profile selections (`src/instance_state.cc`,
   `src/shape_tensor.cc`) could cause out-of-bounds buffer access, excessive
   GPU or host allocation, or denial of service.
5. **Cross-request data exposure on shared GPU memory:** request inputs and
   outputs are staged through shared pinned and device buffers, CUDA graphs
   and output allocators (`src/output_allocator.cc`). Incorrect buffer reuse
   could expose data from one request to another.
6. **Resource exhaustion:** many model instances, large dynamic shapes, CUDA
   graph capture for many shape combinations and multi-device sharding can
   exhaust GPU memory or serialize on shared devices, degrading availability
   for other models served by the same process.

## Critical Security Assumptions

- The model repository is trusted. The backend does not verify the origin,
  signature or integrity of engine files, model configuration or plugin
  libraries.
- Operators enable `version-compatible` and `plugins` only for models and
  libraries they trust, since both result in code execution in the server
  process.
- Plugin library paths are read-only to untrusted users.
- Triton Inference Server, or a proxy in front of it, authenticates and
  authorizes clients, enforces TLS and applies request size and rate limits.
- The server process is isolated, for example by running as an unprivileged
  user in a container with restricted GPU and file access, because model code
  runs with its privileges.
- TensorRT, CUDA and NCCL are kept current. Memory-safety issues inside those
  libraries are outside the scope of this repository and should be reported to
  NVIDIA PSIRT through the channels above.
- GPU memory is not a confidentiality boundary between models served by the
  same process.
