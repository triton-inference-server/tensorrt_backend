<!--
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# TensorRT Backend

The Triton backend for [TensorRT](https://github.com/NVIDIA/TensorRT). 
You can learn more about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems on the [issues
page](https://github.com/triton-inference-server/server/issues).
This backend is designed to run a serialized [TensorRT engine](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#build_engine_c)
models using the TensorRT C++ API.

Where can I ask general questions about Triton and Triton backends?
Be sure to read all the information below as well as the [general
Triton documentation](https://github.com/triton-inference-server/server#triton-inference-server)
available in the main [server](https://github.com/triton-inference-server/server)
repo. If you don't find your answer there you can ask questions on the
main Triton [issues page](https://github.com/triton-inference-server/server/issues).

## Command-line Options

The command-line options configure properties of the TensorRT
backend that are then applied to all models that use the backend.

##### --backend-config=tensorrt,coalesce-request-input=\<boolean\>,plugins="/path/plugin1.so;/path2/plugin2.so"

The coalesce-request-input flag instructs TensorRT to consider the requests' inputs with the same name as
one contiguous buffer if their memory addresses align with each other.
This option should only be enabled if all requests' input tensors are allocated
from the same memory region. Default value is false.

The plugins flag provides a way to load any custom TensorRT plugins that your models rely on. If you have
multiple plugins to load, use a semicolon as the delimiter.

## Build the TensorRT Backend

Appropriate version of TensorRT must be installed on the system. Check the support matrix to find the correct version of TensorRT to be installed.

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]
