// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "shared_library.h"
#include "filesystem.h"
#include "logging.h"
#include "mutex"

/// FIXME: Duplication of core/src/shared_library.cc
/// Separate shared_library to common library and delete this

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace triton { namespace backend { namespace tensorrt {

TRITONSERVER_Error*
SetLibraryDirectory(const std::string& path)
{
#ifdef _WIN32
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("SetLibraryDirectory: path = ") + path).c_str());
  if (!SetDllDirectory(path.c_str())) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        ("unable to set dll path " + path + ": " + errstr).c_str());
  }
#endif

  return nullptr;  // success
}

TRITONSERVER_Error*
ResetLibraryDirectory()
{
#ifdef _WIN32
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "ResetLibraryDirectory");
  if (!SetDllDirectory(NULL)) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        ("unable to reset dll path: " + errstr).c_str());
  }
#endif

  return nullptr;  // success
}

TRITONSERVER_Error*
OpenLibraryHandle(const std::string& path, void** handle)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("OpenLibraryHandle: ") + path).c_str());

#ifdef _WIN32
  // Need to put shared library directory on the DLL path so that any
  // dependencies of the shared library are found
  const std::string library_dir = DirName(path);
  RETURN_IF_ERROR(SetLibraryDirectory(library_dir));

  // HMODULE is typedef of void*
  // https://docs.microsoft.com/en-us/windows/win32/winprog/windows-data-types
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("OpenLibraryHandle: path = ") + path).c_str());
  *handle = LoadLibrary(path.c_str());

  // Remove the dll path added above... do this unconditionally before
  // check for failure in dll load.
  RETURN_IF_ERROR(ResetLibraryDirectory());

  if (*handle == nullptr) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        ("unable to load shared library: " + errstr).c_str());
  }
#else
  *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (*handle == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_NOT_FOUND,
        ("unable to load shared library: " + std::string(dlerror())).c_str());
  }
#endif

  return nullptr;  // success
}

TRITONSERVER_Error*
CloseLibraryHandle(void* handle)
{
  if (handle != nullptr) {
#ifdef _WIN32
    if (FreeLibrary((HMODULE)handle) == 0) {
      LPSTR err_buffer = nullptr;
      size_t size = FormatMessageA(
          FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_IGNORE_INSERTS,
          NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          (LPSTR)&err_buffer, 0, NULL);
      std::string errstr(err_buffer, size);
      LocalFree(err_buffer);
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unable to unload shared library: " + errstr).c_str());
    }
#else
    if (dlclose(handle) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          ("unable to unload shared library: " + std::string(dlerror()))
              .c_str());
    }
#endif
  }

  return nullptr;  // success
}

}}}  // namespace triton::backend::tensorrt
