cmake_minimum_required(VERSION 3.22)
project(TorchLeanLibTorch LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(FATAL_ERROR "TorchLean's LibTorch CUDA backend currently requires Linux or Windows.")
endif()
if(NOT EXISTS "${TORCHLEAN_LEAN_INCLUDE}/lean/lean.h")
  message(FATAL_ERROR "TORCHLEAN_LEAN_INCLUDE must name the pinned Lean include directory.")
endif()
if(NOT EXISTS "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Torch/TorchConfig.cmake")
  message(FATAL_ERROR "TORCHLEAN_LIBTORCH_HOME must contain share/cmake/Torch/TorchConfig.cmake.")
endif()

# Shared-vs-static. Linux always builds the shared (.so) route. Windows defaults to the
# static archive that Lake's exe links; TORCHLEAN_BUILD_SHARED=ON requests the DLL variant
# for interpreter/#eval hosts (lean.exe/lake.exe, which are shared-runtime hosts).
if(WIN32)
  set(torchlean_shared_default OFF)
else()
  set(torchlean_shared_default ON)
endif()
option(TORCHLEAN_BUILD_SHARED "Build the LibTorch backend as a DLL on Windows" ${torchlean_shared_default})
if(WIN32 AND TORCHLEAN_BUILD_SHARED AND NOT TORCHLEAN_LEAN_LIB)
  message(FATAL_ERROR "TORCHLEAN_BUILD_SHARED=ON on Windows requires TORCHLEAN_LEAN_LIB="
    "<Lean shared import-lib dir> (the toolchain 'lib/lean' with the *shared* .dll.a import libs).")
endif()

# Use the selected SDK exclusively. Torch's exported targets carry its ABI flags and dependencies.
set(Torch_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Torch")
set(Caffe2_DIR "${TORCHLEAN_LIBTORCH_HOME}/share/cmake/Caffe2")
list(PREPEND CMAKE_PREFIX_PATH "${TORCHLEAN_LIBTORCH_HOME}")
find_package(Torch REQUIRED CONFIG PATHS "${Torch_DIR}" NO_DEFAULT_PATH)
if(NOT TARGET torch_cuda)
  message(FATAL_ERROR "cuda=true requires a CUDA-enabled LibTorch SDK (torch_cuda is missing).")
endif()

get_target_property(torch_cxx_standard torch CXX_STANDARD)
# Require C++20 or a stricter standard advertised by the selected SDK.
if(NOT torch_cxx_standard OR torch_cxx_standard LESS 20)
  set(torch_cxx_standard 20)
endif()
get_target_property(torch_compile_features torch INTERFACE_COMPILE_FEATURES)
get_target_property(torch_compile_options torch INTERFACE_COMPILE_OPTIONS)
get_target_property(torch_compile_definitions torch INTERFACE_COMPILE_DEFINITIONS)
foreach(feature IN LISTS torch_compile_features)
  if(feature MATCHES "^cxx_std_([0-9]+)$" AND CMAKE_MATCH_1 GREATER torch_cxx_standard)
    set(torch_cxx_standard "${CMAKE_MATCH_1}")
  endif()
endforeach()
set(CMAKE_CXX_STANDARD "${torch_cxx_standard}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(torch_sdk_version "${Torch_VERSION}")
foreach(include_dir IN LISTS TORCH_INCLUDE_DIRS)
  if(EXISTS "${include_dir}/torch/version.h")
    file(STRINGS "${include_dir}/torch/version.h" version_line
      REGEX "^#define TORCH_VERSION \"")
    if(version_line)
      string(REGEX REPLACE "^#define TORCH_VERSION \"([^\"]+)\".*" "\\1"
        torch_sdk_version "${version_line}")
    endif()
  endif()
endforeach()

# Keep the compile/probe in this project. A try_compile scratch project cannot reliably
# re-export SDK alias targets (the NVIDIA SDK's torch_cpu_library uses protobuf::libprotobuf).
# The probe is built as a dependency of the backend and is never run.
# Linux links it as an executable to verify the SDK DLLs link; Windows consumes the backend as
# a static archive linked later by Lake's lld (GNU-mode ld.lld), and an MSVC `vs_link_exe` link
# would exercise a different toolchain and predict nothing about that link, so on Windows the
# probe is a compile + archive check only. The real Windows link check is the executable Lake
# builds: lld resolving torch import libs, CRT records, and uuid via the win-link-shim.
# (The shared DLL variant is linked by CMake itself, so the probe stays EXCLUDE_FROM_ALL there too.)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sdk_link_check.cpp" [=[
  #include <ATen/ATen.h>
  #include <c10/cuda/CUDAFunctions.h>
  #include <torch/version.h>
  #include <string>
  int main() {
    const c10::Device device(std::string("cpu"));
    auto tensor = at::zeros({1}, at::TensorOptions().device(device));
    return tensor.numel() == 1 && c10::cuda::device_count() >= 0 ? 0 : 1;
  }
]=])
if(WIN32)
  add_library(torchlean_sdk_link_check STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/sdk_link_check.cpp")
else()
  add_executable(torchlean_sdk_link_check EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/sdk_link_check.cpp")
endif()
target_link_libraries(torchlean_sdk_link_check PRIVATE ${TORCH_LIBRARIES})
separate_arguments(torch_sdk_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(torchlean_sdk_link_check PRIVATE ${torch_sdk_cxx_flags})
if(NOT WIN32)
  target_link_options(torchlean_sdk_link_check PRIVATE "LINKER:--disable-new-dtags")
endif()

if(WIN32 AND TORCHLEAN_BUILD_SHARED)
  # Shared DLL variant for #eval hosts. Exports the C ABI via WINDOWS_EXPORT_ALL_SYMBOLS
  # (CMake generates a .def from the objects' defined globals) - do NOT define
  # LEAN_EXPORTING, which would turn lean.h's *import* declarations into dllexport and
  # break the import side (re-verified; see windows_notes).
  add_library(torchlean_libtorch SHARED torchlean.cpp)
  set_target_properties(torchlean_libtorch PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
  # The DLL imports the Lean C runtime API + mimalloc from the *shared* chain,
  # exactly like torchlean_tensor_cpu.dll does (libInit_shared.dll exports them).
  target_link_libraries(torchlean_libtorch PRIVATE
    "${TORCHLEAN_LEAN_LIB}/libleanshared.dll.a"
    "${TORCHLEAN_LEAN_LIB}/libleanshared_1.dll.a"
    "${TORCHLEAN_LEAN_LIB}/libleanshared_2.dll.a"
    "${TORCHLEAN_LEAN_LIB}/libInit_shared.dll.a"
    "${TORCHLEAN_LEAN_LIB}/libLake_shared.dll.a")
elseif(WIN32)
  # Windows static archive consumed by the executable link (Lake drives it via the
  # toolchain's ld.lld, which resolves lean_*/mi_* from libleanshared.dll.a natively).
  # The static route is the proven shape for lean_exe hosts (their runtime is baked
  # in-image, so a DLL importing libInit_shared.dll would drag in a second runtime).
  add_library(torchlean_libtorch STATIC torchlean.cpp)
else()
  add_library(torchlean_libtorch SHARED torchlean.cpp)
endif()
add_dependencies(torchlean_libtorch torchlean_sdk_link_check)
target_compile_definitions(torchlean_libtorch PRIVATE TORCHLEAN_LIBTORCH)
if(WIN32)
  # clang-cl: MSYS2's stdnoreturn.h defines `noreturn` even in C++ mode, which leaks into
  # windows.h's __declspec(noreturn); skip the header via its guard, same as the Lean
  # runtime clang-cl build.
  target_compile_definitions(torchlean_libtorch PRIVATE __STDNORETURN_H)
endif()
target_compile_options(torchlean_libtorch PRIVATE ${torch_sdk_cxx_flags})
target_include_directories(torchlean_libtorch PRIVATE
  "${TORCHLEAN_LEAN_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(torchlean_libtorch PRIVATE ${TORCH_LIBRARIES})
if(WIN32)
  # The static archive must land in the directory lake resolves (torchlean_libtorch.lib).
  # MSVC static archives go to ARCHIVE_OUTPUT_DIRECTORY; the shared DLL is a "runtime"
  # artifact on Windows (RUNTIME_OUTPUT_DIRECTORY), with its import library parked
  # separately (see the TORCHLEAN_BUILD_SHARED branch) so it never shadows the static
  # archive's name.
  if(TORCHLEAN_BUILD_SHARED)
    # The DLL's import library and the static archive both want the name
    # `torchlean_libtorch.lib`; park the import library in a private subdirectory so
    # it can never clobber the static archive that the executable link resolves at
    # `TORCHLEAN_OUTPUT_DIR/torchlean_libtorch.lib`.
    set_target_properties(torchlean_libtorch PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${TORCHLEAN_OUTPUT_DIR}/shared-import"
      LIBRARY_OUTPUT_DIRECTORY "${TORCHLEAN_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${TORCHLEAN_OUTPUT_DIR}")
  else()
    set_target_properties(torchlean_libtorch PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${TORCHLEAN_OUTPUT_DIR}")
  endif()
else()
  set_target_properties(torchlean_libtorch PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${TORCHLEAN_OUTPUT_DIR}"
    # Lake links this private library by absolute path. Omitting SONAME retains that path in
    # the executable, including for downstream packages whose build directory differs.
    NO_SONAME ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH ON
  )
  # Use RPATH for transitive SDK dependencies as well (some SDK DSOs omit their own RUNPATH).
  target_link_options(torchlean_libtorch PRIVATE "LINKER:--disable-new-dtags")
endif()

file(WRITE "${CMAKE_BINARY_DIR}/sdk.txt"
  "Torch_VERSION=${torch_sdk_version}
Torch_DIR=${Torch_DIR}
TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}
TORCH_COMPILE_FEATURES=${torch_compile_features}
TORCH_COMPILE_OPTIONS=${torch_compile_options}
TORCH_COMPILE_DEFINITIONS=${torch_compile_definitions}
TORCH_LIBRARIES=${TORCH_LIBRARIES}
CXX_STANDARD=${CMAKE_CXX_STANDARD}
CXX_COMPILER=${CMAKE_CXX_COMPILER}
CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}
CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}
")
message(STATUS "TorchLean: LibTorch ${torch_sdk_version}, C++${CMAKE_CXX_STANDARD}")
message(STATUS "TorchLean SDK flags: ${TORCH_CXX_FLAGS}; ${torch_compile_options}")
