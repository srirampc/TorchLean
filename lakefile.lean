/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lake
import Lake.Util.Proc
open Lake DSL
open System

/-!
This lakefile is organized in the following layers, each depending only on the ones above it:

1. **Config options** — parsing and validation of the `-K` options.
2. **Windows native linking** — host directory resolution, the link shim, and the
   platform-aware compile/link flag functions.
3. **Native build helpers** — compiler discovery/tracing and the job-building functions
   for the CUDA backends and the packed-CPU-tensor runtime.
4. **LibTorch bridge** — helpers for the optional LibTorch SDPA backend library
   (shared on Linux, static on Windows — see the `torchlean_libtorch_sdpa` target).
5. **Target declarations** — thin wiring only: no logic beyond fetching and sequencing.

## Windows native linking

On Windows, a PE DLL/executable must resolve every imported symbol at *link time*,
bound to a static library or to the import library of a named DLL — there is no
"resolve from whatever is loaded in the process" fallback like ELF's global symbol
scope. Furthermore:

* Everything the native objects reference must appear on the executable's
  link line: the CUDA and the LibTorch import libraries (if enabled).
* No `rpath` on Windows: the loader finds `libleanshared.dll`, `torch.dll`,
  `cudart64_*.dll`, etc. through `PATH` at run time, so no runtime search flags
  can be passed at link time.
-/

/-! ## Config options -/

/-- Whether Lake should compile the native CUDA sources instead of the portable C stubs. -/
private def cudaEnabled : Bool :=
  let value := (get_config? cuda).getD "false"
  value == "true" || value == "1"

/-- CUDA toolkit root used for includes, libraries, and runtime search paths. -/
private def cudaHome : String := Id.run do
  let home := ((get_config? cuda_home).getD "").trimAscii.toString
  if home.startsWith "-" then
    panic! s!"cuda_home must be a path, not an option-like value: {home}"
  return if home.isEmpty then "/usr/local/cuda" else home

/--
GPU architectures included in the native CUDA objects.

`all-major` includes native code for each supported major architecture and works on headless
builders. Set `cuda_arch=sm_80`, for example, to build specifically for A100. `native` is rejected:
nvcc otherwise falls back to its default target when no GPU is visible, and the same setting could
produce different objects on different build machines.
-/
private def cudaArch : String := Id.run do
  let value := ((get_config? cuda_arch).getD "all-major").trimAscii.toString
  let arch := if value.isEmpty then "all-major" else value
  if arch == "native" then
    panic! "cuda_arch=native is not supported; use all-major or an explicit target such as sm_80"
  if arch == "all-major" || arch == "all" then
    return arch
  let suffix := (arch.drop 3).toString
  let digits := if suffix.endsWith "a" || suffix.endsWith "f" then
    (suffix.dropEnd 1).toString else suffix
  if arch.startsWith "sm_" && !digits.isEmpty && digits.toList.all Char.isDigit then
    return arch
  panic! s!"invalid cuda_arch: {arch}; expected all-major, all, or a real target such as sm_80"

/-- Optional LibTorch root; relative paths are resolved against the package directory. -/
private def libtorchHomeConfig : Option String :=
  (get_config? libtorch_home).bind fun path =>
    let path := path.trimAscii.toString
    if path.isEmpty then none else some path

/-- Whether to build the optional LibTorch-backed backend capsules. -/
private def libtorchEnabled : Bool :=
  let value := (get_config? libtorch).getD "false"
  value == "true" || value == "1"

/-! ## Windows native linking -/

/-- Validate a required `-K` directory option for Windows CUDA builds.

Reports the failure through Lake's error channel (`JobM`), so the message is attached to
the failing target and printed by Lake itself. An `IO`-level `eprintln` + `Process.exit`
from a worker thread would kill the process before the stderr text is flushed. -/
private def windowsRequiredDir (opt desc eg : String) (cfg : Option String) : JobM String := do
  match cfg with
  | some p =>
      let d := p.trimAscii.toString
      if d.isEmpty || d.startsWith "-" then
        error s!"`-K {opt}=...` must be a directory path ({desc}), got: {d}"
      else
        pure d
  | none =>
      error s!"CUDA builds on Windows require `-K {opt}=...` ({desc}), e.g. `-K {opt}={eg}`"

/-- Normalize and lightly validate a CUDA toolkit root passed through `-K cuda_home=...`. -/
private def cleanCudaHome (p : String) : String :=
  let h := p.trimAscii.toString
  if h.isEmpty then
    "/usr/local/cuda"
  else if h.startsWith "-" then
    panic! s!"cuda_home must be a path, not an option-like value: {h}"
  else
    h

/-- Resolve and validate all three directories required for Windows CUDA builds
`(cuda_home, msvc_lib_dir, msys2_lib_dir)`, failing with a clear message when any
of these three `-K` options is missing or malformed. Runs in `JobM` so the failure
is reported as a normal Lake error instead of a silent process exit. -/
private def windowsResolveDirs : JobM (String × String × String) := do
  let cudaHome ←
    match get_config? cuda_home with
    | some p => pure (cleanCudaHome p)
    | none =>
        windowsRequiredDir "cuda_home" "CUDA toolkit root"
          "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" none
  let msvcLibDir ←
    windowsRequiredDir "msvc_lib_dir" "MSVC x64 library directory"
      "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.51.36231/lib/x64"
      (get_config? msvc_lib_dir)
  let msys2LibDir ←
    windowsRequiredDir "msys2_lib_dir" "MSYS2 MinGW library directory"
      "C:/msys64/mingw64/lib"
      (get_config? msys2_lib_dir)
  pure (cudaHome, msvcLibDir, msys2LibDir)

/-- Directory populated by `windowsPrepareCudaLink` with copies of Windows
libraries that nvcc's MSVC-compiled host objects reference via `/DEFAULTLIB:`. -/
private def windowsLinkLibsDir : FilePath :=
  __dir__ / ".lake" / "build" / "win-link-shim"

/-- Names of the libraries copied into `windowsLinkLibsDir`.

Currently: `LIBCMT.lib`, `libcpmt.lib`, `libvcruntime.lib` (MSVC C/C++ runtime),
`OLDNAMES.lib` (old POSIX names), and `libuuid.a` from MSYS2 (Windows SDK GUID
definitions). The UCRT static library (`libucrt.lib`) is absent since it is provided
by Lean itself. -/
private def windowsLinkLibNames (msvcDir msys2Dir : String) : Array FilePath :=
  #["LIBCMT.lib", "libcpmt.lib", "OLDNAMES.lib", "libvcruntime.lib"].map
    (fun f => FilePath.mk msvcDir / FilePath.mk f)
  ++ #[FilePath.mk msys2Dir / "libuuid.a"]

/-- Validate the Windows host directories and populate `windowsLinkLibsDir`.

Runs inside a `JobM` (from the `torchlean_cuda_compiler` target), so the failure aborts
the build before `nvcc` or the linker emit their more obscure diagnostics. The resolved
directory strings are traced so that changing any of them re-runs the validation and the
shim re-copy. -/
private def windowsPrepareCudaLink : JobM Unit := do
  if Platform.isWindows then
    let (cudaHome, msvcLibDir, msys2LibDir) ← windowsResolveDirs
    addPureTrace (cudaHome, msvcLibDir, msys2LibDir) "windows CUDA link directories"
    let mut missing : Array String := #[]
    for (optName, dir) in #[("cuda_home", cudaHome), ("msvc_lib_dir", msvcLibDir),
        ("msys2_lib_dir", msys2LibDir)] do
      unless (← FilePath.pathExists dir) do
        missing := missing.push s!"-K {optName} (directory does not exist: {dir})"
    unless missing.isEmpty do
      error s!"TorchLean CUDA build on Windows: required directories do not exist:\n  {String.intercalate "\n  " missing.toList}"
    IO.FS.createDirAll windowsLinkLibsDir
    -- Copying is unconditional since the files are only a few MB.
    let mut missingLibs : Array String := #[]
    for src in windowsLinkLibNames msvcLibDir msys2LibDir do
      if (← src.pathExists) then
        copyFile src (windowsLinkLibsDir / src.fileName.getD src.toString)
      else
        missingLibs := missingLibs.push src.toString
    unless missingLibs.isEmpty do
      error s!"TorchLean CUDA build on Windows: required libraries do not exist:\n  {String.intercalate "\n  " missingLibs.toList}"

/-! ## Pure compile/link flag functions -/

/-- CUDA toolkit include directory. -/
private def cudaIncludes : Array String :=
  #["-I", s!"{cudaHome}/include"]

/-- Include paths shared by the CUDA implementations and the portable C stubs. -/
private def nativeIncludeArgs (pkg : Package) : Array String :=
  #[
    "-I", (pkg.dir / "csrc/cuda/common").toString,
    "-I", (pkg.dir / "csrc/cuda/conv_pool").toString
  ]

/-- Compile flags for a CUDA backend object (nvcc).

These are the traced arguments: changing the target or compiler must invalidate the object.
Windows differs from Linux in two ways: MSVC (nvcc's mandatory host compiler) rejects
`-fPIC`, and its frontend rejects the `_Noreturn` C keyword, which is mapped to C++'s
`[[noreturn]]` via a macro. -/
private def cudaCompileFlags : Array String :=
  -- MSVC (nvcc's mandatory host compiler on Windows) has no `-fPIC`.
  let picArgs := if Platform.isWindows then #[] else #["-Xcompiler", "-fPIC"]
  -- `_Noreturn` is a C keyword rejected by nvcc's frontend, so map it to C++ `[[noreturn]]`.
  let msvcArgs := if Platform.isWindows then #["-D_Noreturn=[[noreturn]]"] else #[]
  #["--std=c++17", "-O2", s!"--gpu-architecture={cudaArch}"] ++ picArgs ++ msvcArgs

/-- C++ compile flags for LibTorch C++ sources (g++ on Linux; clang-cl on Windows). -/
private def torchBridgeCompileFlags (pkg : Package) (lean : LeanInstall) (lt : String) : Array String :=
  let compileArgs :=
    if Platform.isWindows then
      -- Only clang-cl works for both the libtorch MSVC ABI and Lean's FFI.
      -- LibTorch requires C++20: its headers use C++20 features unconditionally
      -- (`std::strong_ordering`, `string_view::starts_with`, `requires`).
      -- `/EHa` enables C++ exceptions plus SEH (`__try/__except`): the FFI entries are
      -- wrapped in an SEH boundary because this exe mixes MinGW/Itanium (Lean runtime)
      -- and MSVC-ABI (clang-cl bridge + torch DLLs) exception handling, and a torch
      -- `c10::Error` thrown at the boundary can escape the C++ catch handlers and abort.
      -- `-D__STDNORETURN_H` predefines the include guard of clang's `stdnoreturn.h`,
      -- whose `#define noreturn _Noreturn` is active even in C++ mode.
      #["-O2", "/std:c++20", "/EHa", "-D__STDNORETURN_H"]
    else
      #["-O2", "-fPIC", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"]
  #[
    "-I", lean.includeDir.toString,
    "-I", s!"{pkg.dir}/csrc/cuda/common"
  ] ++ cudaIncludes ++ #[
    "-I", s!"{lt}/include",
    "-I", s!"{lt}/include/torch/csrc/api/include"
  ] ++ compileArgs

/-- LibTorch library names, in link order. -/
private def libtorchLibs : Array String :=
  #["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda"]

/-- CUDA toolkit import libraries for the executable link. -/
private def cudaLinkArgs : Array String :=
  if Platform.isWindows then
    -- CUDA's Windows toolkit keeps import libraries under `lib/x64`.
    #[
      "-L", s!"{cudaHome}/lib/x64",
      "-lcudart", "-lcublas", "-lcufft",
      "-L", windowsLinkLibsDir.toString
    ]
  else
    #[
      "-L", s!"{cudaHome}/lib64", "-lcudart", "-lcublas", "-lcufft",
      "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
    ]

/-- Link flags for the LibTorch SDPA shared library (Linux only; on Windows the bridge
is a static archive whose LibTorch dependencies are named in `torchBridgeExeLinkArgs`). -/
private def torchBridgeLinkArgs (lt : String) : Array String :=
  #[
    "-L", s!"{lt}/lib",
    "-Wl,--no-as-needed"
  ] ++ libtorchLibs ++ #[
    "-L", s!"{cudaHome}/lib64", "-lcudart",
    "-lstdc++",
    "-Wl,-rpath," ++ s!"{lt}/lib",
    "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
  ]

/-- LibTorch flags for the *executable* link. Platform-asymmetric: On
Windows the bridge is a static archive (`.a`), so it must name the import
libraries itself; on Linux the bridge `.so` carries its own dependencies and the
executable only needs the rpath to find it. -/
private def torchBridgeExeLinkArgs (lt : String) : Array String :=
  if Platform.isWindows then
    -- `libvcruntime.lib` supplies the MSVC C++-EH support the clang-cl-compiled bridge
    -- object references (`__CxxFrameHandler4`, `__std_terminate`, `__uncaught_exceptions`)
    -- without emitting a `/DEFAULTLIB:` record for it.
    #["-L", s!"{lt}/lib"] ++ libtorchLibs ++ #["-l:libvcruntime.lib"]
  else
    #["-Wl,-rpath," ++ s!"{lt}/lib"]

/-- Native link flags selected by the `cuda`/`libtorch` Lake options. -/
private def nativeLinkArgs : Array String :=
  if cudaEnabled then
    let lt := libtorchHomeConfig.getD "libtorch"
    cudaLinkArgs ++ (if libtorchEnabled then torchBridgeExeLinkArgs lt else #[]) ++
      if Platform.isWindows || Platform.isOSX then #[] else #["-lstdc++"]
  else if Platform.isWindows || Platform.isOSX then
    -- Windows and macOS provide libm via the default C runtime
    #[]
  else
    -- CPU stubs call functions from `math.h`; Linux keeps these in `libm`.
    -- Keep libstdc++ for mixed native objects when switching between CPU and CUDA builds.
    #["-lm", "-lstdc++"]

package TorchLean where
  buildDir := FilePath.mk ((get_config? torchleanBuildDir).getD ".lake/build")
  version := v!"0.1.0"
  description := "Neural network specification, execution, and verification in Lean 4."
  keywords := #["machine-learning", "neural-networks", "verification", "autograd", "cuda"]
  homepage := "https://lean-dojo.github.io/TorchLean/"
  license := "MIT"
  readmeFile := "README.md"
  testDriver := "nn_tests_suite"
  lintDriver := "torchlean_lint"
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩,
    ⟨`warningAsError, true⟩]
  dynlibs := #[`@/torchlean_tensor_cpu_shared]
  moreLinkArgs := nativeLinkArgs

/-!
## Native backend libraries

TorchLean has a small amount of native code behind Lean `extern` declarations. Each component has
the same build shape: compile the CUDA implementation when the package is built with
`-K cuda=true`; otherwise compile the matching C stub so the Lean package still builds on machines
without a CUDA toolkit.
-/

/-! ## Native build helpers -/

/-- Track project-owned native headers so a header-only edit invalidates every dependent object. -/
private def nativeHeaderDeps (pkg : Package) : SpawnM (Job Unit) := do
  let isHeader := fun path : FilePath => path.extension == some "h"
  let common ← inputDir (pkg.dir / "csrc/cuda/common") true isHeader
  let convPool ← inputDir (pkg.dir / "csrc/cuda/conv_pool") true isHeader
  pure <| common.mix convPool

/--
Find the executable that will be passed to a native compile or link command.
Keep its invocation path: names such as `clang++` can select a language mode even when they are
symlinks to the same binary. The dependency trace separately records the resolved file.
-/
private def nativeCompilerPath (name : String) : JobM FilePath := do
  let path := FilePath.mk name
  let cwd ← IO.currentDir
  -- Windows executables always carry a `.exe` extension, and `pathExists` is a plain
  -- filesystem check (no `PATHEXT` fallback), so a bare `cc`/`g++`/`clang-cl` on PATH
  -- must also be probed with the suffix appended.
  let names : List FilePath :=
    if Platform.isWindows && !path.toString.toLower.endsWith ".exe" then
      [path, FilePath.mk (path.toString ++ ".exe")]
    else
      [path]
  if path.isAbsolute || path.components.length > 1 then
    for nm in names do
      let candidate := cwd / nm
      if (← candidate.pathExists) && !(← candidate.isDir) then
        return candidate.normalize
  else
    for dir in SearchPath.parse ((← IO.getEnv "PATH").getD "") do
      for nm in names do
        let candidate := cwd / dir / nm
        if (← candidate.pathExists) && !(← candidate.isDir) then
          return candidate.normalize
  error s!"native compiler not found: {name}"

/-- Hash tool contents on each build, including replacements installed at the same path. -/
private def traceNativeTool (path : FilePath) : JobM Unit := do
  let resolved ← IO.FS.realPath path
  addPureTrace (path.toString, resolved.toString) "native tool paths"
  addTrace (.ofHash (← computeFileHash resolved) resolved.toString)

/-- Resolve and trace a native compiler before checking whether its object can be reused. -/
private def nativeCompilerJob (name : String) : SpawnM (Job FilePath) := Job.async do
  let compiler ← nativeCompilerPath name
  traceNativeTool compiler
  return compiler

/--
Record nvcc's extra flags without allowing them to replace the traced target or host compiler.
Option files and response files could hide those overrides, so pass ordinary flags directly.
-/
private def traceCudaFlags : JobM Unit := do
  let selectors := #[
    "-arch", "--gpu-architecture", "-code", "--gpu-code", "-gencode", "--generate-code",
    "-ccbin", "--compiler-bindir", "-optf", "--options-file", "@"
  ]
  for name in #["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"] do
    let value := (← IO.getEnv name).getD ""
    if selectors.any (fun selector => (value.splitOn selector).length > 1) then
      error <| s!"{name} cannot select architectures, host compilers, or option files; " ++
        "use cuda_arch and NVCC_CCBIN for the target and host compiler"
    addPureTrace value name

/--
One dependency shared by the four native CUDA components. The Lean revision alone does not
identify nvcc or its host compiler. Record the actual tools, toolkit version, and environment
flags before Lake decides whether a previously compiled object can be reused. On Windows this
also validates the mandatory `-K cuda_home/msvc_lib_dir/msys2_lib_dir` directories and
(re)populates the link shim before any nvcc compile or link runs.
-/
target torchlean_cuda_compiler pkg : FilePath × FilePath := Job.async do
  if Platform.isWindows then windowsPrepareCudaLink
  let home := pkg.dir / cudaHome
  let tool (name : String) := if Platform.isWindows then name ++ ".exe" else name
  let nvcc := home / "bin" / tool "nvcc"
  let defaultHost := if Platform.isWindows then "cl.exe" else "g++"
  let configured := ((← IO.getEnv "NVCC_CCBIN").getD "").trimAscii.toString
  let hostName := if configured.isEmpty then defaultHost else configured
  let hostPath := FilePath.mk hostName
  let hostName ← if ← hostPath.isDir then
    pure (hostPath / defaultHost).toString else pure hostName
  let host ← nativeCompilerPath hostName
  traceCudaFlags
  traceNativeTool nvcc
  traceNativeTool host
  traceNativeTool (home / "bin" / tool "ptxas")
  -- Toolkit packages can update these components independently of the nvcc driver binary.
  for relative in #["nvvm/bin/cicc", "bin/nvcc.profile", "version.json", "version.txt"] do
    let path := home / relative
    if ← path.pathExists then
      traceNativeTool path
  let version ← captureProc {
    cmd := nvcc.toString
    args := #["--version"]
    env := #[("NVCC_PREPEND_FLAGS", none), ("NVCC_APPEND_FLAGS", none)]
  }
  addPureTrace version "CUDA compiler version"
  return (nvcc, host)

/-- Compile a CUDA component or its portable stub, tracking headers and compiler settings. -/
private def buildNativeBackendLib (pkg : Package) (dir stem : String) :
    FetchM (Job FilePath) := do
  let lean ← getLeanInstall
  let headerDeps ← nativeHeaderDeps pkg
  let source := if cudaEnabled then s!"{stem}.cu" else s!"{stem}_stub.c"
  let srcJob ← inputFile (pkg.dir / "csrc/cuda" / dir / source) false
  let srcJob := srcJob.zipWith (fun src _ => src) headerDeps
  let objectStem := if cudaEnabled then stem else s!"{stem}_stub"
  let libraryStem := if cudaEnabled then s!"{stem}_cuda" else objectStem
  let oFile := pkg.buildDir / s!"{objectStem}.o"
  let includes := #["-I", lean.includeDir.toString] ++ nativeIncludeArgs pkg
  let oJob ← if cudaEnabled then
    let compilerJob ← torchlean_cuda_compiler.fetch
    compilerJob.bindM fun (nvcc, host) => do
      -- These are traced arguments: changing the target or compiler must invalidate the object.
      let flags := cudaCompileFlags ++ #[s!"--compiler-bindir={host}"]
      buildO oFile srcJob includes flags nvcc getLeanTrace
  else
    let compilerJob ← nativeCompilerJob "cc"
    compilerJob.bindM fun compiler =>
      buildO oFile srcJob includes #["-O2", "-fPIC"] compiler getLeanTrace
  buildStaticLib (pkg.buildDir / nameToStaticLib libraryStem) #[oJob]

/-- CUDA+cuBLAS matrix multiplication, or portable stubs. -/
target torchlean_dgemm_cuda pkg : FilePath :=
  buildNativeBackendLib pkg "blas" "torchlean_dgemm_cuda"

/-- CUDA kernels, or portable stubs. -/
target torchlean_cuda_kernels pkg : FilePath :=
  buildNativeBackendLib pkg "kernels" "torchlean_cuda_kernels"

/-- CUDA convolution and pooling, or portable stubs. -/
target torchlean_cuda_conv_pool pkg : FilePath :=
  buildNativeBackendLib pkg "conv_pool" "torchlean_cuda_conv_pool"

/-- CUDA tensor buffers, or portable stubs. -/
target torchlean_cuda_tensor pkg : FilePath :=
  buildNativeBackendLib pkg "tensor" "torchlean_cuda_tensor"

/-- Compile the native bulk operations for packed host tensor storage once. -/
private def buildTensorCpuObject (pkg : Package) := do
  let lean ← getLeanInstall
  let srcJob ← inputFile
    (pkg.dir / "csrc/cpu/torchlean_tensor.c") false
  let oFile := pkg.buildDir / "torchlean_tensor_cpu.o"
  let compilerJob ← nativeCompilerJob "cc"
  compilerJob.bindM fun compiler =>
    buildO oFile srcJob #["-I", lean.includeDir.toString] #["-O3", "-fPIC"] compiler getLeanTrace

/-- Object shared by the static executable link and the dynamic elaborator library. -/
target torchlean_tensor_cpu_object pkg : FilePath :=
  buildTensorCpuObject pkg

/-- Packed host tensor primitives linked into compiled executables. -/
target torchlean_tensor_cpu pkg : FilePath := do
  let oJob ← torchlean_tensor_cpu_object.fetch
  let libFile := pkg.buildDir / nameToStaticLib "torchlean_tensor_cpu"
  buildStaticLib libFile #[oJob]

/-- Packed host tensor primitives loaded by Lean for `#eval` and documentation examples. -/
target torchlean_tensor_cpu_shared pkg : Dynlib := do
  let oJob ← torchlean_tensor_cpu_object.fetch
  let libName := "torchlean_tensor_cpu"
  let libFile := pkg.sharedLibDir / nameToSharedLib libName
  buildLeanSharedLib libName libFile #[oJob] #[]

/-- Repair large frees and delayed arena purging in the pinned Linux allocator. -/
target torchlean_allocator pkg : FilePath := do
  let lean ← getLeanInstall
  let buildScript := pkg.dir / "scripts/lean_allocator.py"
  let scriptJob ← inputFile buildScript false
  let compatJob ← inputFile (pkg.dir / "csrc/runtime/lean_libc_compat.h") false
  let headerJob ← inputFile (lean.includeDir / "lean/mimalloc.h") false
  let deps := scriptJob.mix (compatJob.mix headerJob)
  let compilerJob ← nativeCompilerJob "c++"
  compilerJob.bindM fun compiler => do
    let output := pkg.buildDir / "torchlean_allocator.o"
    buildFileAfterDep output deps (fun _ => do
      proc {
        cmd := "python3"
        args := #[buildScript.toString, "--lean-include", lean.includeDir.toString,
          "--compiler", compiler.toString, "--output", output.toString]
      }) getLeanTrace

/-! ## LibTorch bridge -/

/-- LibTorch root for native include and library paths. -/
private def libtorchHome (pkg : Package) : String :=
  (pkg.dir / libtorchHomeConfig.getD "libtorch").toString

/-- Validate the configured SDK and track its path for native builds. -/
private def libtorchResolveJob (pkg : Package) : SpawnM (Job FilePath) := do
  let stamp := pkg.buildDir / "libtorch.path"
  let home : FilePath := libtorchHome pkg
  let configJob ← inputFile (pkg.dir / "lakefile.lean") false
  buildFileAfterDep stamp configJob (fun _ => do
    unless (← (home / "include").isDir) && (← (home / "lib").isDir) do
      error <| s!"LibTorch home must contain include/ and lib/: {home}. " ++
        "Pass -Klibtorch_home=/path/to/libtorch when enabling the optional bridge."
    IO.FS.createDirAll pkg.buildDir
    IO.FS.writeFile stamp (home.toString ++ "\n"))
    (pure <| .ofHash (pureHash home.toString) "LibTorch configuration")

/-- Compile the LibTorch SDPA bridge translation unit (clang-cl on Windows, g++ on Linux). -/
private def buildLibtorchSDPAObj (pkg : Package) : SpawnM (Job FilePath) := do
  let lean ← getLeanInstall
  let resolveJob ← libtorchResolveJob pkg
  let headerDeps ← nativeHeaderDeps pkg
  let lt := libtorchHome pkg
  let cppJob ← inputFile (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp") false
  let cppO := pkg.buildDir / "torchlean_libtorch_sdpa.o"
  let deps := cppJob.zipWith (fun src _ => src) (resolveJob.mix headerDeps)
  let compiler := if Platform.isWindows then "clang-cl" else "c++"
  let compilerJob ← nativeCompilerJob compiler
  compilerJob.bindM fun compiler =>
    buildO cppO deps #[] (torchBridgeCompileFlags pkg lean lt) compiler getLeanTrace

/-- Linkable error-returning symbols when the CUDA LibTorch provider is unavailable. -/
private def buildLibtorchSDPAStub (pkg : Package) := do
  let lean ← getLeanInstall
  let srcJob ← inputFile (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa_stub.c") false
  let oFile := pkg.buildDir / "torchlean_libtorch_sdpa_stub.o"
  let compilerJob ← nativeCompilerJob "cc"
  let oJob ← compilerJob.bindM fun compiler =>
    buildO oFile srcJob #["-I", lean.includeDir.toString] #["-O2", "-fPIC"] compiler getLeanTrace
  let libFile := pkg.buildDir / nameToStaticLib "torchlean_libtorch_sdpa_stub"
  buildStaticLib libFile #[oJob]

/-- Link the LibTorch SDPA bridge shared library from its object file (Linux only).
The shared library encapsulates the LibTorch/CUDA dependencies, so executables only need the
`.so` path resolved at load time (via the `rpath` entries in `torchBridgeLinkArgs`). -/
private def buildLibtorchSDPASo (pkg : Package) (cppOJob : Job FilePath) := do
  let lt := libtorchHome pkg
  let soFile := pkg.buildDir / nameToSharedLib "torchlean_libtorch_sdpa"
  cppOJob.mapM fun o => do
    let linker ← nativeCompilerPath "g++"
    traceNativeTool linker
    addPureTrace (torchBridgeLinkArgs lt) "link flags"
    let art ← buildArtifactUnlessUpToDate soFile (ext := sharedLibExt) (restore := true) do
      compileSharedLib soFile (#[o.toString] ++ torchBridgeLinkArgs lt) linker
    return art.path

/-! ## Target declarations -/

/-- LibTorch SDPA bridge, when built with `-K cuda=true -K libtorch=true`;
otherwise a stub static library whose symbols error at call time.

NOTE: On Linux, the bridge is a *shared* library that encapsulates its
LibTorch/CUDA dependencies (via `rpath`); On Windows, it is a *static*
archive. A Windows DLL would export nothing by default — the bridge
`.cpp` has no `__declspec(dllexport)` annotations — and its LibTorch/MSVC-EH
dependencies would need another DLL-side resolution, so the static archive is
created with the same shape as the stub. This platform-neutral name is what
`NN/Backend/Attention.lean` references in `buildTarget?`. -/
target torchlean_libtorch_sdpa pkg : FilePath := do
  if cudaEnabled && libtorchEnabled then
    let cppOJob ← buildLibtorchSDPAObj pkg
    if Platform.isWindows then
      buildStaticLib (pkg.buildDir / nameToStaticLib "torchlean_libtorch_sdpa") #[cppOJob]
    else
      buildLibtorchSDPASo pkg cppOJob
  else
    pure (Job.pure (pkg.buildDir / "torchlean_libtorch_sdpa_skipped"))

/-- Error-returning stub symbols used when the LibTorch bridge is not enabled. -/
target torchlean_libtorch_sdpa_stub pkg : FilePath :=
  if !cudaEnabled || !libtorchEnabled then
    buildLibtorchSDPAStub pkg
  else
    pure (Job.pure (pkg.buildDir / "torchlean_libtorch_sdpa_stub_skipped"))

@[default_target]
lean_lib NN where
  moreLinkObjs :=
    (if Platform.isWindows || Platform.isOSX then (#[] : TargetArray FilePath)
      else (#[torchlean_allocator] : TargetArray FilePath)) ++
    (#[
      torchlean_tensor_cpu,
      torchlean_dgemm_cuda,
      torchlean_cuda_kernels,
      torchlean_cuda_conv_pool,
      torchlean_cuda_tensor
    ] : TargetArray FilePath) ++
      if cudaEnabled && libtorchEnabled then
        (#[torchlean_libtorch_sdpa] : TargetArray FilePath)
      else
        (#[torchlean_libtorch_sdpa_stub] : TargetArray FilePath)
  -- The reusable library follows its canonical umbrella. Examples, tests, CI-only modules,
  -- documentation, and executable roots have separate targets below.
  roots := #[`NN]

/-- Runnable and narrative examples, kept out of the reusable `NN` library target. -/
lean_lib NNExamples where
  roots := #[`NN.Examples]
  globs := #[.one `NN.Examples, .submodules `NN.Examples]

/-- Test modules used by the curated native test runner. -/
lean_lib NNTests where
  roots := #[`NN.Tests.Suite]
  globs := #[.submodules `NN.Tests]

/-- Ordinary CI-only imports omitted from the downstream `NN` umbrella. -/
lean_lib NNCI where
  roots := #[`NN.CI.All]

/-- Proof-heavy modules typechecked by the docs build or an explicit local target. -/
lean_lib NNSlowProofs where
  roots := #[`NN.CI.SlowProofs]

/-- Complete maintained API documentation surface. -/
lean_lib TorchLeanDocs where
  roots := #[`NN.Docs]

-- Unified verification CLI registry: `lake exe verify -- <tool> [args...]`
lean_exe verify where
  root := `NN.Verification.Main

-- Native runner for `lake test`, including tests that call backend externs.
lean_exe nn_tests_suite where
  root := `NN.Tests.Suite

-- Cross-runtime numerical regression tools.
lean_exe pytorch_export_check where
  root := `NN.Tests.Interop.PyTorchMain

lean_exe native_float32_parity where
  root := `NN.Tests.Floats.NativePrimitiveParityMain

-- Optional LibTorch SDPA bridge test. Requires:
--   lake exe -K cuda=true -K libtorch=true libtorch_sdpa_test
lean_exe libtorch_sdpa_test where
  root := `NN.Tests.Runtime.Cuda.LibTorchSDPA

-- Repo-policy lints (header hygiene, banned constructs, etc.) via `lake lint`.
lean_exe torchlean_lint where
  srcDir := "scripts/checks"
  root := `TorchLeanLint

-- Runnable examples: `lake exe torchlean <example> [args...]`.
-- Build with `lake -R -K cuda=true build` before passing `--cuda` to an example.
lean_exe torchlean where
  root := `NN.Examples.RunnerMain

-- Shared executable numerical formats and refinement proofs.
require floatlib from git
  "https://github.com/lean-dojo/FloatLib" @ "52ab504bfcd8e5395b29a4f64b617b401e5d16ac"

-- Complete API documentation (HTML) via `lake build TorchLeanDocs:docs`.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.34.0"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.34.0"