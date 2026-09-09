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
2. **Windows-specific functions** — host directory resolution and the link shim.
3. **Compile/link flag functions** — compile/link argument arrays per platform/external library.
4. **Build helper functions** — job-building functions for native objects and libraries.
5. **LibTorch bridge** — helpers for the optional LibTorch SDPA backend library
   (shared on Linux, static on Windows — see the `torchlean_libtorch_sdpa` target).
6. **Target declarations** — thin wiring only: no logic beyond fetching and sequencing.

## Windows native linking

On Windows, a PE DLL/executable must resolve every imported symbol at *link time*,
bound to a static library or to the import library of a named DLL — there is no
"resolve from whatever is loaded in the process" fallback like ELF's global symbol
scope. Furthermore,

* Everything the native objects reference must appear on the executable's
  link line: the CUDA and the LibTorch import libraries (if enabled).
* No `rpath` on Windows: the loader finds `libleanshared.dll`, `torch.dll`,
  `cudart64_*.dll`, etc. through `PATH` at run time, so no runtime search flags
  can be passed at link time.
-/

/-! ## Config options -/

/-- Whether Lake should compile the native CUDA sources instead of the portable C stubs. -/
private def cudaEnabled : Bool :=
  match get_config? cuda with
  | some v => v == "true" || v == "1"
  | none => false

/-- Normalize and lightly validate a CUDA toolkit root passed through `-K cuda_home=...`. -/
private def cleanCudaHome (p : String) : String :=
  let h := p.trimAscii.toString
  if h.isEmpty then
    "/usr/local/cuda"
  else if h.startsWith "-" then
    panic! s!"cuda_home must be a path, not an option-like value: {h}"
  else
    h

/-- CUDA toolkit root used for includes, libraries, and runtime search path. -/
private def cudaHome : String :=
  match get_config? cuda_home with
  | some p => cleanCudaHome p
  | none => "/usr/local/cuda"

/-- Whether to build the optional LibTorch-backed backend capsules. -/
private def libtorchEnabled : Bool :=
  match get_config? libtorch with
  | some v => v == "true" || v == "1"
  | none => false

/-- Optional explicit LibTorch root from `-K libtorch_home=...`. -/
private def libtorchHomeConfig : Option String :=
  match get_config? libtorch_home with
  | some p =>
      let t := p.trimAscii.toString
      if t.isEmpty then none else some t
  | none => none

  /-- LibTorch root for `-I` / `-L` (must match `resolve_libtorch.sh`). -/
  private def libtorchHome (pkg : Package) : String :=
    match libtorchHomeConfig with
    | some h => h
    | none => (pkg.dir / "libtorch").toString


/-- nvcc `-gencode` flags to select the GPU architectures,  from `-K cuda_arch=...`.

Accepts a comma-separated list of compute capabilities, e.g. `-K cuda_arch=89` or
`-K cuda_arch=89,90`. Each entry `XY` emits `-gencode arch=compute_XY,code=sm_XY`.
Defaults to `89` (Ada Lovelace consumer/workstation parts) so a plain `-K cuda=true`
build runs on the most common recent GPUs without extra flags. -/
private def cudaGencodeFlags : Array String :=
  let archs :=
    match get_config? cuda_arch with
    | some v =>
        let v := v.trimAscii.toString
        if v.isEmpty then #["89"] else v.splitOn "," |>.toArray
    | none => #["89"]
  archs.flatMap fun arch =>
    let arch := arch.trimAscii.toString
    if arch.isEmpty then #[]
    else #["-gencode", s!"arch=compute_{arch},code=sm_{arch}"]

/-! ## Windows-specific functions -/

/-- Validate a required `-K` directory option for Windows CUDA builds.

Missing / malformed values are reported by printing a message to stderr and exiting with
status. -/
private def windowsRequiredDir (opt desc eg : String) (cfg : Option String) : IO String := do
  match cfg with
  | some p =>
      let d := p.trimAscii.toString
      if d.isEmpty || d.startsWith "-" then
        IO.eprintln s!"error: `-K {opt}=...` must be a directory path ({desc}), got: {d}"
        IO.Process.exit 1
      else
        pure d
  | none =>
      IO.eprintln s!"error: CUDA builds on Windows require `-K {opt}=...` ({desc}), e.g. `-K {opt}={eg}`"
      IO.Process.exit 1

/-- Resolve and validate all three directories required for Windows CUDA builds
`(cuda_home, msvc_lib_dir, msys2_lib_dir)`, exiting with a message when any
of these three `-K` option is missing or malformed. -/
private def windowsResolveDirs : IO (String × String × String) := do
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

/-- Pure reader for the `-K cuda_home` option string (no validation). -/
private def windowsCudaHomeOpt : Option String :=
  (get_config? cuda_home).map cleanCudaHome

/-- Resolve a required Windows directory option in a *pure* context. -/
private def windowsUnsafeResolveDir (opt : Option String) : String :=
  match opt with
  | some d => d
  | none => ""

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

Runs inside the job (`Lake.Build.JobM`), so the failure aborts the job before
`nvcc` or the linker emit their more obscure diagnostics. -/
private def windowsPrepareCudaLink : JobM Unit := do
  if Platform.isWindows then
    let (cudaHome, msvcLibDir, msys2LibDir) ← (windowsResolveDirs : JobM _)
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

/-- Include paths corresponding to TorchLean C/C++/CUDA sources. -/
private def torchLeanIncludes (pkg : Package) : Array String :=
  #[
    "-I", (pkg.dir / "csrc/cuda/common").toString,
    "-I", (pkg.dir / "csrc/cuda/conv_pool").toString
  ]

/-- Compile flags for a portable C stub object (plain C compiler). -/
private def stubCompileFlags (pkg : Package) (lean : LeanInstall) : Array String :=
  #["-I", lean.includeDir.toString] ++ torchLeanIncludes pkg ++ #["-O2", "-fPIC"]

/-- CUDA toolkit include directory for the current platform. -/
private def cudaIncludes : Array String :=
  if Platform.isWindows then
    -- On Windows the CUDA toolkit root comes from the mandatory `-K cuda_home=...`.
    #["-I", s!"{windowsUnsafeResolveDir windowsCudaHomeOpt}/include"]
  else
    #["-I", s!"{cudaHome}/include"]

/-- Compile flags for a CUDA backend object (nvcc).

Returns `(weakArgs, traceArgs)` per expected inputs for Lake's `buildO`: `weakArgs`
(not hashed into the rebuild trace) hold system-dependent include paths';
`traceArgs` (hashed) hold `-gencode`, `-O2`, the `_Noreturn`
workaround, and CUDA/CPU target selection -/
private def cudaCompileFlags (pkg : Package) (lean : LeanInstall) : Array String × Array String :=
  let weakArgs :=
    #["-I", lean.includeDir.toString] ++ cudaIncludes ++ torchLeanIncludes pkg
  let traceArgs :=
    -- MSVC (nvcc's mandatory host compiler on Windows) has no `-fPIC`
    let picArgs := if Platform.isWindows then #[] else #["-Xcompiler", "-fPIC"]
    -- `_Noreturn` is a C keyword rejected by nvcc's frontend, so map it to C++ `[[noreturn]]`.
    let msvcArgs := if Platform.isWindows then #["-D_Noreturn=[[noreturn]]"] else #[]
    #["-c", "--std=c++17", "-O2"] ++ cudaGencodeFlags ++ picArgs ++ msvcArgs
  (weakArgs, traceArgs)

/-- C++ compile flags for LibTorch C++ sources (g++ on Linux; clang-cl on Windows). -/
private def torchBridgeCompileFlags (pkg : Package) (lean : LeanInstall) (lt : String) : Array String :=
  let compileArgs :=
    if Platform.isWindows then
      -- Only clang-cl works both libtorch MSVC ABI and lean's FFI.
      -- LibTorch requires at least C++17.
      -- `/EHsc` enables C++ exceptions (off by default in clang-cl); libtorch's error
      -- handling (`TORCH_CHECK` -> `throw c10::Error`) does not compile without it.
      -- `-D__STDNORETURN_H` predefines the include guard of clang's `stdnoreturn.h`,
      -- whose `#define noreturn _Noreturn` is active even in C++ mode.
      #["-c", "-O2", "/std:c++17", "/EHsc", "-D__STDNORETURN_H"]
    else
      #["-c", "-O2", "-fPIC", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"]
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
private def cudaLinkFlags : Array String :=
  if Platform.isWindows then
    -- CUDA's Windows toolkit keeps import libraries under `lib/x64`.
    #[
      "-L", s!"{windowsUnsafeResolveDir windowsCudaHomeOpt}/lib/x64",
      "-lcudart", "-lcublas", "-lcufft",
      "-L", windowsLinkLibsDir.toString
    ]
  else
    #[
      "-L", s!"{cudaHome}/lib64", "-lcudart", "-lcublas", "-lcufft",
      "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
    ]

/-- Link flags for the LibTorch SDPA shared library (Linux only; on Windows the bridge
is a static archive whose LibTorch dependencies are named in `torchBridgeExeLinkFlags`). -/
private def torchBridgeLinkFlags (lt : String) : Array String :=
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
private def torchBridgeExeLinkFlags (lt : String) : Array String :=
  if Platform.isWindows then
    -- `libvcruntime.lib` supplies the MSVC C++-EH support the clang-cl-compiled bridge
    -- object references (`__CxxFrameHandler4`, `__std_terminate`, `__uncaught_exceptions`)
    -- without emitting a `/DEFAULTLIB:` record for it.
    #["-L", s!"{lt}/lib"] ++ libtorchLibs ++ #["-l:libvcruntime.lib"]
  else
    #["-Wl,-rpath," ++ s!"{lt}/lib"]

/-- Native link flags selected by the `cuda`/`libtorch` Lake options. -/
private def nativeLinkFlags : Array String :=
  if cudaEnabled then
    let lt := match libtorchHomeConfig with | some h => h | none => "libtorch"
    cudaLinkFlags ++ (if libtorchEnabled then torchBridgeExeLinkFlags lt else #[])
  else if Platform.isWindows || Platform.isOSX then
    -- Windows and macOS provide libm via the default C runtime
    #[]
  else
    -- CPU stubs call functions from `math.h`; Linux keeps these in `libm`.
    -- Keep libstdc++ for mixed native objects when switching between CPU and CUDA builds.
    #["-lm", "-lstdc++"]

package TorchLean where
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
    ⟨`backward.privateInPublic, false⟩,
    ⟨`backward.privateInPublic.warn, false⟩]
  moreLinkArgs := nativeLinkFlags

/-! ## Build helpers -/

/-- One native backend: the CUDA implementation plus the portable C stub that stands in
for it when the package is built without `-K cuda=true`. -/
private structure NativeBackendLib where
  stem : String
  cudaSrc : String
  stubSrc : String

/-- Track project-owned native headers so a header-only edit invalidates every dependent
object. -/
private def nativeHeaderDeps (pkg : Package) : SpawnM (Job Unit) := do
  let isHeader := fun path : FilePath => path.extension == some "h"
  let common ← inputDir (pkg.dir / "csrc/cuda/common") true isHeader
  let convPool ← inputDir (pkg.dir / "csrc/cuda/conv_pool") true isHeader
  pure <| common.mix convPool

/-- Compile one native source file to an object, tracking the project-owned headers. -/
private def compileNativeObj (pkg : Package) (src oFile : FilePath)
    (weakArgs traceArgs : Array String) (compiler : FilePath)
    (extraDepTrace : JobM Unit := pure ()) : SpawnM (Job FilePath) := do
  let headerDeps ← nativeHeaderDeps pkg
  let srcJob ← inputFile src false
  buildO oFile srcJob weakArgs traceArgs compiler do
    extraDepTrace
    pure headerDeps.getTrace

/-- Build one native backend library for the current Lake configuration. -/
private def buildNativeBackendLib (pkg : Package) (spec : NativeBackendLib) := do
  let lean ← getLeanInstall
  let libFile := pkg.buildDir / nameToStaticLib spec.stem
  if cudaEnabled then
    let oFile := pkg.buildDir / s!"{spec.stem}.o"
    let (weakArgs, traceArgs) := cudaCompileFlags pkg lean
    let oJob ← compileNativeObj pkg (pkg.dir / spec.cudaSrc) oFile weakArgs traceArgs "nvcc"
      windowsPrepareCudaLink
    buildStaticLib libFile #[oJob]
  else
    let oFile := pkg.buildDir / s!"{spec.stem}_stub.o"
    let oJob ← compileNativeObj pkg (pkg.dir / spec.stubSrc) oFile
      (stubCompileFlags pkg lean) #[] "cc"
    buildStaticLib libFile #[oJob]

/-! ## LibTorch bridge helpers -/

/-- Resolve LibTorch; caches `.lake/build/libtorch.path`. -/
private def libtorchResolveJob (pkg : Package) : SpawnM (Job FilePath) := do
  let stamp := pkg.buildDir / "libtorch.path"
  let resolver := pkg.dir / "scripts" / "setup" / "resolve_libtorch.sh"
  let resolverJob ← inputFile resolver false
  let args :=
    match libtorchHomeConfig with
    | some home => #[resolver.toString, home]
    | none => #[resolver.toString]
  buildFileAfterDep stamp resolverJob fun _ =>
    proc { cmd := "bash", args := args, cwd := some pkg.dir }

/-- Compile the LibTorch SDPA bridge translation unit to an object file. -/
private def buildLibtorchSDPAObj (pkg : Package) : SpawnM (Job FilePath) := do
  let lean ← getLeanInstall
  let _ ← libtorchResolveJob pkg
  let lt := libtorchHome pkg
  let cppO := pkg.buildDir / "torchlean_libtorch_sdpa.o"
  let compiler := if Platform.isWindows then "clang-cl" else "c++"
  compileNativeObj pkg (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp") cppO
    (torchBridgeCompileFlags pkg lean lt) #[] compiler

/-- Linkable error-returning symbols when the CUDA LibTorch provider is unavailable. -/
private def buildLibtorchSDPAStub (pkg : Package) := do
  let lean ← getLeanInstall
  let oFile := pkg.buildDir / "torchlean_libtorch_sdpa_stub.o"
  let oJob ← compileNativeObj pkg (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa_stub.c")
    oFile #["-I", lean.includeDir.toString, "-O2", "-fPIC"] #[] "cc"
  let libFile := pkg.buildDir / nameToStaticLib "torchlean_libtorch_sdpa_stub"
  buildStaticLib libFile #[oJob]

/-- Link the LibTorch SDPA bridge shared library from its object file job (Linux only;
see the `torchlean_libtorch_sdpa` target for the Windows shape). The shared library
encapsulates the LibTorch/CUDA dependencies, so executables only need the `.so` path
resolved at load time (via the `rpath` entries in `torchBridgeLinkFlags`). -/
private def linkLibtorchSDPASo (pkg : Package) (cppOJob : Job FilePath)
    : SpawnM (Job FilePath) := do
  let lt := libtorchHome pkg
  let soFile := pkg.buildDir / nameToSharedLib "torchlean_libtorch_sdpa"
  cppOJob.mapM fun o => do
    let art ← buildArtifactUnlessUpToDate soFile (ext := sharedLibExt) (restore := true) do
      compileSharedLib soFile (#[o.toString] ++ torchBridgeLinkFlags lt) "g++"
    return art.path

/-! ## Target declarations -/

/-- The four CUDA backends, in link order. Each also has a portable C stub used when the
package is built without `-K cuda=true`. -/
private def tensorBackend   : NativeBackendLib :=
  { stem := "torchlean_cuda_tensor"
    cudaSrc := "csrc/cuda/tensor/torchlean_cuda_tensor.cu"
    stubSrc := "csrc/cuda/tensor/torchlean_cuda_tensor_stub.c" }
private def kernelsBackend  : NativeBackendLib :=
  { stem := "torchlean_cuda_kernels"
    cudaSrc := "csrc/cuda/kernels/torchlean_cuda_kernels.cu"
    stubSrc := "csrc/cuda/kernels/torchlean_cuda_kernels_stub.c" }
private def convPoolBackend : NativeBackendLib :=
  { stem := "torchlean_cuda_conv_pool"
    cudaSrc := "csrc/cuda/conv_pool/torchlean_cuda_conv_pool.cu"
    stubSrc := "csrc/cuda/conv_pool/torchlean_cuda_conv_pool_stub.c" }
private def dgemmBackend    : NativeBackendLib :=
  { stem := "torchlean_dgemm_cuda"
    cudaSrc := "csrc/cuda/blas/torchlean_dgemm_cuda.cu"
    stubSrc := "csrc/cuda/blas/torchlean_dgemm_cuda_stub.c" }

/-- Native backend for `torchlean_cuda_tensor`: CUDA buffer runtime when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_tensor (pkg) :=
  buildNativeBackendLib pkg tensorBackend

/-- Native backend for `torchlean_cuda_kernels`: CUDA kernels when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_kernels (pkg) :=
  buildNativeBackendLib pkg kernelsBackend

/-- Native backend for `torchlean_cuda_conv_pool`: CUDA conv/pool when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_conv_pool (pkg) :=
  buildNativeBackendLib pkg convPoolBackend

/-- Native backend for `torchlean_dgemm_cuda`: CUDA+cuBLAS when `-K cuda=true`, else C stub. -/
extern_lib torchlean_dgemm_cuda (pkg) :=
  buildNativeBackendLib pkg dgemmBackend

/-- LibTorch SDPA bridge, when built with `-K cuda=true -K libtorch=true`;
otherwise a stub static library whose symbols error at call time.

NOTE: On Linux, the bridge is a *shared* library that encapsulates its
LibTorch/CUDA dependencies (via `rpath`); On Windows, it is a *static*
archive. A Windows DLL would export nothing by default — the bridge
`.cpp` has no `__declspec(dllexport)` annotations — and its LibTorch/MSVC-EH
dependencies would need another DLL-side resolution, so the static archive is
created with the same shape as the stub -/
target torchlean_libtorch_sdpa pkg : FilePath := do
  if cudaEnabled && libtorchEnabled then
    let cppOJob ← buildLibtorchSDPAObj pkg
    if Platform.isWindows then
      buildStaticLib (pkg.buildDir / nameToStaticLib "torchlean_libtorch_sdpa") #[cppOJob]
    else
      linkLibtorchSDPASo pkg cppOJob
  else
    buildLibtorchSDPAStub pkg

@[default_target]
lean_lib NN where
  moreLinkObjs := #[torchlean_libtorch_sdpa]
  -- The reusable library follows its canonical umbrella. Examples, tests, CI-only modules,
  -- documentation, and executable roots have separate targets below.
  roots := #[`NN]

/-- Runnable and narrative examples, kept out of the reusable `NN` library target. -/
lean_lib NNExamples where
  roots := #[`NN.Examples.Zoo]
  globs := #[.submodules `NN.Examples]

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

-- Curated test suite runner (native executable).
-- We run this via `lake exe nn_tests_suite` instead of `lean --run ...` because the Lean
-- interpreter cannot execute definitions from precompiled `.olean`s unless the whole dependency
-- closure is built with interpreter support.
lean_exe nn_tests_suite where
  root := `NN.Tests.Suite

-- Optional LibTorch SDPA bridge test. Requires:
--   lake exe -K cuda=true -K libtorch=true libtorch_sdpa_test
lean_exe libtorch_sdpa_test where
  root := `NN.Tests.Runtime.Cuda.LibTorchSDPA

-- Repo-policy lints (header hygiene, banned constructs, etc.) via `lake lint`.
lean_exe torchlean_lint where
  srcDir := "scripts/checks"
  root := `TorchLeanLint

-- Device-agnostic runnable examples (CPU by default; pass `--cuda` after building with CUDA).
--
-- This single executable supports all runnable examples (MLP/CNN/Transformer/Vit/ResNet/GPT2/PPO)
-- via a simple
-- subcommand interface:
--   `lake exe torchlean <example> [args...]`
--
-- CUDA build: `lake build -R -K cuda=true`
lean_exe torchlean where
  root := `NN.Examples.Models.RunnerMain

-- Self-checking positive/negative example for the functional transcendental +
-- scalar-affine ops (`nn.functional.{exp,log,scale,shift,affine}`). Runs the
-- autograd checks compiled; exits non-zero on any regression.
--   `lake exe transcendentals_check`
lean_exe transcendentals_check where
  root := `NN.Examples.Functional.Transcendentals

-- Complete API documentation (HTML) via `lake build TorchLeanDocs:docs`.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.33.0"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.33.0"
