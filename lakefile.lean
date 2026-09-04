/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lake
import Lake.Util.Proc
open Lake DSL
open System

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

/-- Optional explicit LibTorch root from `-K libtorch_home=...`. -/
private def libtorchHomeConfig : Option String :=
  match get_config? libtorch_home with
  | some p =>
      let t := p.trimAscii.toString
      if t.isEmpty then none else some t
  | none => none

/-- Whether to build the optional LibTorch-backed backend capsules. -/
private def libtorchEnabled : Bool :=
  match get_config? libtorch with
  | some v => v == "true" || v == "1"
  | none => false

/-- `-gencode` flags selecting the GPU architectures nvcc compiles for, from `-K cuda_arch=...`.

Accepts a comma-separated list of compute capabilities, e.g. `-K cuda_arch=89` or
`-K cuda_arch=89,90`. Each entry `XY` emits `-gencode arch=compute_XY,code=sm_XY`.
This allows runs on machines whose NVIDIA driver is older than the
CUDA toolkit: without matching SASS, the driver must JIT-compile the embedded PTX,
and an older driver rejects a newer toolkit's PTX with
`cudaErrorUnsupportedPtxVersion` (error 222) at kernel launch.
Defaults to `89` (Ada Lovelace consumer/workstation parts) so a plain `-K cuda=true`
build runs on the most common recent GPUs without extra flags. -/
private def cudaGencodeArgs : Array String :=
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

/-- Validate a required `-K` directory option for Windows CUDA builds.

Missing / malformed values are reported by printing a message to stderr and exiting with
status.  -/
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
private def resolveWindowsDirs : IO (String × String × String) := do
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
private def unsafeResolveWindowsDir (opt : Option String) : String :=
  match opt with
  | some d => d
  | none =>
      -- Unreachable on a Windows CUDA build: the build job runs `prepareWindowsCudaLink` first,
      -- which validates every directory and exits with a clear message when one is missing.
      ""

/-- Directory populated by `prepareWindowsCudaLink` with copies of the handful of Windows
libraries that nvcc's MSVC-compiled host objects reference through `/DEFAULTLIB:`.

The libs directory should contain *only* the libraries nothing else can provide
(eg. `LIBCMT`, `libcpmt`, `OLDNAMES`, and `uuid`), so that its early position in link line
is harmless. -/
private def windowsLinkLibsDir : FilePath :=
  __dir__ / ".lake" / "build" / "win-link-shim"

/-- The list of libraries copied into `windowsLinkLibsDir` to accomodate
the names in nvcc's MSVC-compiled host objects via `/DEFAULTLIB:`.

Currently, this includes:
1. LIBCMT.lib, libcpmt.lib and libvcruntime.lib: MSVC C/C++ runtime libs
2. OLDNAMES.lib : Old POSIX names
3. libuuid.a (from MSYS2) : Windows SDK GUID definitions
The UCRT static library (`libucrt.lib`) is absent since it is
provided by Lean itself.  -/
private def windowsLinkLibsInputs (msvcDir msys2Dir : String) : Array FilePath :=
  #["LIBCMT.lib", "libcpmt.lib", "OLDNAMES.lib", "libvcruntime.lib"].map
    (fun f => FilePath.mk msvcDir / FilePath.mk f)
  ++ #[FilePath.mk msys2Dir / "libuuid.a"]

/-- Populate `windowsLinkLibsDir`,  but if a directory or library required by a
Windows CUDA build is not provided or does not exist, fail build job with a clear
message.

Runs inside the job (`Lake.Build.JobM`), so the failure aborts the job before
`nvcc` or the linker emit their more obscure diagnostics. -/
private def prepareWindowsCudaLink : JobM Unit := do
  if Platform.isWindows then
    let (cudaHome, msvcLibDir, msys2LibDir) ← (resolveWindowsDirs : JobM _)
    let mut missing : Array String := #[]
    for (optName, dir) in #[("cuda_home", cudaHome), ("msvc_lib_dir", msvcLibDir),
        ("msys2_lib_dir", msys2LibDir)] do
      unless (← FilePath.pathExists dir) do
        missing := missing.push s!"-K {optName} (directory does not exist: {dir})"
    unless missing.isEmpty do
      error s!"TorchLean CUDA build on Windows: required directories do not exist:\n  {String.intercalate "\n  " missing.toList}"
    -- Copy the windows libraries. Copying unconditionally since the file sizes are only a few MBs.
    -- NOTE: We cannot add the MSVC / MSYS2 library directories with `-L` because
    -- Lake always inserts package `moreLinkArgs` before the Lean toolchain's
    -- own library directories (`mkLeanLinkArgs`).
    -- When added first, Windows libraries shadows the Lean-included `libgmp.a`/`libuv.a`/...,
    -- leading to linker errors.
    IO.FS.createDirAll windowsLinkLibsDir
    let mut missingLibs : Array String := #[]
    for src in windowsLinkLibsInputs msvcLibDir msys2LibDir do
      if (← src.pathExists) then
        copyFile src (windowsLinkLibsDir / src.fileName.getD src.toString)
      else
        missingLibs := missingLibs.push src.toString
    unless missingLibs.isEmpty do
      error s!"TorchLean CUDA build on Windows: required libraries do not exist:\n  {String.intercalate "\n  " missingLibs.toList}"

/-- Native link flags selected by the `cuda` Lake option. -/
private def nativeLinkArgs : Array String :=
  if cudaEnabled then
    let lt := match libtorchHomeConfig with | some h => h | none => "libtorch"
    if Platform.isWindows then
      -- CUDA's Windows toolkit keeps import libraries under `lib/x64`, and the loader resolves
      -- the runtime DLLs through PATH, so there is no rpath flag to pass.
      -- The MSVC/MSYS2 libraries come from `windowsLinkLibsDir` (see its docstring).
      let cudaArgs := #[
        "-L", s!"{unsafeResolveWindowsDir windowsCudaHomeOpt}/lib/x64",
        "-lcudart", "-lcublas", "-lcufft",
        "-L", windowsLinkLibsDir.toString,
      ]
      if libtorchEnabled then
        cudaArgs ++ #["-L", s!"{lt}/lib"]
      else
        cudaArgs
    else
      let cudaArgs := #[
        "-L", s!"{cudaHome}/lib64", "-lcudart", "-lcublas", "-lcufft",
        "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
      ]
      if libtorchEnabled then
        cudaArgs.push ("-Wl,-rpath," ++ s!"{lt}/lib")
      else
        cudaArgs
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
  moreLinkArgs := nativeLinkArgs

/-!
## Native backend libraries

TorchLean has a small amount of native code behind Lean `extern` declarations. Each component has
the same build shape: compile the CUDA implementation when the package is built with
`-K cuda=true`; otherwise compile the matching C stub so the Lean package still builds on machines
without a CUDA toolkit.
-/

private structure NativeBackendLib where
  stem : String
  cudaSrc : String
  stubSrc : String

/-- LibTorch root for `-I` / `-L` (must match `resolve_libtorch.sh`). -/
private def libtorchHome (pkg : Package) : String :=
  match libtorchHomeConfig with
  | some h => h
  | none => (pkg.dir / "libtorch").toString

/-- g++ compile flags for LibTorch C++ sources. -/
private def libtorchCppCompileArgs (pkg : Package) (lean : LeanInstall) (lt : String) : Array String :=
  #[
    "-I", lean.includeDir.toString,
    "-I", s!"{pkg.dir}/csrc/cuda/common",
    "-I", s!"{cudaHome}/include",
    "-I", s!"{lt}/include",
    "-I", s!"{lt}/include/torch/csrc/api/include",
    "-c", "-O2", "-fPIC", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"
  ]

/-- g++ link flags for the LibTorch SDPA shared library. -/
private def libtorchSDPALinkArgs (lt : String) : Array String :=
  #[
    "-L", s!"{lt}/lib",
    "-Wl,--no-as-needed",
    "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
    "-L", s!"{cudaHome}/lib64", "-lcudart",
    "-lstdc++",
    "-Wl,-rpath," ++ s!"{lt}/lib",
    "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
  ]

/-- Include paths shared by the CUDA implementations and the portable C stubs. -/
private def nativeIncludeArgs (pkg : Package) : Array String :=
  #[
    "-I", (pkg.dir / "csrc/cuda/common").toString,
    "-I", (pkg.dir / "csrc/cuda/conv_pool").toString
  ]

/-- Track project-owned native headers so a header-only edit invalidates every dependent object. -/
private def nativeHeaderDeps (pkg : Package) : SpawnM (Job Unit) := do
  let isHeader := fun path : FilePath => path.extension == some "h"
  let common ← inputDir (pkg.dir / "csrc/cuda/common") true isHeader
  let convPool ← inputDir (pkg.dir / "csrc/cuda/conv_pool") true isHeader
  pure <| common.mix convPool

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

/-- LibTorch SDPA forward/backward bridge as a shared library. -/
private def buildLibtorchSDPASo (pkg : Package) := do
  let lean ← getLeanInstall
  let _ ← libtorchResolveJob pkg
  let headerDeps ← nativeHeaderDeps pkg
  let lt := libtorchHome pkg
  let cppJob ← inputFile (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp") false
  let cppO := pkg.buildDir / "torchlean_libtorch_sdpa.o"
  let cppOJob ← buildO cppO cppJob (libtorchCppCompileArgs pkg lean lt) #[] "c++"
    (pure headerDeps.getTrace)
  let soFile := pkg.buildDir / nameToSharedLib "torchlean_libtorch_sdpa"
  cppOJob.mapM fun o => do
    let art ← buildArtifactUnlessUpToDate soFile (ext := sharedLibExt) (restore := true) do
      compileSharedLib soFile (#[o.toString] ++ libtorchSDPALinkArgs lt) "g++"
    return art.path

/-- Linkable error-returning symbols when the CUDA LibTorch provider is unavailable. -/
private def buildLibtorchSDPAStub (pkg : Package) := do
  let lean ← getLeanInstall
  let srcJob ← inputFile (pkg.dir / "csrc/cuda/kernels/torchlean_libtorch_sdpa_stub.c") false
  let oFile := pkg.buildDir / "torchlean_libtorch_sdpa_stub.o"
  let oJob ← buildO oFile srcJob
    #["-I", lean.includeDir.toString, "-O2", "-fPIC"] #[] "cc"
  let libFile := pkg.buildDir / nameToStaticLib "torchlean_libtorch_sdpa_stub"
  buildStaticLib libFile #[oJob]

target torchlean_libtorch_sdpa_so pkg : FilePath :=
  if cudaEnabled && libtorchEnabled then
    buildLibtorchSDPASo pkg
  else
    pure (Job.pure (pkg.buildDir / "torchlean_libtorch_sdpa_skipped"))

target torchlean_libtorch_sdpa_stub pkg : FilePath :=
  if !cudaEnabled || !libtorchEnabled then
    buildLibtorchSDPAStub pkg
  else
    pure (Job.pure (pkg.buildDir / "torchlean_libtorch_sdpa_stub_skipped"))

@[default_target]
lean_lib NN where
  moreLinkObjs :=
    if cudaEnabled && libtorchEnabled then #[torchlean_libtorch_sdpa_so]
    else #[torchlean_libtorch_sdpa_stub]
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

/-- Build one native backend library for the current Lake configuration. -/
private def buildNativeBackendLib (pkg : Package) (spec : NativeBackendLib) := do
  let lean ← getLeanInstall
  let headerDeps ← nativeHeaderDeps pkg
  let includeArgs := nativeIncludeArgs pkg
  let libFile := pkg.buildDir / nameToStaticLib spec.stem
  if cudaEnabled then
    -- On Windows the CUDA toolkit root comes from the mandatory `-K cuda_home=...`.
    let cudaIncDir :=
      if Platform.isWindows then s!"{unsafeResolveWindowsDir windowsCudaHomeOpt}/include"
      else s!"{cudaHome}/include"
    let srcJob ← inputFile (pkg.dir / spec.cudaSrc) false
    let oFile := pkg.buildDir / s!"{spec.stem}.o"
    -- MSVC (nvcc's mandatory host compiler on Windows) has no -fPIC;
    -- x64 Windows code is position-independent by construction.
    let picArgs := if Platform.isWindows then #[] else #["-Xcompiler", "-fPIC"]
    -- Use C++ [[noreturn]] on Windows.
    let msvcArgs :=
      if Platform.isWindows then #["-D_Noreturn=[[noreturn]]"] else #[]
    -- Split flags per Lake's `buildO`'s `weakArgs`/`traceArgs`.
    let oJob ← buildO oFile srcJob
      -- `weakArgs` (not hashed into the rebuild trace) for system-dependent include paths.
      (#["-I", lean.includeDir.toString, "-I", cudaIncDir] ++ includeArgs)
      -- `traceArgs` (hashed) hold everything that changes the compiled artifact:
      --     `-gencode`, `-O2`,  `_Noreturn` workaround, and the CUDA/CPU target selection.
      -- `-gencode` in `traceArgs` is triggers rebuild when the arch list changes.
      (#["-c", "--std=c++17", "-O2"] ++ cudaGencodeArgs ++ picArgs ++ msvcArgs)
      "nvcc"
      (do prepareWindowsCudaLink; pure headerDeps.getTrace)
    buildStaticLib libFile #[oJob]
  else
    let srcJob ← inputFile (pkg.dir / spec.stubSrc) false
    let oFile := pkg.buildDir / s!"{spec.stem}_stub.o"
    let oJob ← buildO oFile srcJob
      (#["-I", lean.includeDir.toString] ++ includeArgs ++ #["-O2", "-fPIC"])
      #[] "cc" (pure headerDeps.getTrace)
    buildStaticLib libFile #[oJob]

/-- Native backend for `torchlean_dgemm_cuda`: CUDA+cuBLAS when `-K cuda=true`, else C stub. -/
extern_lib torchlean_dgemm_cuda (pkg) :=
  buildNativeBackendLib pkg {
    stem := "torchlean_dgemm_cuda"
    cudaSrc := "csrc/cuda/blas/torchlean_dgemm_cuda.cu"
    stubSrc := "csrc/cuda/blas/torchlean_dgemm_cuda_stub.c"
  }

/-- Native backend for `torchlean_cuda_kernels`: CUDA kernels when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_kernels (pkg) :=
  buildNativeBackendLib pkg {
    stem := "torchlean_cuda_kernels"
    cudaSrc := "csrc/cuda/kernels/torchlean_cuda_kernels.cu"
    stubSrc := "csrc/cuda/kernels/torchlean_cuda_kernels_stub.c"
  }

/-- Native backend for `torchlean_cuda_conv_pool`: CUDA conv/pool when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_conv_pool (pkg) :=
  buildNativeBackendLib pkg {
    stem := "torchlean_cuda_conv_pool"
    cudaSrc := "csrc/cuda/conv_pool/torchlean_cuda_conv_pool.cu"
    stubSrc := "csrc/cuda/conv_pool/torchlean_cuda_conv_pool_stub.c"
  }

/-- Native backend for `torchlean_cuda_tensor`: CUDA buffer runtime when `-K cuda=true`, else C stub. -/
extern_lib torchlean_cuda_tensor (pkg) :=
  buildNativeBackendLib pkg {
    stem := "torchlean_cuda_tensor"
    cudaSrc := "csrc/cuda/tensor/torchlean_cuda_tensor.cu"
    stubSrc := "csrc/cuda/tensor/torchlean_cuda_tensor_stub.c"
  }

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
