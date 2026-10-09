/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

import Lake
import Lake.Util.Proc
open Lake DSL
open System

/-- Whether Lake should link the LibTorch CUDA backend instead of the unavailable-backend shim. -/
private def cudaEnabled : Bool :=
  let value := (get_config? cuda).getD "false"
  value == "true" || value == "1"

/-- Explicit SDK root; otherwise the builder uses `TORCHLEAN_LIBTORCH_HOME` or `libtorch/`. -/
private def libtorchHomeConfig : Option String :=
  (get_config? libtorch_home).bind fun path =>
    let path := path.trimAscii.toString
    if path.isEmpty then none else some path

-- ======================  Windows Only Config/Flags ===========================
/-- LibTorch import libraries for the executable link, in link order. Only named
on Windows, where the static backend archive's dependencies must be resolved by the executable. -/
private def libtorchLibs : Array String :=
  #["-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda"]

/-- CUDA toolkit import libraries for the executable link (Windows only). -/
private def cudaLinkArgs : Array String :=
  if Platform.isWindows then
    let cudaHome := ((get_config? cuda_home).getD "").trimAscii.toString
    if cudaHome.isEmpty then
      #[]
    else
      #["-L", s!"{cudaHome}/lib/x64", "-lcudart", "-lcublas", "-lcufft"]
  else
    #[]

/-- MSVC x64 library directory (optional). The clang-cl object references MSVC C++-EH
support symbols (`__CxxFrameHandler4`, `__std_terminate`, `__uncaught_exceptions`) that
live in `libvcruntime.lib`; the LibTorch build helper copies that library into the
win-link-shim so the executable link resolves it by exact name. Pass `-K msvc_lib_dir=...`
in a vcvars shell. -/
private def msvcLibDirConfig : Option String :=
  (get_config? msvc_lib_dir).bind fun path =>
    let path := path.trimAscii.toString
    if path.isEmpty then none else some path

/-- MSYS2 library directory (optional); provides the plain `libuuid.a` that torch's
import libraries reference via `/DEFAULTLIB:uuid`, which the GNU-mode linker looks
up by its Unix-style name. -/
private def msys2LibDirConfig : Option String :=
  (get_config? msys2_lib_dir).bind fun path =>
    let path := path.trimAscii.toString
    if path.isEmpty then none else some path

/-- Private directory holding copies of the Windows libraries that must be resolvable by
bare name on the executable link (`libmsvcrt.a`/`liboldnames.a`/`libmsvcprt.a` from the
MSVC dir, `libvcruntime.a`, and `libuuid.a`), written here by the LibTorch build helper, so
the link depends only on this one `-L` without broadly exposing the MSVC/MSYS2 dirs and
shadowing toolchain libraries. Matches the helper's `build_root / "win-link-shim"`. -/
private def windowsLinkLibsDir : FilePath :=
  __dir__ / ".lake" / "build" / "libtorch" / "win-link-shim"

/-- LibTorch flags for the *executable* link, Windows only. The Windows backend is a static
archive, so the executable must name the import libraries itself: `libtorchLibs` for torch,
and the win-link-shim's exact-named `libvcruntime.lib` (MSVC C++-EH symbols the clang-cl
object references) and `libuuid.a` (torch's `/DEFAULTLIB:uuid`). The link depends only on
`<lt>/lib` and the shim, never on a broad `-L` of the MSVC/MSYS2 dirs. Linux needs no exe-link
flags for the backend: the `.so` carries its own dependencies. -/
private def torchBridgeExeLinkArgs (lt : String) : Array String :=
    #["-L", s!"{lt}/lib"] ++ libtorchLibs ++
    #["-L", windowsLinkLibsDir.toString, "-l:libvcruntime.a"]

-- ==============================  Link Args ===================================
/-- LibTorch's SDK link flags and runtime paths are carried by its private shared library.
On Windows the backend is a static archive whose dependencies the executable must name
itself; on Linux the `.so` carries them and only the rpath is needed, so the executable can
find torch's transitive SDK DSOs (some omit their own RUNPATH). -/
private def nativeLinkArgs : Array String :=
  if cudaEnabled && Platform.isWindows then
    let lt := libtorchHomeConfig.getD "libtorch"
    cudaLinkArgs ++ torchBridgeExeLinkArgs lt
  else if cudaEnabled then
    let lt := libtorchHomeConfig.getD "libtorch"
    #["-Wl,-rpath," ++ s!"{lt}/lib"]
  else if Platform.isWindows || Platform.isOSX then
    -- Windows and macOS provide libm via the default C runtime
    #[]
  else
    -- The packed host tensor primitives call `math.h`; Linux keeps these in `libm`.
    #["-lm"]

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
  dynlibs :=
    if cudaEnabled && Platform.isWindows then
      #[`@/torchlean_tensor_cpu_shared, `@/torchlean_libtorch_shared]
    else
      #[`@/torchlean_tensor_cpu_shared]
  moreLinkArgs := nativeLinkArgs

/-!
## Native backend libraries

`-K cuda=true` builds one LibTorch C++ library containing the numerical C ABI exports. CMake obtains
the ABI, language standard, libraries, and runtime paths from the selected SDK. The default build
needs neither LibTorch nor a CUDA toolkit: it links a small C file that reports the backend as not
linked and fails every GPU call with an explanation.
-/

/--
Find the executable that will be passed to a native compile or link command.
Keep its invocation path: names such as `clang++` can select a language mode even when they are
symlinks to the same binary. The dependency trace separately records the resolved file.
-/
private def nativeCompilerPath (name : String) : JobM FilePath := do
  let path := FilePath.mk name
  let cwd ← IO.currentDir
  -- Windows executables always carry a `.exe` extension, and `pathExists` is a plain
  -- filesystem check (no `PATHEXT` fallback), so a bare `cc`/`c++` on PATH must also
  -- be probed with the suffix appended.
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
    let candidates : List FilePath := List.flatMap (fun dir => names.map (cwd / dir / ·))
      (SearchPath.parse ((← IO.getEnv "PATH").getD ""))
    for candidate in candidates do
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

/-- Arguments for the LibTorch CMake build helper (`scripts/libtorch_build.py`), shared by
its static-archive and Windows shared-DLL variant targets. -/
private def libtorchBuildArgs (pkg : Package) (leanInclude : FilePath) (scriptPath : FilePath) : Array String :=
  let base := #[scriptPath.toString, "--package-dir", pkg.dir.toString,
    "--build-dir", pkg.buildDir.toString, "--lean-include", leanInclude.toString]
  let args := match libtorchHomeConfig with
    | some home => base.push s!"--libtorch-home={home}"
    | none => base
  let cudaHome := ((get_config? cuda_home).getD "").trimAscii.toString
  let args := if !cudaHome.isEmpty then args.push s!"--cuda-home={cudaHome}" else args
  let args := match msys2LibDirConfig with
    | some msys2 => args.push s!"--msys2-lib-dir={msys2}"
    | none => args
  let args := match msvcLibDirConfig with
    | some msvc => args.push s!"--msvc-lib-dir={msvc}"
    | none => args
  args

/-- Numerical CUDA primitives built and linked with the selected LibTorch SDK. -/
target torchlean_libtorch pkg : FilePath := do
  let lean ← getLeanInstall
  let scriptJob ← inputFile (pkg.dir / "scripts/libtorch_build.py") false
  scriptJob.mapM fun scriptPath => do
    unless cudaEnabled do
      error "torchlean_libtorch requires -Kcuda=true; the default build does not link LibTorch"
    let args := libtorchBuildArgs pkg lean.includeDir scriptPath
    -- The helper checks SDK/tool/source contents even when Lake previously built this target.
    let fingerprint ← captureProc { cmd := "python3", args := args }
    addPureTrace fingerprint "LibTorch SDK, compiler, flags, and native sources"
    addTrace (← getLeanTrace)
    let output ← IO.FS.realPath
      (pkg.buildDir / "libtorch" /
        if Platform.isWindows then "torchlean_libtorch.lib"
        else nameToSharedLib "torchlean_libtorch")
    addTrace (.ofHash (← computeFileHash output) output.toString)
    return output

/-- Windows DLL variant of the LibTorch backend for interpreter and `#eval` hosts.
The CMake helper builds it with `--shared`: it exports the C ABI and imports the Lean
runtime from the shared DLL chain, so `lean`/`lake` (shared-runtime hosts) load it
soundly — mirroring the CPU path's `torchlean_tensor_cpu_shared`. -/
target torchlean_libtorch_shared pkg : Dynlib := do
  let lean ← getLeanInstall
  let scriptJob ← inputFile (pkg.dir / "scripts/libtorch_build.py") false
  scriptJob.mapM fun scriptPath => do
    unless cudaEnabled do
      error "torchlean_libtorch_shared requires -Kcuda=true; the default build does not link LibTorch"
    unless Platform.isWindows do
      error "torchlean_libtorch_shared is a Windows-only DLL variant; on other platforms the backend .so is torchlean_libtorch"
    let args := libtorchBuildArgs pkg lean.includeDir scriptPath
    -- The helper checks SDK/tool/source contents even when Lake previously built this target.
    let fingerprint ← captureProc { cmd := "python3", args := args.push "--shared" }
    addPureTrace fingerprint "LibTorch SDK, compiler, flags, and native sources (shared)"
    addTrace (← getLeanTrace)
    let output ← IO.FS.realPath (pkg.buildDir / "libtorch" / "torchlean_libtorch.dll")
    addTrace (.ofHash (← computeFileHash output) output.toString)
    return (Dynlib.mk output "torchlean_libtorch" false #[] #[])

/-- Object shared by the static executable link and the dynamic elaborator library. -/
target torchlean_tensor_cpu_object pkg : FilePath := do
  let lean ← getLeanInstall
  let srcJob ← inputFile
    (pkg.dir / "csrc/cpu/torchlean_tensor.c") false
  let oFile := pkg.buildDir / "torchlean_tensor_cpu.o"
  let compilerJob ← nativeCompilerJob "cc"
  compilerJob.bindM fun compiler =>
    buildO oFile srcJob #["-I", lean.includeDir.toString] #["-O3", "-fPIC"] compiler getLeanTrace

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

/-- GPU ABI exports for builds without LibTorch: status reports "not linked", calls fail. -/
target torchlean_libtorch_unavailable pkg : FilePath := do
  let lean ← getLeanInstall
  let srcJob ← inputFile (pkg.dir / "csrc/libtorch/unavailable.c") false
  let operationsJob ← inputFile (pkg.dir / "csrc/libtorch/operations.h") false
  let source := (srcJob.mix operationsJob).map fun _ => pkg.dir / "csrc/libtorch/unavailable.c"
  let oFile := pkg.buildDir / "torchlean_libtorch_unavailable.o"
  let compilerJob ← nativeCompilerJob "cc"
  let oJob ← compilerJob.bindM fun compiler =>
    buildO oFile source #["-I", lean.includeDir.toString] #["-O2", "-fPIC"] compiler getLeanTrace
  buildStaticLib (pkg.buildDir / nameToStaticLib "torchlean_libtorch_unavailable") #[oJob]

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

@[default_target]
lean_lib NN where
  moreLinkObjs :=
    (if Platform.isWindows || Platform.isOSX then (#[] : TargetArray FilePath)
      else (#[torchlean_allocator] : TargetArray FilePath)) ++
    (#[torchlean_tensor_cpu] : TargetArray FilePath) ++
      if cudaEnabled then
        (#[torchlean_libtorch] : TargetArray FilePath)
      else
        (#[torchlean_libtorch_unavailable] : TargetArray FilePath)
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

-- Unified verification CLI registry: `scripts/lake.sh exe verify -- <tool> [args...]`
lean_exe verify where
  root := `NN.Verification.Main

-- Native runner for `scripts/lake.sh test`, including tests that call backend externs.
lean_exe nn_tests_suite where
  root := `NN.Tests.Suite

-- Cross-runtime numerical regression tools.
lean_exe pytorch_export_check where
  root := `NN.Tests.Interop.PyTorchMain

lean_exe native_float32_parity where
  root := `NN.Tests.Floats.NativePrimitiveParityMain

-- Focused SDPA regression, linked with the complete LibTorch numerical backend:
--   scripts/lake.sh -Kcuda=true exe libtorch_sdpa_test
lean_exe libtorch_sdpa_test where
  root := `NN.Tests.Runtime.Cuda.LibTorchSDPA

-- Repo-policy lints (header hygiene, banned constructs, etc.) via `scripts/lake.sh lint`.
lean_exe torchlean_lint where
  srcDir := "scripts/checks"
  root := `TorchLeanLint

-- Runnable examples: `scripts/lake.sh exe torchlean <example> [args...]`.
-- Build with `scripts/lake.sh -Kcuda=true build` before passing `--cuda` to an example.
lean_exe torchlean where
  root := `NN.Examples.RunnerMain

-- Shared executable numerical formats and refinement proofs.
require floatlib from git
  "https://github.com/lean-dojo/FloatLib" @ "main"

-- Complete API documentation (HTML) via `scripts/lake.sh build TorchLeanDocs:docs`.
require «doc-gen4» from git
  "https://github.com/leanprover/doc-gen4" @ "v4.34.0"

-- Keep `mathlib` last so Mathlib’s dependency versions win, which is required for cache tooling.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.34.0"
