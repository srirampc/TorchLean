#!/usr/bin/env python3
"""Build the LibTorch C ABI library through the selected SDK's CMake package.

Lake calls this helper even on cache hits, so changes to the SDK, toolchain, flags,
or source contents cannot reuse an incompatible backend. Compilation needs a
Linux host with a full LibTorch SDK; --resolve-home and --help only inspect paths.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


SOURCES = (
    "torchlean.cpp", "torchlean_libtorch.h", "operations.h", "CMakeLists.txt",
)
BUILD_ENV = (
    "PATH", "CXX", "CC", "CXXFLAGS", "CFLAGS", "CPPFLAGS", "LDFLAGS",
    "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE", "CMAKE_GENERATOR",
    "CUDACXX", "CUDAHOSTCXX", "CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT",
    "CUDA_TOOLKIT_ROOT_DIR", "TORCH_CUDA_ARCH_LIST", "CMAKE_CUDA_ARCHITECTURES",
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN",
    "TORCHLEAN_CMAKE", "TORCHLEAN_CXX", "TORCHLEAN_LIBTORCH_CMAKE_ARGS",
    "CMAKE_ARGS", "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_CUDA_COMPILER_LAUNCHER",
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_record(path: Path, *, contents: bool = True) -> dict:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    record = {
        "path": str(path), "resolved": str(resolved), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
    }
    if contents:
        record["sha256"] = file_hash(resolved)
    return record


def tree_records(root: Path, *, contents: bool = True) -> list[dict]:
    # Only explicitly selected SDK/header directories; never walk the package's build cache.
    return [file_record(path, contents=contents) for path in sorted(root.rglob("*"))
            if path.is_file()]


def resolve_home(package: Path, configured: str | None) -> Path:
    value = (configured or "").strip()
    if not value:
        value = (os.environ.get("TORCHLEAN_LIBTORCH_HOME") or "libtorch").strip()
    if not value or value.startswith("-"):
        raise ValueError("libtorch_home must be a directory path, not an option")
    home = (package / value).resolve()
    for relative in ("include", "lib", "share/cmake/Torch/TorchConfig.cmake"):
        path = home / relative
        if not (path.is_file() if relative.endswith(".cmake") else path.is_dir()):
            raise ValueError(
                f"cuda=true requires a LibTorch SDK; missing {home / relative}. "
                "Pass -Klibtorch_home=/path/to/libtorch or set TORCHLEAN_LIBTORCH_HOME."
            )
    return home


def executable(name: str) -> Path:
    found = shutil.which(name)
    if not found:
        raise ValueError(f"required build tool not found: {name}")
    # Preserve the invocation name (clang++ can be a symlink to clang).
    return Path(os.path.abspath(found))


def tool_record(path: Path) -> dict:
    result = file_record(path)
    result["version"] = subprocess.check_output(
        [str(path), "--version"], text=True, stderr=subprocess.STDOUT
    )
    return result


def cmake_cache(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line and not line.startswith(("#", "//")) and "=" in line:
                key, value = line.split("=", 1)
                values[key.split(":", 1)[0]] = value
    return values


def discovered_inputs(directory: Path) -> list[dict]:
    """Track tools/dependencies that the SDK, rather than TorchLean, selected."""
    records = {}
    cache = cmake_cache(directory / "CMakeCache.txt")
    for key, value in cache.items():
        if key.endswith(("_COMPILER", "_LINKER", "_AR", "_RANLIB", "_MAKE_PROGRAM")):
            path = Path(value)
            if path.is_file():
                records[str(path)] = file_record(path)
        elif any(word in key.upper() for word in ("LIBRARY", "INCLUDE", "TOOLKIT")):
            for part in value.split(";"):
                path = Path(part)
                if path.is_absolute() and path.is_file():
                    records[str(path)] = file_record(path, contents=False)
    for key in ("CUDAToolkit_ROOT", "CUDA_TOOLKIT_ROOT_DIR", "CUDAToolkit_BIN_DIR"):
        root = Path(cache.get(key, ""))
        if not root.is_absolute():
            continue
        if key.endswith("BIN_DIR"):
            root = root.parent
        for relative in ("bin/nvcc", "bin/ptxas", "nvvm/bin/cicc", "bin/nvcc.profile",
                         "version.json", "version.txt"):
            path = root / relative
            if path.is_file():
                records[str(path)] = file_record(path)
    # CMake's file API lists all files read during configure, including toolkit modules.
    reply = directory / ".cmake/api/v1/reply"
    for file in sorted(reply.glob("cmakeFiles-v1-*.json")):
        for item in json.loads(file.read_text()).get("inputs", []):
            path = Path(item["path"])
            if path.is_absolute() and path.is_file() and not item.get("isGenerated"):
                records[str(path)] = file_record(path)
    # The C++ dependency files include CUDA and system headers outside the LibTorch prefix.
    # Hash the actual dependencies, without recursively walking a system/toolkit installation.
    for depfile in sorted((directory / "CMakeFiles/torchlean_libtorch.dir").glob("*.d")):
        text = depfile.read_text().replace("\\\n", "")
        for name in shlex.split(text.partition(":")[2]):
            path = Path(name)
            if not path.is_absolute():
                path = directory / path
            if str(path) not in records and path.is_file():
                records[str(path)] = file_record(path)
    return [records[key] for key in sorted(records)]


def write_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def run(command: list[str], *, env: dict[str, str]) -> None:
    # stdout is reserved for the fingerprint returned to Lake.
    print("+ " + " ".join(command), file=sys.stderr, flush=True)
    subprocess.run(command, check=True, env=env, stdout=sys.stderr)


def nvcc_wrapper(cmd_path: Path, nvcc: Path, host_compiler: str | Path | None) -> None:
    """Write a `.cmd` that forwards to nvcc and forces the MSVC host compiler.

    Torch's `Caffe2/public/cuda.cmake` unconditionally sets `CMAKE_CUDA_HOST_COMPILER`
    to the CXX compiler when its ID is "Clang", which nvcc rejects on Windows
    ("Host compiler targets unsupported OS"). Adding `-ccbin` in the wrapper
    overrides any `-ccbin` CMake emits, so nvcc always uses MSVC's cl.exe.
    """
    if host_compiler is None:
        raise ValueError("Windows LibTorch build requires a vcvars64 shell "
                         "(cl.exe is not on PATH)")
    cmd_path.write_text(
        '@echo off\r\n'
        f'"{nvcc}" %* -ccbin="{host_compiler}"\r\n',
        encoding="utf-8")


def msvc_tool_dir() -> Path | None:
    """Absolute MSVC tools dir (``...\\VC\\Tools\\MSVC\\<ver>\\bin\\Hostx64\\x64``) in a vcvars64 shell.

    Returns ``cl.exe``'s parent directory as the MSVC tools dir that
    also holds ``link.exe``. ``None`` outside a vcvars64 shell (cl.exe not on PATH).
    """
    if sys.platform != "win32":
        return None
    cl = shutil.which("cl.exe")
    return Path(cl).parent if cl else None


def msvc_lib_dir(flag: str) -> str | None:
    """MSVC ``lib/<arch>`` dir for the executable link: ``--msvc-lib-dir`` when given,
    else derived from cl.exe's parent directory.

    ``...\\bin\\Hostx64\\x64\\cl.exe`` -> ``VC\\Tools\\MSVC\\<ver>\\lib\\<arch>``, where the
    target arch is the tools dir's own name (``x64``), so cross setups like ``Hostx86/x64``
    also resolve correctly without hardcoding ``x64``.
    """
    if flag:
        return flag
    tool_dir = msvc_tool_dir()
    if tool_dir is None:
        return None
    return str(tool_dir.parents[2] / "lib" / tool_dir.name)


def win_link_shim(args: argparse.Namespace, build_root: Path) -> None:
    """Provision the private dir of Windows libs the executable link needs.

    On Windows, the LibTorch SDK's import libraries carry a `/DEFAULTLIB:uuid`
    record that the GNU-mode linker looks up as the plain name `libuuid.a`;
    also, the Lean runtime's `IO.Process` path pulls libuv. Just adding the MSYS2
    lib directory to the search path would shadow the toolchain's own
    ABI-matched `libuv.a` (first-match wins) and reintroduce the `_snwprintf_s`
    failure. To solve, copy `libuuid.a` into a private shim directory. Libraries
    corresponding to the clang-cl /MD object's `/DEFAULTLIB:` records (msvcrt,
    oldnames, msvcprt, and our explicit `-l:libvcruntime.a`) are also
    copied from the MSVC lib dir.
    """
    if sys.platform != "win32":
        return
    msys2 = args.msys2_lib_dir
    msvc = msvc_lib_dir(args.msvc_lib_dir)
    shim = build_root / "win-link-shim"
    # Validate all library directory locations.
    if not msys2:
        raise ValueError("Windows LibTorch build requires --msys2-lib-dir=<MSYS2 lib dir> "
                         "(pass -K msys2_lib_dir=... to lake)")
    if not msvc:
        raise ValueError("Windows LibTorch build requires --msvc-lib-dir=<MSVC lib dir> "
                         "or a vcvars64 shell so cl.exe is on PATH "
                         "(pass -K msvc_lib_dir=... to lake)")
    libuuid_src = Path(msys2) / "libuuid.a"
    if not libuuid_src.is_file():
        raise ValueError(f"Windows LibTorch build requires libuuid.a in {libuuid_src}")
    msvc_dir = Path(msvc)
    # Copy all four libraries corresponding to /DEFAULTLIB: under their GNU
    # names so the executable link depends only on this private shim directory.
    msvc_libraries = (("msvcrt.lib", "libmsvcrt.a"), ("oldnames.lib", "liboldnames.a"),
                      ("msvcprt.lib", "libmsvcprt.a"), ("vcruntime.lib", "libvcruntime.a"))
    for name, _ in msvc_libraries:
        if not (msvc_dir / name).is_file():
            raise ValueError(f"Windows LibTorch build requires {name} in {msvc_dir}")
    # All inputs verified: populate the shim.
    shim.mkdir(parents=True, exist_ok=True)
    shutil.copy2(libuuid_src, shim / "libuuid.a")
    for name, gnu in msvc_libraries:
        shutil.copy2(msvc_dir / name, shim / gnu)


def win_cmake_options(args: argparse.Namespace, package: Path, build_root: Path) -> list[str]:
    """CMake options for the Windows backend build, or ``[]`` on other platforms.

    MSYS2's ``/usr/bin/link.exe`` shadows MSVC's linker on PATH, so ``CMAKE_LINKER``
    is pinned to the absolute MSVC ``link.exe``. Torch's ``cuda.cmake`` forces nvcc's
    host compiler to clang-cl (rejected by nvcc), so ``CMAKE_CUDA_COMPILER`` names a
    generated wrapper that adds a trailing ``-ccbin=<MSVC cl.exe>``.
    """
    if sys.platform != "win32":
        return []
    options: list[str] = []
    tool_dir = msvc_tool_dir()
    if tool_dir is not None:
        linker = tool_dir / "link.exe"
        if linker.is_file():
            options.append(f"-DCMAKE_LINKER={linker}")
        host_compiler = tool_dir / "cl.exe"
    else:
        host_compiler = None
    toolkit = None
    if args.cuda_home:
        toolkit = (package / args.cuda_home).resolve()
    nvcc = toolkit / "bin" / "nvcc.exe" if toolkit else None
    if nvcc is None or not nvcc.is_file():
        raise ValueError("Windows LibTorch build requires -K cuda_home=<CUDA toolkit>")
    wrapper = build_root / "nvcc_host_wrapper.cmd"
    nvcc_wrapper(wrapper, nvcc, host_compiler)
    options.append(f"-DCMAKE_CUDA_COMPILER={wrapper}")
    return options


@contextlib.contextmanager
def _build_lock(build_root: Path):
    """Advisory exclusive lock guarding concurrent builds of one build root.

    POSIX uses ``fcntl.flock``; Windows has no ``fcntl`` module, so lock the
    same bytes with ``msvcrt.locking`` instead. The lock is advisory: both
    processes are this helper, so they agree on the protocol.
    """
    handle = (build_root / "build.lock").open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt
            # The lock region must exist in the file; extend it by one byte.
            handle.seek(0, 2)
            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def build(args: argparse.Namespace, package: Path, home: Path) -> str:
    if sys.platform not in ("linux", "win32"):
        raise ValueError("cuda=true currently requires a Linux or Windows LibTorch SDK")
    if not args.build_dir or not args.lean_include:
        raise ValueError("building requires --build-dir and --lean-include")
    source = package / "csrc/libtorch"
    lean_include = Path(args.lean_include).resolve(strict=True)
    if not (lean_include / "lean/lean.h").is_file():
        raise ValueError(f"Lean headers not found in {lean_include}")
    for name in SOURCES:
        if not (source / name).is_file():
            raise ValueError(f"LibTorch backend source is missing: {source / name}")
    cmake = executable(os.environ.get("TORCHLEAN_CMAKE", "cmake"))
    default_compiler = "clang-cl" if sys.platform == "win32" else "c++"
    compiler = executable(os.environ.get("TORCHLEAN_CXX") or os.environ.get("CXX") or default_compiler)
    extra = json.loads(os.environ.get("TORCHLEAN_LIBTORCH_CMAKE_ARGS", "[]"))
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise ValueError("TORCHLEAN_LIBTORCH_CMAKE_ARGS must be a JSON array of CMake -D options")
    for option in extra:
        if not option.startswith("-D") or "=" not in option:
            raise ValueError("extra CMake arguments must be individual -Dkey=value options")
        key = option[2:].split("=", 1)[0].split(":", 1)[0]
        if (key.startswith("TORCHLEAN_") or key in {
            "Torch_DIR", "Caffe2_DIR", "CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE",
            "CMAKE_LIBRARY_OUTPUT_DIRECTORY", "CMAKE_SKIP_RPATH", "CMAKE_SKIP_INSTALL_RPATH",
            "CMAKE_INSTALL_RPATH", "CMAKE_INSTALL_RPATH_USE_LINK_PATH",
        }):
            raise ValueError(f"reserved build setting: {key}")

    configuration = {
        "format": 1, "package": str(package), "home": str(home),
        "cmake": tool_record(cmake), "compiler": tool_record(compiler),
        "python": file_record(Path(sys.executable)),
        "helper": file_record(Path(__file__).resolve()),
        "cmakelists": file_record(source / "CMakeLists.txt"),
        "environment": {name: os.environ.get(name) for name in BUILD_ENV},
        "cuda_home": args.cuda_home, "extra": extra,
        "sdk_cmake": tree_records(home / "share/cmake"),
        "sdk_metadata": [
            file_record(home / name) for name in ("version.py", "build-version", "build-hash")
            if (home / name).is_file()
        ],
        # Header metadata is enough to detect an SDK swap; hashing ~10k headers costs every call.
        "sdk_headers": tree_records(home / "include", contents=False),
        # SDK DSOs can be several GB each. Replacement metadata, including ctime and
        # resolved symlinks, is tracked without rereading every DSO on every Lake call.
        "sdk_libraries": tree_records(home / "lib", contents=False),
        "lean_headers": tree_records(lean_include),
    }
    inputs = {
        "configuration": configuration,
        "sources": [file_record(path) for path in sorted(source.iterdir())
                    if path.suffix in (".cpp", ".h", ".hpp")],
    }
    build_root = Path(args.build_dir).resolve() / "libtorch"
    build_root.mkdir(parents=True, exist_ok=True)
    win_link_shim(args, build_root)
    # Lock direct Lake/helper invocations too; the wrapper's checkout lock is not always held.
    with _build_lock(build_root):
        directory = build_root / "cmake"
        output = build_root / (
            "torchlean_libtorch.lib" if sys.platform == "win32" else "libtorchlean_libtorch.so"
        )
        state_file = build_root / "build.json"
        try:
            previous = json.loads(state_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}
        signature = digest(inputs)
        discovered = discovered_inputs(directory)
        if (previous.get("signature") == signature
                and previous.get("discovered") == discovered
                and output.is_file()
                and previous.get("output_sha256") == file_hash(output)):
            return digest(previous)

        # A changed compiler, SDK, header, or flag must not survive CMake's own timestamp cache.
        # Reconfigure cleanly on misses; the usual unchanged build skips CMake entirely.
        if directory.exists():
            shutil.rmtree(directory)
        query = directory / ".cmake/api/v1/query"
        query.mkdir(parents=True)
        (query / "cmakeFiles-v1").touch()
        env = dict(os.environ)
        # TorchConfig.cmake also consults this variable; prevent a second SDK winning discovery.
        env["TORCH_INSTALL_PREFIX"] = str(home)
        generator = "Ninja" if sys.platform == "win32" else "Unix Makefiles"
        command = [
            str(cmake), "-S", str(source), "-B", str(directory), "-G", generator,
            "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DTORCHLEAN_LIBTORCH_HOME={home}", f"-DTORCHLEAN_LEAN_INCLUDE={lean_include}",
            f"-DTORCHLEAN_OUTPUT_DIR={build_root}",
        ]
        command.extend(win_cmake_options(args, package, build_root))
        if args.cuda_home:
            if args.cuda_home.startswith("-"):
                raise ValueError("cuda_home must be a directory path, not an option")
            # as_posix() so Windows paths work. Torch's bundled
            # FindCUDA.cmake parses PATHS as a CMake argument list and rejects
            # backslash escapes in Windows paths.
            toolkit = (package / args.cuda_home).resolve().as_posix()
            command.extend([f"-DCUDAToolkit_ROOT={toolkit}", f"-DCUDA_TOOLKIT_ROOT_DIR={toolkit}"])
        command.extend(extra)
        run(command, env=env)
        jobs = os.environ.get("TORCHLEAN_LIBTORCH_JOBS", "2")
        if not jobs.isdigit() or int(jobs) < 1:
            raise ValueError("TORCHLEAN_LIBTORCH_JOBS must be a positive integer")
        output.unlink(missing_ok=True)
        run([str(cmake), "--build", str(directory), "--target", "torchlean_libtorch",
             "--parallel", jobs], env=env)
        if not output.is_file():
            raise ValueError(f"CMake did not produce the LibTorch backend: {output}")
        state = {
            "signature": signature, "inputs": inputs,
            "discovered": discovered_inputs(directory),
            "sdk": (directory / "sdk.txt").read_text(),
            "output": str(output), "output_sha256": file_hash(output),
        }
        write_json(state_file, state)
        return digest(state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--libtorch-home")
    parser.add_argument("--build-dir")
    parser.add_argument("--lean-include")
    parser.add_argument("--cuda-home", default="")
    parser.add_argument("--msys2-lib-dir", default="")
    parser.add_argument("--msvc-lib-dir", default="")
    parser.add_argument("--resolve-home", action="store_true")
    args = parser.parse_args()
    try:
        package = Path(args.package_dir).resolve(strict=True)
        home = resolve_home(package, args.libtorch_home)
        print(home if args.resolve_home else build(args, package, home))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: LibTorch build: {error}", file=sys.stderr)
        print("cuda=true requires a compatible LibTorch CUDA SDK. "
              "SDK CMake discovery may require a matching CUDA development toolkit.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
