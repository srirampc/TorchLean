/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# TorchLeanLint

TorchLean linter driver.

This is wired into Lake as `lintDriver := "torchlean_lint"`, so you can run:

* `scripts/lake.sh lint` locally
* (optionally) the same command in CI

Unlike mathlib’s `#lint`-heavy pipeline, TorchLean’s “lint” is primarily **repo policy**:
forbidden constructs (unverified proof stubs, unexpected axioms, native decision procedures, etc.),
whitespace hygiene, and consistent file headers. The checks are tuned for high-signal feedback while
the library is still evolving quickly.
-/

@[expose] public section

/-- `scripts/lake.sh lint` entrypoint, propagating the repository policy check's failure. -/
def main (_args : List String) : IO Unit := do
  let cmd := "python3"
  let args := #["scripts/checks/repo_lint.py", "--fail-on-warn"]
  let out ← IO.Process.output { cmd := cmd, args := args }
  if !out.stdout.isEmpty then
    IO.print out.stdout
  if !out.stderr.isEmpty then
    IO.eprint out.stderr
  if out.exitCode != 0 then
    throw <| IO.userError
      s!"command failed ({out.exitCode}): {cmd} {String.intercalate " " args.toList}"
