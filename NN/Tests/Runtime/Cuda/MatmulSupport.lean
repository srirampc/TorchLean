/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Runtime.Autograd.Engine.LibTorch.Convert
public import NN.Runtime.Autograd.Engine.LibTorch.DGemm
public import NN.Runtime.Autograd.Engine.LibTorch.Kernels
public import NN.Runtime.Autograd.Engine.LibTorch.Tape

/-!
# Native matrix multiplication test support

Adapters for comparing the FP32 and FP64 native entrypoints against `Tensor.matmul`.
These deliberately call the FFI directly so the tests exercise both native precision paths.
-/

@[expose] public section


namespace Tests.Cuda.MatmulSupport

open Runtime.Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
Precision selector for LibTorch matmul over Lean `Float` tensors.

- `.fp32` routes through `LibTorch.Buffer` and ATen `bmm`, matching the precision used by the eager
  CUDA tensor-buffer path.
- `.fp64` routes through the host `FloatArray` bridge and ATen `matmul`, preserving the binary64
  element type of Lean `Float`.
-/
inductive MatmulPrecision where
  | fp32
  | fp64
deriving Repr, DecidableEq

open Runtime.Autograd.LibTorch (AnyBuffer)

namespace Internal

/-- Unflatten a kernel result, failing if the native call returned an unexpected element count. -/
def unflatten {m p : Nat} (what : String) (flat : FloatArray) : Result (Tensor Float [m, p]) :=
  match Runtime.Autograd.LibTorch.Convert.unflattenFloat? (s := .dim m (.dim p .scalar)) flat with
  | some tensor => pure tensor
  | none =>
      throw s!"autograd: native matmul: {what} returned {flat.size} elements, expected {m * p}"

end Internal

/--
Matrix multiplication at the requested LibTorch precision.

The `.fp32` path rounds inputs to float32 buffers, calls ATen `bmm` with one matrix pair,
and downloads the result to Lean `Float`. The `.fp64` path preserves binary64 through the
`FloatArray` bridge and ATen `matmul`. Both reject dimensions outside the FFI's `UInt32` range
and unexpected result sizes.
-/
def matmul (precision : MatmulPrecision) {m n p : Nat}
    (a : Tensor Float [m, n])
    (b : Tensor Float [n, p]) :
    Result (Tensor Float [m, p]) :=
  match precision with
  | .fp32 => do
      let aBuf := Runtime.Autograd.LibTorch.Buffer.ofFloatArray
        (Runtime.Autograd.LibTorch.Convert.flattenFloat (s := .dim m (.dim n .scalar)) a)
      let bBuf := Runtime.Autograd.LibTorch.Buffer.ofFloatArray
        (Runtime.Autograd.LibTorch.Convert.flattenFloat (s := .dim n (.dim p .scalar)) b)
      let cBuf := Runtime.Autograd.LibTorch.Buffer.bmm aBuf bBuf
        1 (← AnyBuffer.natToU32Checked m) (← AnyBuffer.natToU32Checked n)
        (← AnyBuffer.natToU32Checked p)
      Internal.unflatten "ATen float32 bmm" (Runtime.Autograd.LibTorch.Buffer.toFloatArray cBuf)
  | .fp64 => do
      let flatA := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := .dim m (.dim n .scalar)) a
      let flatB := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := .dim n (.dim p .scalar)) b
      let flatC := Runtime.Autograd.LibTorch.torchleanDgemmCuda flatA flatB
        (← AnyBuffer.natToU32Checked m) (← AnyBuffer.natToU32Checked n)
        (← AnyBuffer.natToU32Checked p)
      Internal.unflatten "ATen float64 matmul" flatC

end Tests.Cuda.MatmulSupport
