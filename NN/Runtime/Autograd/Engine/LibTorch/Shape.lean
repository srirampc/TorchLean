/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA helpers: shape/broadcast metadata derived from `Spec.Shape` proofs.

In particular, CUDA broadcast kernels operate on explicit runtime arrays:
- `inDims  : Array Nat` (outermost-first)
- `outDims : Array Nat` (outermost-first)
- `axisMap : Array Nat` of length `outDims.size`

The `axisMap` encoding matches `csrc/libtorch/torchlean.cpp`:
- `axisMap[j] = 0` means output axis `j` is an inserted/broadcast axis (input coordinate is `0`)
- `axisMap[j] = inAxis+1` maps output axis `j` to input axis `inAxis` (0-based), with the `+1`
  sentinel so `0` can be reserved for inserted axes.

This module provides a total function producing `axisMap` from a `Shape.CanBroadcastTo` proof.
-/

module

public import NN.Spec.Core.TensorReductionShape.Reductions

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean

namespace Broadcast

/-! ### `axisMap` generation -/

/-- Generate the CUDA `axisMap` for right-aligned broadcasting.

The proof establishes compatibility, but the map itself is determined solely by the two ranks:
missing leading axes map to zero, and every remaining output axis maps to its aligned input axis. -/
def axisMap {s₁ s₂ : Shape} (_cb : Shape.CanBroadcastTo s₁ s₂) : Array Nat :=
  Array.replicate (Shape.rank s₂ - Shape.rank s₁) 0 ++
    (Array.range (Shape.rank s₁)).map (fun i => i + 1)

/-- CUDA axis map that restores the axis removed by `shapeAfterSum`.

An out-of-range axis leaves the map unchanged, matching `shapeAfterSum`. Constructing each entry
once avoids repeatedly copying and shifting the suffix while descending through the shape. -/
def afterSumAxisMap (s : Shape) (axis : Nat) : Array Nat :=
  Array.ofFn fun i : Fin (Shape.rank s) =>
    if i.val < axis then i.val + 1 else if i.val = axis then 0 else i.val

/-- CUDA metadata for `TorchLean.Tensor.broadcastAfterSum`. -/
def afterSumArgs (s : Shape) (axis : Nat) : Array Nat × Array Nat × Array Nat :=
  (Shape.toArray (TorchLean.Tensor.shapeAfterSum s axis), Shape.toArray s, afterSumAxisMap s axis)

end Broadcast

end LibTorch
end Autograd
end Runtime
