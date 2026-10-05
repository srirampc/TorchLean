/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Ops

/-!
# Layer Operations over Backend References

Linear layers, normalization, attention, convolution, and pooling. Most names re-export `Ops`
directly; the remaining adapters provide the frontend's argument order and attention defaults.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

-- Re-export the capability operations; only argument adapters and compositions need bodies.
export Ops
  (maxPool avgPool linear mseLoss layerNorm batchNorm conv convTranspose)

variable {m : Type → Type} {α : Type} [Storage α] [Context α] [Monad m]
    [Ops (m := m) (α := α)]

@[inherit_doc Ops.smoothMaxPool]
def smoothMaxPool {d C : Nat} [DecidableEq α]
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (x : Ref (m := m) (α := α) (Shape.ofList (C :: Tensor.to inSpatial (List Nat))))
    (beta : α) :
    m (Ref (m := m) (α := α)
      (Shape.ofList
        (C :: Tensor.to (Spec.poolOutSpatialPad inSpatial kernel stride padding) (List Nat)))) :=
  Ops.smoothMaxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    x beta

@[inherit_doc Ops.attention]
def attention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (wq wk wv : Ref (m := m) (α := α) [dModel, numHeads * headDim])
    (wo : Ref (m := m) (α := α) [numHeads * headDim, dModel])
    (batch : Option Nat := none)
    (x : Ref (m := m) (α := α)
      (match (generalizing := false) batch with | none => [n, dModel] | some b => [b, n, dModel]))
    (mask : Option (Tensor Bool [n, n]) := none)
    (hBatch : batch.getD 1 ≠ 0 := by decide) :
    m (Ref (m := m) (α := α)
      (match (generalizing := false) batch with
        | none => [n, dModel] | some b => [b, n, dModel])) :=
  Ops.attention (m := m) (α := α) (batch := batch)
    hBatch h1 wq wk wv wo x mask

end Runtime.Autograd.Torch
