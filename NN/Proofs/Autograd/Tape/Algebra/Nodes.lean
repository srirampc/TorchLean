/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Core.SemiringCorrectness
public import NN.Proofs.Autograd.Tape.Algebra.Soundness

/-!
# Nodes

Convenience constructors for algebraic tape nodes/graphs.

This is the "approach (a)" authoring layer: you build an SSA/DAG graph out of local nodes,
then lower it to a runtime tape via `NN/Proofs/Autograd/Runtime/Link.lean`.

The nodes here are a unary adapter for any `OpSpecCorrect` and a binary addition node. Both read
their parents through typed context indices (`Idx`) and scatter their VJPs into the context with
`TensorPack.single`.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec TorchLean
open TorchLean TorchLean.Tensor
open TensorAlgebra

noncomputable section

namespace NodeData

/-- Executable binary add node (two parents of the same shape). -/
def add {α : Type} {Δ : Type} [TorchLean.Storage α] [Zero α] [Add α]
    {Γ : List Shape} {s : Shape}
    (a b : Idx Γ s) : NodeData α Δ Γ s :=
  { forward := fun ctx _d => addSpec (getIdx (xs := ctx) a) (getIdx (xs := ctx) b)
    jvp := fun _ctx dctx _d => addSpec (getIdx (xs := dctx) a) (getIdx (xs := dctx) b)
    vjp := fun _ctx _d δ =>
      TorchLean.TensorPack.add (α := α) (ss := Γ)
        (TensorPack.single (α := α) (Γ := Γ) a δ)
        (TensorPack.single (α := α) (Γ := Γ) b δ) }

end NodeData

namespace Node

/-- Build a proof-carrying unary node from an `OpSpecCorrect`. -/
def ofOpSpecCorrect {α : Type} {Δ : Type} [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {σ τ : Shape}
    (idx : Idx Γ σ) (op : OpSpecCorrect (α := α) σ τ) : Node (α := α) (Δ := Δ) Γ τ :=
  { toNodeData :=
      { forward := fun ctx _d => op.op.forward (getIdx (xs := ctx) idx)
        jvp := fun ctx dctx _d => op.jvp (getIdx (xs := ctx) idx) (getIdx (xs := dctx) idx)
        vjp := fun ctx _d δ =>
          TensorPack.single (α := α) (Γ := Γ) idx (op.op.backward (getIdx (xs := ctx) idx) δ) }
    correct := by
      intro ctx dctx d δ
      -- Reduce to the per-op adjointness law and the `TensorPack.single` dot lemma.
      let x := getIdx (xs := ctx) idx
      let dx := getIdx (xs := dctx) idx
      have hop := op.correct x dx δ
      have hsingle :
          dot (α := α) dx (op.op.backward x δ) =
            TensorPack.dotList (α := α) dctx
              (TensorPack.single (α := α) (Γ := Γ) idx (op.op.backward x δ)) := by
        simpa using (TensorPack.dotList_single (α := α) (Γ := Γ) (dx := dctx) (idx := idx)
          (v := op.op.backward x δ)).symm
      -- `hop` gives `dot (jvp ...) δ = dot dx (backward ...)`.
      -- Rewrite the RHS into the `TensorPack.dotList` form.
      simpa [x, dx] using hop.trans hsingle }

/-- Proof-carrying binary add node (two parents of the same shape). -/
def add {α : Type} {Δ : Type} [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    Node (α := α) (Δ := Δ) Γ s :=
  { toNodeData := NodeData.add (α := α) (Δ := Δ) (Γ := Γ) (s := s) a b
    correct := by
      intro ctx dctx d δ
      -- Reduce to dot distribution and the fact that `TensorPack.single` is the adjoint of
      -- `getIdx`.
      let da := getIdx (xs := dctx) a
      let db := getIdx (xs := dctx) b
      have hsplit :
          dot (α := α) (addSpec da db) δ = dot (α := α) da δ + dot (α := α) db δ := by
        exact TensorAlgebra.dot_add_left da db δ
      have hsingleA :
          TensorPack.dotList (α := α) dctx (TensorPack.single (α := α) (Γ := Γ) a δ) =
            dot (α := α) da δ := by
        simpa [da] using
          (TensorPack.dotList_single (α := α) (Γ := Γ) (dx := dctx) (idx := a) (v := δ))
      have hsingleB :
          TensorPack.dotList (α := α) dctx (TensorPack.single (α := α) (Γ := Γ) b δ) =
            dot (α := α) db δ := by
        simpa [db] using
          (TensorPack.dotList_single (α := α) (Γ := Γ) (dx := dctx) (idx := b) (v := δ))
      have hadd :
          TensorPack.dotList (α := α) dctx
              (TorchLean.TensorPack.add (α := α) (ss := Γ)
                (TensorPack.single (α := α) (Γ := Γ) a δ)
                (TensorPack.single (α := α) (Γ := Γ) b δ))
            =
          TensorPack.dotList (α := α) dctx (TensorPack.single (α := α) (Γ := Γ) a δ) +
            TensorPack.dotList (α := α) dctx (TensorPack.single (α := α) (Γ := Γ) b δ) := by
        simpa using
          (TensorPack.dotList_add_right (α := α) (ss := Γ) (x := dctx)
            (y := TensorPack.single (α := α) (Γ := Γ) a δ)
            (z := TensorPack.single (α := α) (Γ := Γ) b δ))
      calc
        dot (α := α) (NodeData.add (α := α) (Δ := Δ) (Γ := Γ) (s := s) a b |>.jvp ctx dctx d) δ
            = dot (α := α) (addSpec da db) δ := by
                simp [NodeData.add, da, db]
        _ = dot (α := α) da δ + dot (α := α) db δ := hsplit
        _ = TensorPack.dotList (α := α) dctx (TensorPack.single (α := α) (Γ := Γ) a δ) +
              TensorPack.dotList (α := α) dctx (TensorPack.single (α := α) (Γ := Γ) b δ) := by
                simp [hsingleA, hsingleB]
        _ = TensorPack.dotList (α := α) dctx
              (NodeData.add (α := α) (Δ := Δ) (Γ := Γ) (s := s) a b |>.vjp ctx d δ) := by
                simp [NodeData.add, hadd] }

end Node

end
end Algebra
end Autograd
end Proofs
