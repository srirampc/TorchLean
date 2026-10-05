/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.ToDAG.Core

/-!
# Primitive Embedding

GraphSpec’s DAG language (`NN.GraphSpec.DAG`) is the “general graph” IR. To avoid duplicating the
operator semantics, unary sequential primitives

`p : Primitive ps σ τ`

are embedded into DAG primitive ops

`LowerToDAG.Primitive.toDAGPrimOp p : DAG.PrimOp (ps ++ [σ]) τ`.

This file proves the bookkeeping theorem that keeps the “no duplicated semantics” contract explicit:
if you take a sequential primitive, embed it into DAG form, and feed it the obvious argument list
`params ++ [x]`, you get exactly the same pure forward computation.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Primitive

open Spec TorchLean
open TorchLean.Tensor

/--
Embedding a sequential primitive into DAG form preserves its pure `specFwd` semantics.

This theorem states that “the DAG primitive is the sequential primitive with
its parameters made explicit as ordinary inputs”.
-/
theorem toDAGPrimOp_specFwd
    {α : Type} [TorchLean.Storage α] [Context α]
    {ps : List Shape} {σ τ : Shape}
    (p : Primitive ps σ τ)
    (params : TorchLean.TensorPack α ps) (x : TorchLean.Tensor α σ) :
    (LowerToDAG.Primitive.toDAGPrimOp (ps := ps) (σ := σ) (τ := τ) p).specFwd (α := α)
        (TorchLean.TensorPack.append (α := α)
          (ss₁ := ps) (ss₂ := [σ]) params (.cons x .nil))
    =
    p.specFwd (α := α) params x := by
  simp only [LowerToDAG.Primitive.toDAGPrimOp, TensorPack.split_append]

end Primitive
end GraphSpec
end NN
