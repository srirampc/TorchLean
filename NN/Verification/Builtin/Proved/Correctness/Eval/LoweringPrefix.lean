/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.WellFormed

/-!
# Lowered Forward Evaluation: Prefix Preservation

Lowering a forward let-chain only appends IR nodes and only inserts payload at fresh node ids.
These lemmas record that every node and payload entry below the starting graph size is unchanged
by lowering the rest of the chain.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin

/-- The lowering accumulator after appending the lowering of one node at the fresh id. -/
def lowerStep
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {mid : Shape}
    (node : Node α paramShapes inShape ss mid) (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α) : NN.Verification.Builtin.LoweredIR α :=
  let res := lowerNode (α := α) c.graph.nodes.size node params c.ps
  { c with
      graph := { nodes := c.graph.nodes.push res.1 }
      ps := res.2
      outputId := c.graph.nodes.size }

/-- Lowering a `let1` chain lowers the head node and continues from the extended accumulator. -/
theorem lowerForwardLetChain_let1
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {mid out : Shape}
    (node : Node α paramShapes inShape ss mid)
    (gNext : ForwardLetChain α paramShapes inShape (ss ++ [mid]) out)
    (params : TorchLean.TensorPack α paramShapes) (c : NN.Verification.Builtin.LoweredIR α) :
    lowerForwardLetChain (α := α) (ForwardLetChain.let1 node gNext) params c =
      lowerForwardLetChain (α := α) gNext params (lowerStep node params c) := by
  rfl


/--
Generic prefix-preservation argument for `ParamStore` lookups.

The forward lowering pass appends exactly one fresh IR node at each let-binding. Any payload
lookup that is preserved by a single `lowerNode` step for keys below the fresh id is therefore
preserved by the whole lowered suffix.
-/
private theorem lowerForwardLetChain_ps_lookup_get?_lt
    {α β : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (read : NN.MLTheory.CROWN.Graph.ParamStore α → Nat → Option β)
    (hStep :
      ∀ {ss₀ : List Shape} {mid₀ : Shape} {node : Node α paramShapes inShape ss₀ mid₀}
        (id k : Nat) (params : TorchLean.TensorPack α paramShapes)
        (ps : NN.MLTheory.CROWN.Graph.ParamStore α),
        k < id →
        read
            (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := ss₀) (out := mid₀) id node params ps).2 k =
          read ps k)
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    read
        (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss) (out := out) g params c).ps k =
      read c.ps k := by
  induction g generalizing c with
  | ret y =>
      simp [lowerForwardLetChain]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      rw [lowerForwardLetChain_let1]
      refine (ih (lowerStep node params c)
        (by simpa [lowerStep] using Nat.lt_succ_of_lt hk)).trans ?_
      simpa [lowerStep] using
        hStep (id := c.graph.nodes.size) (k := k) (params := params) (ps := c.ps) hk

/--
Lowering a let-chain does not change `ps.constVals` entries for keys `< c.graph.nodes.size`.
Lowering only inserts payload at the fresh node id, so older keys are unchanged.
-/
theorem lowerForwardLetChain_ps_constVals_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params c).ps.constVals.get? k = c.ps.constVals.get? k := by
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.FlatTensor α)
      (read := fun ps k => ps.constVals.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := Nat.ne_of_gt hk
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

/--
Lowering a let-chain does not change `ps.linearWB` entries for keys `< c.graph.nodes.size`.
Lowering only inserts linear payload at the fresh node id, so older keys are unchanged.
-/
theorem lowerForwardLetChain_ps_linearWB_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params c).ps.linearWB.get? k = c.ps.linearWB.get? k := by
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.MLTheory.CROWN.Graph.LinParams α)
      (read := fun ps k => ps.linearWB.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := Nat.ne_of_gt hk
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

/--
Lowering a let-chain does not change `ps.convCfg` entries for keys `< c.graph.nodes.size`.
Lowering only inserts convolution payloads at fresh node ids, so older keys are unchanged.
-/
theorem lowerForwardLetChain_ps_convCfg_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params c).ps.convCfg.get? k = c.ps.convCfg.get? k := by
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.IR.ConvParams α)
      (read := fun ps k => ps.convCfg.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := Nat.ne_of_gt hk
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_insert, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

/--
Lowering a let-chain does not change `ps.batchNormEval` entries for keys below the
starting graph size. Eval-mode BatchNorm payloads enter through the broader IR/import bridge,
not through this proved first-order fragment.
-/
theorem lowerForwardLetChain_ps_batchNormEval_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params c).ps.batchNormEval.get? k =
      c.ps.batchNormEval.get? k := by
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.IR.BatchNormEvalParams α)
      (read := fun ps k => ps.batchNormEval.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        cases node <;> simp [lowerNode])
      g params c hk

/--
Lowering a let-chain preserves LayerNorm payloads below the starting graph size.

A payload-free LayerNorm step erases only its own fresh id, preventing stale future entries from
changing the source fragment's unit-affine semantics.
-/
theorem lowerForwardLetChain_ps_layerNorm_get?_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {k : Nat} (hk : k < c.graph.nodes.size) :
    (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
      (ss := ss) (out := out) g params c).ps.layerNorm.get? k = c.ps.layerNorm.get? k := by
  exact
    lowerForwardLetChain_ps_lookup_get?_lt
      (α := α) (β := NN.IR.LayerNormParams α)
      (read := fun ps k => ps.layerNorm.get? k)
      (hStep := by
        intro ss₀ mid₀ node id k params ps hk
        have hidk : id ≠ k := Nat.ne_of_gt hk
        cases node <;>
          simp [lowerNode, Std.HashMap.getElem?_erase, beq_eq_false_iff_ne.mpr hidk])
      g params c hk

/--
`lowerForwardLetChain` does not change existing nodes at indices `< c.graph.nodes.size`.
Lowering only appends nodes, so `getNode` agrees on the prefix.
-/
theorem lowerForwardLetChain_getNode_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    {i : Nat} (hi : i < c.graph.nodes.size) :
    (NN.IR.Graph.getNode
      (g := (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss) (out := out) g params c).graph) i)
      =
    NN.IR.Graph.getNode (g := c.graph) i := by
  induction g generalizing c with
  | ret y =>
      simp [lowerForwardLetChain]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      rw [lowerForwardLetChain_let1]
      exact (ih (lowerStep node params c)
        (by simpa [lowerStep] using Nat.lt_succ_of_lt hi)).trans
          (by simpa [lowerStep] using getNode_push_lt (g := c.graph)
                (n := (lowerNode c.graph.nodes.size node params c.ps).1) hi)

/-- `lowerForwardLetChain` is monotone in `graph.nodes.size` (it only appends nodes). -/
theorem lowerForwardLetChain_nodesSize_le
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α) :
    c.graph.nodes.size ≤
      (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := out) g params c).graph.nodes.size := by
  induction g generalizing c with
  | ret y =>
      simp [lowerForwardLetChain]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      rw [lowerForwardLetChain_let1]
      exact Nat.le_trans (by simp [lowerStep]) (ih (lowerStep node params c))

end Correctness

end NN.Verification.Builtin.Proved
