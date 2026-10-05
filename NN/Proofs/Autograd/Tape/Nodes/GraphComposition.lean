/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Elementwise
public import NN.Proofs.Autograd.Tape.Nodes.Reductions

/-!
# Differentiable graph composition

The `DGraph` wrapper packages a tape graph together with node-local `NodeFDerivCorrect` proofs.
A graph is built incrementally with `DGraph.snoc`, `DGraph.append`, and `DGraph.weakenContext`,
after which `DGraph.backpropVec_eq_adjoint_fderiv` applies without threading a separate proof
object. These adapters let model-level VJP theorems reuse the proved primitive nodes rather than
reproving backprop correctness for each architecture from scratch.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

open Graph

namespace TapeNodes

/-- `takeLeftVec` packaged as a continuous linear map. -/
def takeLeftCLM (m n : Nat) : Vec (m + n) →L[ℝ] Vec m := by
  classical
  let fLin : Vec (m + n) →ₗ[ℝ] Vec m :=
    { toFun := takeLeftVec (m := m) (n := n)
      map_add' := by
        intro x y
        ext i
        simp [takeLeftVec]
      map_smul' := by
        intro r x
        ext i
        simp [takeLeftVec] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- `takeRightVec` packaged as a continuous linear map. -/
def takeRightCLM (m n : Nat) : Vec (m + n) →L[ℝ] Vec n := by
  classical
  let fLin : Vec (m + n) →ₗ[ℝ] Vec n :=
    { toFun := takeRightVec (m := m) (n := n)
      map_add' := by
        intro x y
        ext i
        simp [takeRightVec]
      map_smul' := by
        intro r x
        ext i
        simp [takeRightVec] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

end TapeNodes

/-- A tape graph bundled with the proof that every one of its nodes is analytically correct. -/
structure DGraph (Γ : List Shape) (ss : List Shape) where
  /-- The underlying tape/DAG graph. -/
  g : Graph Γ ss
  /-- Proof that every node is analytically correct (`jvp = fderiv`). -/
  hg : GraphFDerivCorrect (Γ := Γ) g

namespace DGraph

/--
Drop an unused middle context block from a graph node context.

When a graph `dg : DGraph Γ ss` is reused inside a larger context `Γ ++ extra`, a node that
originally reads `Γ ++ ss` is evaluated in the actual context `(Γ ++ extra) ++ ss`. This projection
keeps the original inputs `Γ` and the already-computed intermediates `ss`, and ignores the carried
parameters in `extra`.
-/
def dropMiddleCLM (Γ extra ss : List Shape) :
    CtxVec ((Γ ++ extra) ++ ss) →L[ℝ] CtxVec (Γ ++ ss) :=
  let splitAll :=
    (Graph.castCLM (h := ctxSize_append (Γ ++ extra) ss) :
      CtxVec ((Γ ++ extra) ++ ss) →L[ℝ] Vec (ctxSize (Γ ++ extra) + ctxSize ss))
  let baseExtra :=
    (TapeNodes.takeLeftCLM (ctxSize (Γ ++ extra)) (ctxSize ss)).comp splitAll
  let saved :=
    (TapeNodes.takeRightCLM (ctxSize (Γ ++ extra)) (ctxSize ss)).comp splitAll
  let splitBaseExtra :=
    (Graph.castCLM (h := ctxSize_append Γ extra) :
      CtxVec (Γ ++ extra) →L[ℝ] Vec (ctxSize Γ + ctxSize extra))
  let base :=
    (TapeNodes.takeLeftCLM (ctxSize Γ) (ctxSize extra)).comp (splitBaseExtra.comp baseExtra)
  (Graph.castCLM (h := (ctxSize_append Γ ss).symm)).comp
    ((Graph.appendCLM (ctxSize Γ) (ctxSize ss)).comp (base.prod saved))

/-- Unfolds the middle-dropping projection into its take, append and cast components.

Composing two tapes leaves the inner tape's private saved values in the middle of the context; this
map is what discards them, keeping the caller's context and the composed outputs. -/
@[simp] theorem dropMiddleCLM_apply {Γ extra ss : List Shape}
    (x : CtxVec ((Γ ++ extra) ++ ss)) :
    dropMiddleCLM Γ extra ss x =
      castVec (ctxSize_append Γ ss).symm
        (appendVec
          (m := ctxSize Γ) (n := ctxSize ss)
          (TapeNodes.takeLeftVec (m := ctxSize Γ) (n := ctxSize extra)
            (castVec (ctxSize_append Γ extra)
              (TapeNodes.takeLeftVec (m := ctxSize (Γ ++ extra)) (n := ctxSize ss)
                (castVec (ctxSize_append (Γ ++ extra) ss) x))))
          (TapeNodes.takeRightVec (m := ctxSize (Γ ++ extra)) (n := ctxSize ss)
            (castVec (ctxSize_append (Γ ++ extra) ss) x))) := by
  rfl

/--
Reuse a node in a context that carries extra unused inputs between the original inputs and the
current SSA intermediates.

The VJP is obtained by applying the adjoint of `dropMiddleCLM`, so gradients land only in the
original inputs and previous intermediates; the extra carried parameters receive zero contribution
from this reused node.
-/
def weakenNodeMiddle {Γ extra ss : List Shape} {τ : Shape}
    (node : Node (Γ ++ ss) τ) : Node ((Γ ++ extra) ++ ss) τ :=
  let L := dropMiddleCLM Γ extra ss
  Node.ofFn
    (Γ := (Γ ++ extra) ++ ss) (τ := τ)
    (f := fun x => node.forwardVec (Γ := Γ ++ ss) (τ := τ) (L x))
    (jvp := fun x dx => node.jvpVec (Γ := Γ ++ ss) (τ := τ) (L x) (L dx))
    (vjp := fun x δ => L.adjoint (node.vjpVec (Γ := Γ ++ ss) (τ := τ) (L x) δ))
    (correct_inner := by
      intro x dx δ
      have hnode :=
        Node.correct_inner (node := node) (L x) (L dx) δ
      have hadj :
          inner ℝ dx (L.adjoint (node.vjpVec (Γ := Γ ++ ss) (τ := τ) (L x) δ))
            =
          inner ℝ (L dx) (node.vjpVec (Γ := Γ ++ ss) (τ := τ) (L x) δ) := by
        simpa using (ContinuousLinearMap.adjoint_inner_right (A := L) (x := dx)
          (y := node.vjpVec (Γ := Γ ++ ss) (τ := τ) (L x) δ))
      exact hnode.trans hadj.symm)

/-- Transport a global node derivative certificate across `weakenNodeMiddle`. -/
def weakenNodeMiddleFDerivCorrect {Γ extra ss : List Shape} {τ : Shape}
    {node : Node (Γ ++ ss) τ} (hn : NodeFDerivCorrect node) :
    NodeFDerivCorrect (weakenNodeMiddle (Γ := Γ) (extra := extra) (ss := ss) node) := by
  let L := dropMiddleCLM Γ extra ss
  refine
    { deriv := fun x => (hn.deriv (L x)).comp L
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro x
    have hnode :
        HasFDerivAt
          (fun y => node.forwardVec (Γ := Γ ++ ss) (τ := τ) (L y))
          ((hn.deriv (L x)).comp L) x :=
      (hn.hasFDerivAt (L x)).comp x (L.hasFDerivAt (x := x))
    simpa [weakenNodeMiddle, Node.forwardVec_ofFn, L] using hnode
  · intro x dx
    simpa [weakenNodeMiddle, L, ContinuousLinearMap.comp_apply] using hn.jvp_eq (L x) (L dx)

/-- Empty differentiable graph. -/
def nil {Γ : List Shape} : DGraph Γ [] :=
  ⟨.nil, PUnit.unit⟩

/-- Append a node together with its `NodeFDerivCorrect` certificate. -/
def snoc {Γ : List Shape} {ss : List Shape} {τ : Shape}
    (dg : DGraph Γ ss) (node : Node (Γ ++ ss) τ) (hn : NodeFDerivCorrect node) :
    DGraph Γ (ss ++ [τ]) :=
  ⟨.snoc dg.g node, ⟨dg.hg, hn⟩⟩

/--
Transport a node across a definitional/context-list equality.

This is mostly used by graph composition: the second graph sees its context as `(Γ ++ ss₁) ++ ss₂`,
while the composed graph sees the same values as `Γ ++ (ss₁ ++ ss₂)`.
-/
def castNodeContext {Γ₁ Γ₂ : List Shape} {τ : Shape}
    (h : Γ₁ = Γ₂) (node : Node Γ₁ τ) : Node Γ₂ τ := by
  subst h
  exact node

/-- Transport a node F-derivative certificate along `castNodeContext`. -/
def castNodeFDerivCorrect {Γ₁ Γ₂ : List Shape} {τ : Shape}
    (h : Γ₁ = Γ₂) {node : Node Γ₁ τ} (hn : NodeFDerivCorrect node) :
    NodeFDerivCorrect (castNodeContext (τ := τ) h node) := by
  subst h
  exact hn

namespace Internal

/-- Recursive implementation for `DGraph.append`, stated over an explicit graph and proof. -/
def append {Γ : List Shape} {ss₁ ss₂ : List Shape}
    (dg₁ : DGraph Γ ss₁)
    (g₂ : Graph (Γ ++ ss₁) ss₂) (hg₂ : GraphFDerivCorrect (Γ := Γ ++ ss₁) g₂) :
    DGraph Γ (ss₁ ++ ss₂) := by
  induction g₂ with
  | nil =>
      simpa using dg₁
  | @snoc ss τ g node ih =>
      rcases hg₂ with ⟨hg, hn⟩
      let dgPrefix : DGraph Γ (ss₁ ++ ss) := ih hg
      let hctx : (Γ ++ ss₁) ++ ss = Γ ++ (ss₁ ++ ss) := by
        simp [List.append_assoc]
      let node' : Node (Γ ++ (ss₁ ++ ss)) τ := castNodeContext (τ := τ) hctx node
      let hn' : NodeFDerivCorrect node' := castNodeFDerivCorrect (τ := τ) hctx hn
      have htarget : (ss₁ ++ ss) ++ [τ] = ss₁ ++ (ss ++ [τ]) := by
        simp [List.append_assoc]
      exact htarget ▸ DGraph.snoc (dg := dgPrefix) (node := node') (hn := hn')

/-- Recursive implementation for `DGraph.weakenContext`, stated over an explicit graph and proof. -/
def weakenContext {Γ ss : List Shape} (extra : List Shape)
    (g : Graph Γ ss) (hg : GraphFDerivCorrect (Γ := Γ) g) :
    DGraph (Γ ++ extra) ss := by
  induction g with
  | nil =>
      exact DGraph.nil
  | @snoc ssPrefix τ g node ih =>
      rcases hg with ⟨hgPrefix, hn⟩
      let dgPrefix : DGraph (Γ ++ extra) ssPrefix :=
        ih hgPrefix
      exact DGraph.snoc
        (dg := dgPrefix)
        (node := weakenNodeMiddle (Γ := Γ) (extra := extra) (ss := ssPrefix) node)
        (hn := weakenNodeMiddleFDerivCorrect (Γ := Γ) (extra := extra) (ss := ssPrefix) hn)

end Internal

/--
Append a proof-carrying graph after another proof-carrying graph.

If `dg₁ : DGraph Γ ss₁` has already computed some SSA values, then a second graph
`dg₂ : DGraph (Γ ++ ss₁) ss₂` may use both the original inputs and those saved values. `append`
turns the pair into one `DGraph Γ (ss₁ ++ ss₂)`.

This is the general composition adapter needed for model-level proofs: residual attention can feed
LayerNorm, a recurrent cell can feed the next unrolled step, and larger blocks can be assembled
while reusing the existing node-level correctness proofs.
-/
def append {Γ : List Shape} {ss₁ ss₂ : List Shape}
    (dg₁ : DGraph Γ ss₁) (dg₂ : DGraph (Γ ++ ss₁) ss₂) :
    DGraph Γ (ss₁ ++ ss₂) :=
  Internal.append (Γ := Γ) (ss₁ := ss₁) (ss₂ := ss₂) dg₁ dg₂.g dg₂.hg

/--
Run a proof-carrying graph while carrying extra unused inputs.

If `dg : DGraph Γ ss`, then `weakenContext dg extra : DGraph (Γ ++ extra) ss` evaluates the same
nodes while preserving an enlarged input context. Each reused node sees the projection
`Γ ++ ss_so_far` of the actual context `(Γ ++ extra) ++ ss_so_far`; gradients are inserted back by
the adjoint projection, so the carried extras receive no gradient contribution from nodes that do
not read them.
-/
def weakenContext {Γ ss : List Shape} (dg : DGraph Γ ss) (extra : List Shape) :
    DGraph (Γ ++ extra) ss :=
  Internal.weakenContext (Γ := Γ) (ss := ss) extra dg.g dg.hg

/--
End-to-end analytic theorem for bundled graphs.

This is just `Graph.backpropVec_eq_adjoint_fderiv` with the bundled proof `dg.hg`.
-/
theorem backpropVec_eq_adjoint_fderiv
    {Γ : List Shape} {ss : List Shape} (dg : DGraph Γ ss) :
    ∀ (xV : CtxVec Γ) (seedV : CtxVec (Γ ++ ss)),
      Graph.backpropVec (Γ := Γ) (ss := ss) dg.g xV seedV
        =
      (fderiv ℝ (Graph.evalVec (Γ := Γ) (ss := ss) dg.g) xV).adjoint seedV :=
  Graph.backpropVec_eq_adjoint_fderiv (Γ := Γ) (ss := ss) dg.g dg.hg

end DGraph

end

end Autograd
end Proofs
