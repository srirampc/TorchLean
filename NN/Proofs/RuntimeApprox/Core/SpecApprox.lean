/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.Proofs.RuntimeApprox.Core.Tolerance
public import NN.Spec.Core.Scalar
public import NN.Spec.Core.Context.Real

/-!
# SpecApprox

Spec/runtime approximation bridge with explicit error bounds.

This is a spec-level statement: runtime values are mapped into `Real` and compared
against the spec using a chosen norm.

Trust boundary:
- This file is purely about *stating* approximation predicates. Turning it into an end-to-end
  theorem requires per-op approximation lemmas and a composition argument.
- Lean supplies a logical model for `Float`, but this file does not yet provide per-operation
  approximation lemmas for that model. Native execution also requires a separate provider
  agreement. Neither connection is assumed here.
- The intended proof-relevant path is to use rounding-model backends (FloatLib `NF`) where
  rounding error bounds are explicit and can be composed.

## PyTorch correspondence / citations
Conceptually, `approxWith` / `approxTensorWithTol` are theorem-level versions of “runtime tensor is
close to spec tensor under a chosen norm”, similar to how PyTorch uses norms and `rtol`/`atol`
style checks in
testing/validation.
https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
https://pytorch.org/docs/stable/generated/torch.allclose.html
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open NN.MLTheory.Robustness.Spec

noncomputable section

/-- Convert a runtime tensor into the spec scalar by mapping a scalar function. -/
def tensorToSpec {α : Type} [TorchLean.Storage α] {s : Shape} (toSpec : α → SpecScalar)
    (t : Tensor α s) : SpecTensor s :=
  TorchLean.Tensor.map toSpec t

/-- Linf norm on spec tensors. -/
def linfNorm : ∀ {s : Shape}, SpecTensor s → SpecScalar :=
  tensorLinfNorm (α := SpecScalar)

/-- Approximation predicate with an explicit error bound. -/
def approxWith {α : Type} [TorchLean.Storage α] {s : Shape}
    (toSpec : α → SpecScalar)
    (norm : ∀ {s : Shape}, SpecTensor s → SpecScalar)
    (spec : SpecTensor s)
    (runtime : Tensor α s)
    (eps : SpecScalar) : Prop :=
  tensorDistance (α := SpecScalar) norm spec (tensorToSpec toSpec runtime) ≤ eps

/-- Abs+rel approximation predicate with a `ApproxTol` budget (scaled by `max ‖spec‖ ‖runtime‖`). -/
def approxWithTol {α : Type} [TorchLean.Storage α] {s : Shape}
    (toSpec : α → SpecScalar)
    (norm : ∀ {s : Shape}, SpecTensor s → SpecScalar)
    (spec : SpecTensor s)
    (runtime : Tensor α s)
    (tol : ApproxTol) : Prop :=
  let runtimeS := tensorToSpec toSpec runtime
  tensorDistance (α := SpecScalar) norm spec runtimeS ≤
    approxBound tol (norm spec) (norm runtimeS)

/-- Default abs+rel tensor approximation (uses `linfNorm`). -/
def approxTensorWithTol {α : Type} [TorchLean.Storage α] {s : Shape}
    (toSpec : α → SpecScalar)
    (spec : SpecTensor s)
    (runtime : Tensor α s)
    (tol : ApproxTol) : Prop :=
  approxWithTol (toSpec := toSpec) (norm := linfNorm) spec runtime tol

/-- A plain `eps` bound is an abs-only tolerance bound.

The clamping in `Real.toNNReal` only ever weakens the claim, so no sign hypothesis on `eps` is
needed in this direction; the converse `approxWithTol_absOnly_iff` does need one. -/
theorem approxWithTol_absOnly_of_approxWith {α : Type} [TorchLean.Storage α] {s : Shape}
    {toSpec : α → SpecScalar}
    {norm : ∀ {s : Shape}, SpecTensor s → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} (eps : ℝ)
    (h : approxWith (toSpec := toSpec) (norm := norm) spec runtime eps) :
    approxWithTol (toSpec := toSpec) (norm := norm) spec runtime (ApproxTol.absOnly eps) := by
  dsimp [approxWithTol]
  rw [approxBound_absOnly]
  exact le_trans h (Real.le_coe_toNNReal eps)

/-- The same lift specialized to the default `linfNorm` tensor relation. -/
theorem approxTensorWithTol_absOnly_of_approxWith {α : Type} [TorchLean.Storage α] {s : Shape}
    {toSpec : α → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} (eps : ℝ)
    (h : approxWith (toSpec := toSpec) (norm := linfNorm) spec runtime eps) :
    approxTensorWithTol (toSpec := toSpec) spec runtime (ApproxTol.absOnly eps) :=
  approxWithTol_absOnly_of_approxWith (toSpec := toSpec) (norm := linfNorm) eps h

/-- Conversely, a tolerance bound is a plain bound at the tolerance's own evaluated budget. -/
theorem approxWith_of_approxWithTol {α : Type} [TorchLean.Storage α] {s : Shape}
    {toSpec : α → SpecScalar}
    {norm : ∀ {s : Shape}, SpecTensor s → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} {tol : ApproxTol}
    (h : approxWithTol (toSpec := toSpec) (norm := norm) spec runtime tol) :
    approxWith (toSpec := toSpec) (norm := norm) spec runtime
      (approxBound tol (norm spec) (norm (tensorToSpec toSpec runtime))) :=
  h

/-- Tensor approximation is preserved when the tolerance is weakened in any field. -/
theorem approxWithTol_mono {α : Type} [TorchLean.Storage α] {s : Shape}
    {toSpec : α → SpecScalar}
    {norm : ∀ {s : Shape}, SpecTensor s → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} {tol₁ tol₂ : ApproxTol}
    (habs : tol₁.abs ≤ tol₂.abs) (hrel : tol₁.rel ≤ tol₂.rel) (hslack : tol₁.slack ≤ tol₂.slack)
    (h : approxWithTol (toSpec := toSpec) (norm := norm) spec runtime tol₁) :
    approxWithTol (toSpec := toSpec) (norm := norm) spec runtime tol₂ :=
  le_trans h (approxBound_mono habs hrel hslack _ _)

/-- For nonnegative `eps` the two formulations coincide, so nothing is lost by working with
whichever is convenient at each step. -/
theorem approxWithTol_absOnly_iff {α : Type} [TorchLean.Storage α] {s : Shape}
    {toSpec : α → SpecScalar}
    {norm : ∀ {s : Shape}, SpecTensor s → SpecScalar}
    {spec : SpecTensor s} {runtime : Tensor α s} {eps : ℝ} (heps : 0 ≤ eps) :
    approxWithTol (toSpec := toSpec) (norm := norm) spec runtime (ApproxTol.absOnly eps) ↔
      approxWith (toSpec := toSpec) (norm := norm) spec runtime eps := by
  -- With no relative part, `absOnly eps` evaluates to the constant budget `eps`.
  simp [approxWithTol, approxWith, approxBound_absOnly, Real.coe_toNNReal eps heps]

/-! ## Notation

Use `open scoped ApproxTol` to enable:

`spec ≈ᵀ[toSpec, tol] runtime` meaning: `approxTensorWithTol toSpec spec runtime tol`.
-/

scoped[ApproxTol] notation:50 spec " ≈ᵀ[" toSpec ", " tol "] " runtime =>
  Proofs.RuntimeApprox.approxTensorWithTol (toSpec := toSpec) spec runtime tol

/-- Packaged approximation witness (defaults to Linf on spec tensors). -/
structure Witness (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Map a runtime scalar into the specification scalar domain. -/
  toSpec : α → SpecScalar
  /-- Specification tensor. -/
  spec : SpecTensor s
  /-- Runtime tensor being compared with the specification tensor. -/
  runtime : Tensor α s
  /-- Absolute error budget for the Linf comparison. -/
  eps : SpecScalar
  /-- Checked approximation statement connecting `spec` and `runtime`. -/
  bound : approxWith (α := α) (toSpec := toSpec) (norm := linfNorm) spec runtime eps

end

end RuntimeApprox
end Proofs
