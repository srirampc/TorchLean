/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Elementwise

/-!
# Softmax

Fréchet-derivative facts for **axis softmax** on Euclidean vectors.

This is the analytic ($\mathbb R$) ingredient used to justify attention-style row softmax nodes
(`Vec n → Vec n`) in the tape/DAG autograd proofs.

## References
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018).
- The Matrix Cookbook (softmax Jacobian identities / vector calculus conventions).
- PyTorch docs for naming/behavior alignment (not used for theorems):
  https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open scoped BigOperators

noncomputable section

/--
`sumExp x = ∑ᵢ exp(xᵢ)`.

This is the normalizing denominator in softmax.
-/
def sumExp {n : Nat} (x : Vec n) : ℝ :=
  ∑ i : Fin n, Real.exp (x i)

/-- `sumExp x` is strictly positive when the index type is nonempty. -/
theorem sumExp_pos {n : Nat} (x : Vec (Nat.succ n)) : 0 < sumExp (n := Nat.succ n) x := by
  simpa [sumExp] using Finset.sum_pos (fun i _ => Real.exp_pos (x i)) Finset.univ_nonempty

/-- Convenience corollary: `sumExp x ≠ 0` (for `n = succ _`). -/
theorem sumExp_ne_zero {n : Nat} (x : Vec (Nat.succ n)) : sumExp (n := Nat.succ n) x ≠ 0 :=
  ne_of_gt (sumExp_pos (n := n) x)

/-- The coordinate exponential `x ↦ exp (x j)` has derivative `dx ↦ exp (x j) * dx j`. -/
theorem hasFDerivAt_exp_apply {n : Nat} (x : Vec n) (j : Fin n) :
    HasFDerivAt (fun x : Vec n => Real.exp (x j)) (Real.exp (x j) • evalCLM (n := n) j) x := by
  simpa only [evalCLM_apply] using ((evalCLM (n := n) j).hasFDerivAt (x := x)).exp

/-- `sumExp` has derivative `dx ↦ ∑ j, exp (x j) * dx j`. -/
theorem hasFDerivAt_sumExp {n : Nat} (x : Vec n) :
    HasFDerivAt (sumExp (n := n)) (∑ j : Fin n, Real.exp (x j) • evalCLM (n := n) j) x := by
  change HasFDerivAt (fun y : Vec n => ∑ j : Fin n, Real.exp (y j))
    (∑ j : Fin n, Real.exp (x j) • evalCLM (n := n) j) x
  exact HasFDerivAt.fun_sum (u := (Finset.univ : Finset (Fin n)))
    (fun j _ => hasFDerivAt_exp_apply x j)

/--
Softmax on Euclidean vectors.

For `n = succ _`:
`softmaxVec x i = exp(xᵢ) / sumExp x`.

The `n = 0` branch is the identity on the trivial space.
-/
def softmaxVec : {n : Nat} → Vec n → Vec n
  | 0, x => x
  | Nat.succ n, x => vecOfFun (n := Nat.succ n) fun i => Real.exp (x i) / sumExp x

/--
The dot-product functional `x ↦ ∑ᵢ yᵢ * xᵢ`, packaged as a continuous linear map.

This is used to express the softmax Jacobian in a clean coordinate-free way.
-/
def dotCLM {n : Nat} (y : Vec n) : Vec n →L[ℝ] ℝ :=
  ∑ j : Fin n, (evalCLM (n := n) j).smulRight (y j)

/-- The dot functional evaluates to the expected sum of products. -/
@[simp] theorem dotCLM_apply {n : Nat} (y x : Vec n) :
    dotCLM (n := n) y x = ∑ j : Fin n, y j * x j := by
  classical
  simp [dotCLM, evalCLM_apply, ContinuousLinearMap.smulRight_apply, mul_comm]

/--
The `i`th output coordinate of the softmax derivative at `x`, as a continuous linear map.

If `y = softmaxVec x`, then this is the linear functional:
`dx ↦ yᵢ * dxᵢ - yᵢ * ⟪y, dx⟫`.
-/
def softmaxDerivCoord {n : Nat} (x : Vec n) (i : Fin n) : Vec n →L[ℝ] ℝ :=
  let y := softmaxVec (n := n) x
  (evalCLM (n := n) i).smulRight (y i) - (dotCLM (n := n) y).smulRight (y i)

/-- The full Fréchet derivative of `softmaxVec` at `x`, packaged as a CLM `Vec n →L Vec n`. -/
def softmaxDerivCLM {n : Nat} (x : Vec n) : Vec n →L[ℝ] Vec n :=
  (euclideanEquiv n).symm.toContinuousLinearMap.comp <|
    ContinuousLinearMap.pi (fun i : Fin n => softmaxDerivCoord (n := n) x i)

/--
Closed-form JVP (directional derivative) for softmax.

If `y = softmaxVec x` and `s = ⟪y, dx⟫`, then `(softmaxJvp x dx)ᵢ = yᵢ * (dxᵢ - s)`.
-/
def softmaxJvp : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, dx => dx
  | Nat.succ n, x, dx =>
      let y := softmaxVec (n := Nat.succ n) x
      let s : ℝ := dotCLM (n := Nat.succ n) y dx
      vecOfFun (n := Nat.succ n) fun i => y i * (dx i - s)

/-- The closed-form JVP `softmaxJvp` agrees with the CLM derivative `softmaxDerivCLM`. -/
theorem softmaxJvp_eq_deriv {n : Nat} (x dx : Vec n) :
    softmaxJvp (n := n) x dx = (softmaxDerivCLM (n := n) x) dx := by
  classical
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      ext i
      have hR :
          ((softmaxDerivCLM (n := Nat.succ n) x) dx) i =
            (softmaxDerivCoord (n := Nat.succ n) x i) dx := by
        simp [softmaxDerivCLM, euclideanEquiv]
      rw [hR]
      simp [softmaxJvp, softmaxDerivCoord, dotCLM_apply, evalCLM_apply,
        ContinuousLinearMap.smulRight_apply, sub_eq_add_neg, mul_add, mul_comm]

/--
Self-adjointness identity for the softmax Jacobian in this inner-product encoding.

This lemma is used to show the VJP can be expressed by reusing the JVP formula.
-/
theorem inner_softmaxJvp_comm {n : Nat} (x dx δ : Vec n) :
    inner ℝ (softmaxJvp (n := n) x dx) δ = inner ℝ dx (softmaxJvp (n := n) x δ) := by
  cases n with
  | zero =>
      -- all vectors are `0`, so both sides are `0`
      simp [softmaxJvp]
  | succ n =>
      rw [inner_eq_sum_mul, inner_eq_sum_mul]
      simp only [softmaxJvp, vecOfFun_apply, dotCLM_apply]
      set y := softmaxVec x
      -- Both sides expand to `∑ yᵢ dxᵢ δᵢ - (∑ yⱼ dxⱼ) (∑ yᵢ δᵢ)`.
      have hleft :
          (∑ i, y i * (dx i - ∑ j, y j * dx j) * δ i) =
            (∑ i, y i * dx i * δ i) - (∑ j, y j * dx j) * ∑ i, y i * δ i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        apply Finset.sum_congr rfl
        intro i _
        ring
      have hright :
          (∑ i, dx i * (y i * (δ i - ∑ j, y j * δ j))) =
            (∑ i, y i * dx i * δ i) - (∑ j, y j * δ j) * ∑ i, y i * dx i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        apply Finset.sum_congr rfl
        intro i _
        ring
      rw [hleft, hright]
      ring

/-- Softmax is Fréchet-differentiable everywhere, with derivative `softmaxDerivCLM`. -/
theorem hasFDerivAt_softmaxVec {n : Nat} (x : Vec n) :
    HasFDerivAt (softmaxVec (n := n)) (softmaxDerivCLM (n := n) x) x := by
  cases n with
  | zero =>
      -- `softmaxVec` is the identity on the trivial space, and all CLMs coincide.
      have hD : softmaxDerivCLM (n := 0) x = (1 : (Vec 0) →L[ℝ] (Vec 0)) := by
        ext dx i
        exact i.elim0
      rw [hD]
      change HasFDerivAt (fun x : Vec 0 => x) (1 : (Vec 0) →L[ℝ] (Vec 0)) x
      exact ((1 : (Vec 0) →L[ℝ] (Vec 0)).hasFDerivAt (x := x))
  | succ n =>
      -- Each coordinate is the quotient `exp (x i) / sumExp x`. Differentiate it as a product
      -- with the reciprocal of the positive denominator, then assemble the coordinates.
      have hsum_ne : sumExp (n := Nat.succ n) x ≠ 0 := sumExp_ne_zero x
      have hinv := (hasFDerivAt_inv (𝕜 := ℝ) hsum_ne).comp x (hasFDerivAt_sumExp x)
      have hcoord (i : Fin (Nat.succ n)) :
          HasFDerivAt (fun y : Vec (Nat.succ n) => softmaxVec (n := Nat.succ n) y i)
            (softmaxDerivCoord (n := Nat.succ n) x i) x := by
        have hmul := (hasFDerivAt_exp_apply x i).mul hinv
        have hmap :
            Real.exp (x i) •
                ((ContinuousLinearMap.toSpanSingleton ℝ (-(sumExp x ^ 2)⁻¹)).comp
                  (∑ j : Fin (Nat.succ n), Real.exp (x j) • evalCLM (n := Nat.succ n) j)) +
              (sumExp x)⁻¹ • (Real.exp (x i) • evalCLM (n := Nat.succ n) i) =
            softmaxDerivCoord (n := Nat.succ n) x i := by
          ext dx
          -- Pull the common denominator out of the dot product before comparing coefficients.
          have hsum :
              (∑ j : Fin (Nat.succ n), Real.exp (x j) / sumExp x * dx j) =
                (∑ j : Fin (Nat.succ n), Real.exp (x j) * dx j) / sumExp x := by
            rw [div_eq_mul_inv, Finset.sum_mul]
            exact Finset.sum_congr rfl fun j _ => by ring
          simp only [_root_.add_apply, smul_apply, ContinuousLinearMap.comp_apply,
            ContinuousLinearMap.toSpanSingleton_apply, sub_apply, sum_apply, evalCLM_apply,
            dotCLM_apply, ContinuousLinearMap.smulRight_apply, smul_eq_mul, softmaxDerivCoord,
            softmaxVec, vecOfFun_apply]
          rw [hsum]
          ring
        have hfun :
            (fun y : Vec (Nat.succ n) => softmaxVec (n := Nat.succ n) y i) =
              (fun y : Vec (Nat.succ n) => Real.exp (y i)) *
                ((fun s : ℝ => s⁻¹) ∘ sumExp (n := Nat.succ n)) := by
          funext y
          simp only [softmaxVec, vecOfFun_apply, div_eq_mul_inv, Pi.mul_apply,
            Function.comp_apply]
        exact (hmul.congr_of_eventuallyEq hfun.eventuallyEq).congr_fderiv hmap
      have hpi := (hasFDerivAt_pi (𝕜 := ℝ)
        (φ := fun i (y : Vec (Nat.succ n)) => softmaxVec (n := Nat.succ n) y i)
        (φ' := fun i => softmaxDerivCoord (n := Nat.succ n) x i) (x := x)).2 hcoord
      have hfun :
          softmaxVec (n := Nat.succ n) =
            (euclideanEquiv (Nat.succ n)).symm ∘
              (fun (y : Vec (Nat.succ n)) i => softmaxVec (n := Nat.succ n) y i) := by
        funext y
        ext i
        rfl
      have hcomp :=
        (euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap.hasFDerivAt.comp x hpi
      exact hcomp.congr_of_eventuallyEq hfun.eventuallyEq

end
end Autograd
end Proofs
