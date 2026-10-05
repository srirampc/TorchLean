/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Softmax

/-!
# LogSoftmax

Fréchet-derivative facts for **log-softmax** on Euclidean vectors.

This is the analytic ($\mathbb R$) ingredient used to justify `logSoftmax` nodes
(`Vec n → Vec n`) in the tape/DAG autograd proofs.

## References
- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018).
- PyTorch docs for naming/behavior alignment (not used for theorems):
  https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open scoped BigOperators

noncomputable section

/--
Log-softmax on Euclidean vectors.

For `n = succ _`:
`logSoftmaxVec x i = xᵢ - log(sumExp x)`.

The `n = 0` branch is the identity on the trivial space.
-/
def logSoftmaxVec : {n : Nat} → Vec n → Vec n
  | 0, x => x
  | Nat.succ n, x =>
      vecOfFun (n := Nat.succ n) fun i => x i - Real.log (sumExp (n := Nat.succ n) x)

/--
The `i`th output coordinate of the log-softmax derivative at `x` (for `n = succ _`).

If `y = softmaxVec x`, then this is the linear functional `dx ↦ dxᵢ - ⟪y, dx⟫`.
-/
def logSoftmaxDerivCoord {n : Nat} (x : Vec (Nat.succ n)) (i : Fin (Nat.succ n)) :
    Vec (Nat.succ n) →L[ℝ] ℝ :=
  let y := softmaxVec (n := Nat.succ n) x
  evalCLM (n := Nat.succ n) i - dotCLM (n := Nat.succ n) y

/-- The full Fréchet derivative of `logSoftmaxVec` at `x`, packaged as a CLM. -/
def logSoftmaxDerivCLM : {n : Nat} → Vec n → Vec n →L[ℝ] Vec n
  | 0, _x => (1 : (Vec 0) →L[ℝ] (Vec 0))
  | Nat.succ n, x =>
      (euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap.comp <|
        ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i)

/--
Closed-form JVP (directional derivative) for log-softmax.

For `n = succ _`, if `y = softmaxVec x` and `s = ⟪y, dx⟫`, then `(logSoftmaxJvp x dx)ᵢ = dxᵢ - s`.
-/
def logSoftmaxJvp : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, dx => dx
  | Nat.succ n, x, dx =>
      let y := softmaxVec (n := Nat.succ n) x
      let s : ℝ := dotCLM (n := Nat.succ n) y dx
      vecOfFun (n := Nat.succ n) fun i => dx i - s

/--
Closed-form VJP for log-softmax (transpose-Jacobian product).

For `n = succ _`, if `y = softmaxVec x` and `t = ∑ᵢ δᵢ`, then `(logSoftmaxVjp x δ)ᵢ = δᵢ - yᵢ * t`.
-/
def logSoftmaxVjp : {n : Nat} → Vec n → Vec n → Vec n
  | 0, _x, δ => δ
  | Nat.succ n, x, δ =>
      let y := softmaxVec (n := Nat.succ n) x
      let t : ℝ := ∑ i : Fin (Nat.succ n), δ i
      vecOfFun (n := Nat.succ n) fun i => δ i - y i * t

/-- The closed-form JVP `logSoftmaxJvp` agrees with the CLM derivative `logSoftmaxDerivCLM`. -/
theorem logSoftmaxJvp_eq_deriv {n : Nat} (x dx : Vec n) :
    logSoftmaxJvp (n := n) x dx = (logSoftmaxDerivCLM (n := n) x) dx := by
  classical
  cases n with
  | zero =>
      ext i
      exact i.elim0
  | succ n =>
      ext i
      have hR :
          ((logSoftmaxDerivCLM (n := Nat.succ n) x) dx) i =
            (logSoftmaxDerivCoord (n := n) x i) dx := by
        simp [logSoftmaxDerivCLM, logSoftmaxDerivCoord, euclideanEquiv]
      rw [hR]
      simp [logSoftmaxJvp, logSoftmaxDerivCoord, dotCLM_apply, evalCLM_apply,
        sub_eq_add_neg, mul_comm]

/--
Adjointness identity: the log-softmax JVP and VJP are adjoint w.r.t. the Euclidean inner product.

This is the analytic statement that justifies using `logSoftmaxVjp` as backward.
-/
theorem inner_logSoftmaxJvp_vjp {n : Nat} (x dx δ : Vec n) :
    inner ℝ (logSoftmaxJvp (n := n) x dx) δ = inner ℝ dx (logSoftmaxVjp (n := n) x δ) := by
  classical
  cases n with
  | zero =>
      simp [logSoftmaxJvp, logSoftmaxVjp]
  | succ n =>
      let y : Vec (Nat.succ n) := softmaxVec (n := Nat.succ n) x
      have hjvp (i : Fin (Nat.succ n)) :
          logSoftmaxJvp (n := Nat.succ n) x dx i = dx i - ∑ j, y j * dx j := by
        simp [logSoftmaxJvp, y]
      have hvjp (i : Fin (Nat.succ n)) :
          logSoftmaxVjp (n := Nat.succ n) x δ i = δ i - y i * ∑ j, δ j := by
        simp [logSoftmaxVjp, y]
      rw [inner_eq_sum_mul, inner_eq_sum_mul]
      simp only [hjvp, hvjp]
      -- Both sides equal `∑ᵢ dxᵢ δᵢ - (∑ⱼ yⱼ dxⱼ) (∑ⱼ δⱼ)`.
      have hleft :
          (∑ i, (dx i - ∑ j, y j * dx j) * δ i) =
            (∑ i, dx i * δ i) - (∑ j, y j * dx j) * ∑ i, δ i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
      have hright :
          (∑ i, dx i * (δ i - y i * ∑ j, δ j)) =
            (∑ i, dx i * δ i) - (∑ j, δ j) * ∑ i, y i * dx i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun i _ => by ring
      rw [hleft, hright]
      ring

/-- Log-softmax is Fréchet-differentiable everywhere, with derivative `logSoftmaxDerivCLM`. -/
theorem hasFDerivAt_logSoftmaxVec {n : Nat} (x : Vec n) :
    HasFDerivAt (logSoftmaxVec (n := n)) (logSoftmaxDerivCLM (n := n) x) x := by
  classical
  cases n with
  | zero =>
      have hD : logSoftmaxDerivCLM (n := 0) x = (1 : (Vec 0) →L[ℝ] (Vec 0)) := rfl
      rw [hD]
      change HasFDerivAt (fun x : Vec 0 => x) (1 : (Vec 0) →L[ℝ] (Vec 0)) x
      exact ((1 : (Vec 0) →L[ℝ] (Vec 0)).hasFDerivAt (x := x))
  | succ n =>
      -- Derivative of `sumExp`, a finite sum of the coordinate exponentials `x ↦ exp (x j)`.
      let sumDeriv : Vec (Nat.succ n) →L[ℝ] ℝ :=
        ∑ j : Fin (Nat.succ n), Real.exp (x j) • evalCLM (n := Nat.succ n) j
      have hsumF : HasFDerivAt (sumExp (n := Nat.succ n)) sumDeriv x :=
        hasFDerivAt_sumExp x

      -- Derivative of `x ↦ log(sumExp x)`.
      let y : Vec (Nat.succ n) := softmaxVec (n := Nat.succ n) x
      let logDeriv : Vec (Nat.succ n) →L[ℝ] ℝ :=
        (dotCLM (n := Nat.succ n) y)
      have hlog :
          HasFDerivAt (fun x : Vec (Nat.succ n) => Real.log (sumExp (n := Nat.succ n) x)) logDeriv x
            := by
        -- `(∑ⱼ exp(xⱼ) dxⱼ) / sumExp x = ∑ⱼ softmax(x)ⱼ dxⱼ`.
        have hlin : (sumExp (n := Nat.succ n) x)⁻¹ • sumDeriv = logDeriv := by
          ext dx
          simp [logDeriv, y, sumDeriv, dotCLM_apply, softmaxVec, sumExp, div_eq_mul_inv,
            Finset.mul_sum, mul_assoc, mul_left_comm, mul_comm]
        exact (hsumF.log (sumExp_ne_zero (n := n) x)).congr_fderiv hlin

      -- Assemble coordinatewise via `hasFDerivAt_pi`, then transport through `euclideanEquiv`.
      have hcoord :
          ∀ i : Fin (Nat.succ n),
            HasFDerivAt (fun x : Vec (Nat.succ n) => (logSoftmaxVec (n := Nat.succ n) x).ofLp i)
              (logSoftmaxDerivCoord (n := n) x i) x := by
        intro i
        have hid : HasFDerivAt (fun x : Vec (Nat.succ n) => x i) (evalCLM (n := Nat.succ n) i) x :=
          (evalCLM (n := Nat.succ n) i).hasFDerivAt
        have hsub := hid.sub hlog
        have hfun :
            ((fun x : Vec (Nat.succ n) => x.ofLp i) -
              fun x : Vec (Nat.succ n) => Real.log (sumExp (n := Nat.succ n) x))
              =
            (fun x : Vec (Nat.succ n) => (logSoftmaxVec (n := Nat.succ n) x).ofLp i) := by
          funext x'
          simp [logSoftmaxVec, vecOfFun]
        -- derivative already matches by definition
        rw [hfun] at hsub
        simpa [logSoftmaxDerivCoord, logDeriv, y] using hsub

      have hFun :
          HasFDerivAt (fun x : Vec (Nat.succ n) => fun i : Fin (Nat.succ n) =>
            (logSoftmaxVec (n := Nat.succ n) x).ofLp i)
            (ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i))
              x := by
        refine (hasFDerivAt_pi (𝕜 := ℝ)
          (φ := fun i : Fin (Nat.succ n) => fun x : Vec (Nat.succ n) =>
            (logSoftmaxVec (n := Nat.succ n) x).ofLp i)
          (φ' := fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i)
          (x := x)).2 ?_
        intro i
        simpa using hcoord i
      have he' :
          HasFDerivAt (fun g : Fin (Nat.succ n) → ℝ => (euclideanEquiv (Nat.succ n)).symm g)
            ((euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap)
            (fun i : Fin (Nat.succ n) => (logSoftmaxVec (n := Nat.succ n) x).ofLp i) :=
        (ContinuousLinearMap.hasFDerivAt ((euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap))
      have hcomp := he'.comp x hFun
      change HasFDerivAt
        (fun x : Vec (Nat.succ n) => WithLp.toLp 2 fun i : Fin (Nat.succ n) =>
          x.ofLp i - Real.log (sumExp (n := Nat.succ n) x))
        ((euclideanEquiv (Nat.succ n)).symm.toContinuousLinearMap.comp
          (ContinuousLinearMap.pi (fun i : Fin (Nat.succ n) => logSoftmaxDerivCoord (n := n) x i)))
        x
      simpa [logSoftmaxVec, logSoftmaxDerivCLM, euclideanEquiv, Function.comp_def,
        ContinuousLinearMap.comp_apply]
        using hcomp

end
end Autograd
end Proofs
