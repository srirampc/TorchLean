/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Soundness proofs for CROWN affine bounds and LiRPA verification.
-/

module

public import NN.MLTheory.CROWN.Models.Mlp

/-!
# Soundness of affine relaxations (CROWN / LiRPA)

This file proves basic inequalities and compositional lemmas used by affine-relaxation bound
propagation (LiRPA) methods, in particular the CROWN/DeepPoly family.

This file records the scalar inequalities behind the relaxations:

- soundness of affine images of scalar intervals (`affine_scalar_interval_sound`),
- soundness of the ReLU triangular upper relaxation (`relu_affine_upper_bound_sound`).

The end-to-end statement for the two-layer MLP is `NN.MLTheory.CROWN.Theorems.bound_affine_sound`
in `NN.MLTheory.CROWN.Models.Mlp`; `boundAffine` falls back to IBP there, so that theorem is sound
but not tight.

For certificate-checking theorems over the graph dialect (the form used by TorchLean verification
examples), see:

- `NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness`
- `NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness`

## References

- CROWN: Zhang et al., *Efficient Neural Network Robustness Certification with General Activation
  Functions*, NeurIPS 2018. (arXiv:1811.00866)
- β-CROWN / α/β-CROWN: Wang et al., *Beta-CROWN: Efficient Bound Propagation with Provable
  Guarantees*, NeurIPS 2021. (arXiv:2103.06624)
- `auto_LiRPA`: https://github.com/Verified-Intelligence/auto_LiRPA
- `alpha-beta-CROWN`: https://github.com/Verified-Intelligence/alpha-beta-CROWN
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Soundness

/--
If $x\in[\ell,h]$, then the affine function $f(x)=ax+b$ satisfies

$$
f(x)\in[\min(a\ell+b,ah+b),\max(a\ell+b,ah+b)].
$$
-/
theorem affine_scalar_interval_sound (a b x lo hi : ℝ)
    (hx : lo ≤ x ∧ x ≤ hi) :
    min (a * lo + b) (a * hi + b) ≤ a * x + b ∧
    a * x + b ≤ max (a * lo + b) (a * hi + b) := by
  have hle : lo ≤ hi := le_trans hx.1 hx.2
  constructor
  · -- Lower bound
    by_cases ha : a ≥ 0
    · -- Case: a ≥ 0, so a*lo ≤ a*x ≤ a*hi
      have h1 : a * lo ≤ a * x := mul_le_mul_of_nonneg_left hx.1 ha
      have h2 : a * lo ≤ a * hi := mul_le_mul_of_nonneg_left hle ha
      have min_eq : min (a * lo + b) (a * hi + b) = a * lo + b := by
        apply min_eq_left; linarith
      rw [min_eq]; linarith
    · -- Case: a < 0, so a*hi ≤ a*x ≤ a*lo
      push Not at ha
      have h1 : a * hi ≤ a * x := mul_le_mul_of_nonpos_left hx.2 (le_of_lt ha)
      have h2 : a * hi ≤ a * lo := mul_le_mul_of_nonpos_left hle (le_of_lt ha)
      have min_eq : min (a * lo + b) (a * hi + b) = a * hi + b := by
        apply min_eq_right; linarith
      rw [min_eq]; linarith
  · -- Upper bound
    by_cases ha : a ≥ 0
    · -- Case: a ≥ 0
      have h1 : a * x ≤ a * hi := mul_le_mul_of_nonneg_left hx.2 ha
      have h2 : a * lo ≤ a * hi := mul_le_mul_of_nonneg_left hle ha
      have max_eq : max (a * lo + b) (a * hi + b) = a * hi + b := by
        apply max_eq_right; linarith
      rw [max_eq]; linarith
    · -- Case: a < 0
      push Not at ha
      have h1 : a * x ≤ a * lo := mul_le_mul_of_nonpos_left hx.1 (le_of_lt ha)
      have h2 : a * hi ≤ a * lo := mul_le_mul_of_nonpos_left hle (le_of_lt ha)
      have max_eq : max (a * lo + b) (a * hi + b) = a * lo + b := by
        apply max_eq_left; linarith
      rw [max_eq]; linarith

/--
The ReLU upper affine bound is sound. For $\operatorname{ReLU}(z)$ with $z\in[l,u]$,
the triangular relaxation provides an upper bound.

The CROWN relaxation uses:

- If $l\ge 0$: slope $=1$, bias $=0$; ReLU is the identity.
- If $u\le 0$: slope $=0$, bias $=0$; ReLU is zero.
- If $l<0<u$: slope $=\frac{u}{u-l}$ and bias $=-\frac{lu}{u-l}$, the linear upper envelope.
-/
theorem relu_affine_upper_bound_sound (z l u : ℝ) (hz : l ≤ z ∧ z ≤ u) :
    let relu_z := max 0 z
    let slope := if l ≥ 0 then 1 else if u ≤ 0 then 0 else u / (u - l)
    let bias := if l ≥ 0 then 0 else if u ≤ 0 then 0 else -l * u / (u - l)
    relu_z ≤ slope * z + bias := by
  -- Case analysis on the sign of [l, u]
  by_cases hl : l ≥ 0
  · -- Case 1: l ≥ 0 (both endpoints non-negative)
    -- ReLU is identity, slope = 1, bias = 0
    have hz_nonneg : 0 ≤ z := le_trans hl hz.1
    show max 0 z ≤ (if l ≥ 0 then 1 else if u ≤ 0 then 0 else u / (u - l)) * z +
                    (if l ≥ 0 then 0 else if u ≤ 0 then 0 else -l * u / (u - l))
    simp only [hl, ite_true]
    rw [max_eq_right hz_nonneg, one_mul, add_zero]
  · -- l < 0
    push Not at hl
    by_cases hu : u ≤ 0
    · -- Case 2: u ≤ 0 (both endpoints non-positive)
      -- ReLU is zero, slope = 0, bias = 0
      have hz_nonpos : z ≤ 0 := le_trans hz.2 hu
      show max 0 z ≤ (if l ≥ 0 then 1 else if u ≤ 0 then 0 else u / (u - l)) * z +
                      (if l ≥ 0 then 0 else if u ≤ 0 then 0 else -l * u / (u - l))
      simp only [not_le.mpr hl, ite_false, hu, ite_true]
      rw [max_eq_left hz_nonpos, zero_mul, add_zero]
    · -- Case 3: l < 0 < u (interval crosses zero)
      push Not at hu
      -- Use triangular relaxation: slope = u/(u-l), bias = -l*u/(u-l)
      show max 0 z ≤ (if l ≥ 0 then 1 else if u ≤ 0 then 0 else u / (u - l)) * z +
                      (if l ≥ 0 then 0 else if u ≤ 0 then 0 else -l * u / (u - l))
      simp only [not_le.mpr hl, ite_false, not_le.mpr hu, ite_false]

      have h_ul_pos : 0 < u - l := by linarith

      by_cases hz_neg : z ≤ 0
      · -- If z ≤ 0, then max(0,z) = 0
        -- Need: 0 ≤ u/(u-l) * z - l*u/(u-l) = u*(z-l)/(u-l)
        rw [max_eq_left hz_neg]
        have eq : u / (u - l) * z + -l * u / (u - l) = u * (z - l) / (u - l) := by
          field_simp; ring
        rw [eq]
        apply div_nonneg
        · apply mul_nonneg; linarith; linarith [hz.1]
        · linarith
      · -- If z > 0, then max(0,z) = z
        -- Need: z ≤ u/(u-l) * z - l*u/(u-l)
        push Not at hz_neg
        rw [max_eq_right (le_of_lt hz_neg)]
        -- Simplify the RHS and prove z ≤ (u*z - l*u)/(u-l)
        -- This is equivalent to z*(u-l) ≤ u*z - l*u
        have eq : u / (u - l) * z + -l * u / (u - l) = (u * z - l * u) / (u - l) := by
          field_simp; ring
        -- Clearing the positive denominator leaves `l * u ≤ l * z`, from `l < 0` and `z ≤ u`.
        rw [eq, le_div_iff₀ h_ul_pos]
        linarith [mul_le_mul_of_nonpos_left hz.2 (le_of_lt hl)]

end NN.MLTheory.CROWN.Soundness
