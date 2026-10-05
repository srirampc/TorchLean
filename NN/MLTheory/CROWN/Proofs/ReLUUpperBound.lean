/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.Spec.Core.Context.Real

/-!
# Scalar ReLU Upper Bound

The CROWN affine upper relaxation bounds ReLU throughout its input interval. The MLP, graph,
and phase-aware certificate proofs all use this scalar inequality.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Proofs

open Spec TorchLean

/-- The CROWN upper chord dominates ReLU on the interval used to construct it. -/
theorem relu_relax_scalar_upper_real_runtime
    (l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
    let rp := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) l u
    Activation.Math.reluSpec (α := ℝ) x ≤ rp.slope * x + rp.bias := by
  unfold NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · have hxpos : 0 < x := lt_of_lt_of_le hlpos hlx
      have hxnonneg : 0 ≤ x := le_of_lt hxpos
      simp [hu, hlpos, Activation.Math.reluSpec_eq_max, max_eq_left hxnonneg]
    · have hle0 : l ≤ 0 := le_of_not_gt hlpos
      have hden : 0 < (u - l) := by linarith
      simp only [hu, hlpos, ite_true, ite_false]
      by_cases hxpos : 0 < x
      · have hxnonneg : 0 ≤ x := le_of_lt hxpos
        simp [Activation.Math.reluSpec_eq_max, max_eq_left hxnonneg]
        -- Triangular relaxation: after clearing `u - l > 0` the claim is `l * (u - x) ≤ 0`.
        have hx_to_goal : x ≤ u / (u - l) * (x - l) := by
          rw [div_mul_eq_mul_div, le_div_iff₀ hden]
          linarith [mul_nonpos_of_nonpos_of_nonneg hle0 (sub_nonneg.mpr hxu)]
        have h2 : u / (u - l) * (x - l) = u / (u - l) * x + -(u / (u - l)) * l := by ring
        simpa [h2] using hx_to_goal
      · have hxle : x ≤ 0 := le_of_not_gt hxpos
        have h1 : u / (u - l) * x + -(u / (u - l) * l) = u / (u - l) * (x - l) := by ring
        have : 0 ≤ u / (u - l) * (x - l) := by
          apply mul_nonneg
          · have : 0 ≤ u := le_of_lt hu
            exact div_nonneg this (le_of_lt hden)
          · linarith
        simpa [Activation.Math.reluSpec_eq_max, max_eq_right hxle, h1] using this
  · have hxle : x ≤ 0 := le_trans hxu (le_of_not_gt hu)
    simp [hu, Activation.Math.reluSpec_eq_max, max_eq_right hxle]

end NN.MLTheory.CROWN.Proofs
