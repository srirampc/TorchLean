/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.Numeric.Angle.Real
public import NN.Spec.Core.Context

/-!
# The real scalar dictionary

This module supplies the noncomputable `Context ℝ` instance and its compatibility laws with
Mathlib's real field. Keeping these instances separate lets callers import the general `Context`
interface without selecting the real scalar instance.
-/

@[expose] public section

/-- Full `Context` instance for `ℝ` (proof backend, noncomputable). -/
noncomputable instance : Context ℝ :=
  { defaultEpsilon := 1e-6, decidableGT := Classical.decRel _, ratCast := Rat.cast }

/-- The real dictionary uses Mathlib's field operations and has strictly positive tolerance. -/
instance : LawfulContext ℝ where
  add_eq _ _ := rfl
  mul_eq _ _ := rfl
  sub_eq _ _ := rfl
  div_eq _ _ := rfl
  neg_eq _ := rfl
  zero_eq := rfl
  one_eq := rfl
  lt_iff _ _ := Iff.rfl
  le_iff _ _ := Iff.rfl
  max_eq _ _ := rfl
  min_eq _ _ := rfl
  beq_iff _ _ := beq_iff_eq
  natCast_eq _ := rfl
  ratCast_eq _ := rfl
  abs_eq _ := rfl
  defaultEpsilon_pos := by
    show (0 : ℝ) < 1e-6
    norm_num
