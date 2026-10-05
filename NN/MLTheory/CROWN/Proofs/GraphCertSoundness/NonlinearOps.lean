/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.Spec.Core.Context.Real

/-!
# Nonlinear IBP Soundness Lemmas

Monotonicity and Lipschitz facts for the nonlinear graph operations handled by the IBP certificate
soundness theorem.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-- Lift a scalar enclosure through a shape-preserving elementwise map. -/
private theorem mapBounds_sound_real {s : Shape}
    (f : ℝ → ℝ) (lower upper : ℝ → ℝ → ℝ)
    (hscalar : ∀ l u v, l ≤ v ∧ v ≤ u → lower l u ≤ f v ∧ f v ≤ upper l u)
    (xB : Box ℝ s) (x : Tensor ℝ s) (hx : Box.contains xB x) :
    Box.contains
      ⟨Tensor.map2Spec lower xB.lo xB.hi, Tensor.map2Spec upper xB.lo xB.hi⟩
      (Tensor.mapSpec f x) := by
  induction s with
  | scalar =>
      simpa only [Box.contains, Tensor.toScalar_map2Spec, Tensor.toScalar_mapSpec]
        using hscalar xB.lo.item xB.hi.item x.item hx
  | dim n inner ih =>
      intro i
      change Box.contains
        ⟨(Tensor.map2Spec lower xB.lo xB.hi).unstack i,
          (Tensor.map2Spec upper xB.lo xB.hi).unstack i⟩
        ((Tensor.mapSpec f x).unstack i)
      rw [show (Tensor.map2Spec lower xB.lo xB.hi).unstack i =
          Tensor.map2Spec lower (xB.lo.unstack i) (xB.hi.unstack i) from
            (TorchLean.Tensor.Internal.Rep.zipWith_unstack lower xB.lo xB.hi i).symm,
        show (Tensor.map2Spec upper xB.lo xB.hi).unstack i =
          Tensor.map2Spec upper (xB.lo.unstack i) (xB.hi.unstack i) from
            (TorchLean.Tensor.Internal.Rep.zipWith_unstack upper xB.lo xB.hi i).symm,
        show (Tensor.mapSpec f x).unstack i = Tensor.mapSpec f (x.unstack i) from
          (TorchLean.Tensor.Internal.Rep.map_unstack f x i).symm]
      exact ih ⟨xB.lo.unstack i, xB.hi.unstack i⟩ (x.unstack i) (hx i)

/-!
### Soundness of `Runtime.Ops.IBP.mapMinmax` for monotone scalar functions

`Runtime.Ops.IBP.sigmoid` and `Runtime.Ops.IBP.tanh` are defined using `mapMinmax`.
If the activation is monotone, then the min/max of the endpoints is a correct enclosure.
-/

theorem map_minmax_sound_real {s : Shape} (f : ℝ → ℝ) (hf : Monotone f)
    (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (Runtime.Ops.IBP.mapMinmax f xB)
      (Tensor.mapSpec f x) := by
  refine mapBounds_sound_real f _ _ ?_ xB x hx
  intro l u v hv
  have hflfu : f l ≤ f u := hf (le_trans hv.1 hv.2)
  have hnot : ¬f l > f u := not_lt_of_ge hflfu
  simpa only [ite_eq_right hnot] using And.intro (hf hv.1) (hf hv.2)

/-!
### Soundness of the 1-Lipschitz `sin`/`cos` enclosures

`Runtime.Ops.IBP.sin` / `Runtime.Ops.IBP.cos` use a midpoint enclosure with radius `r=(u-l)/2`,
clamped to `[-1,1]`. This avoids periodic case splits while remaining sound: the Lipschitz constant
one (`Real.abs_sin_sub_sin_le`, `Real.abs_cos_sub_cos_le`) turns the input half-width directly into
an output half-width.
-/

/-- Interval bound propagation through `sin` is sound over `ℝ`.

Each component uses the midpoint Lipschitz enclosure, intersected with `[-1, 1]`.
-/
theorem ibp_sin_sound_real {s : Shape} (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (Runtime.Ops.IBP.sin xB)
      (Tensor.mapSpec Real.sin x) := by
  refine mapBounds_sound_real Real.sin _ _ ?_ xB x hx
  intro l u v hv
  let m : ℝ := (l + u) / 2
  let r : ℝ := (u - l) / 2
  have hxm : |v - m| ≤ r := by
    apply abs_le.2
    constructor <;> dsimp [m, r] <;> nlinarith [hv.1, hv.2]
  have hLip : |Real.sin v - Real.sin m| ≤ r :=
    le_trans (Real.abs_sin_sub_sin_le v m) hxm
  have hdiff : -r ≤ Real.sin v - Real.sin m ∧ Real.sin v - Real.sin m ≤ r :=
    abs_le.1 hLip
  have hmidLo : Real.sin m - r ≤ Real.sin v := by linarith [hdiff.1]
  have hmidHi : Real.sin v ≤ Real.sin m + r := by linarith [hdiff.2]
  have hsinRange : (-1 : ℝ) ≤ Real.sin v ∧ Real.sin v ≤ (1 : ℝ) :=
    ⟨Real.neg_one_le_sin v, Real.sin_le_one v⟩
  have hBounds :
      max (-1 : ℝ) (Real.sin m - r) ≤ Real.sin v ∧
        Real.sin v ≤ min (1 : ℝ) (Real.sin m + r) :=
    ⟨max_le_iff.2 ⟨hsinRange.1, hmidLo⟩,
      le_min_iff.2 ⟨hsinRange.2, hmidHi⟩⟩
  simpa only [MathFunctions.sin, m, r] using hBounds

/-- Interval bound propagation through `cos` is sound over `ℝ`, by the same argument. -/
theorem ibp_cos_sound_real {s : Shape} (xB : Box ℝ s) (x : Tensor ℝ s)
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (Runtime.Ops.IBP.cos xB)
      (Tensor.mapSpec Real.cos x) := by
  refine mapBounds_sound_real Real.cos _ _ ?_ xB x hx
  intro l u v hv
  let m : ℝ := (l + u) / 2
  let r : ℝ := (u - l) / 2
  have hxm : |v - m| ≤ r := by
    apply abs_le.2
    constructor <;> dsimp [m, r] <;> nlinarith [hv.1, hv.2]
  have hLip : |Real.cos v - Real.cos m| ≤ r :=
    le_trans (Real.abs_cos_sub_cos_le v m) hxm
  have hdiff : -r ≤ Real.cos v - Real.cos m ∧ Real.cos v - Real.cos m ≤ r :=
    abs_le.1 hLip
  have hmidLo : Real.cos m - r ≤ Real.cos v := by linarith [hdiff.1]
  have hmidHi : Real.cos v ≤ Real.cos m + r := by linarith [hdiff.2]
  have hcosRange : (-1 : ℝ) ≤ Real.cos v ∧ Real.cos v ≤ (1 : ℝ) :=
    ⟨Real.neg_one_le_cos v, Real.cos_le_one v⟩
  have hBounds :
      max (-1 : ℝ) (Real.cos m - r) ≤ Real.cos v ∧
        Real.cos v ≤ min (1 : ℝ) (Real.cos m + r) :=
    ⟨max_le_iff.2 ⟨hcosRange.1, hmidLo⟩,
      le_min_iff.2 ⟨hcosRange.2, hmidHi⟩⟩
  simpa only [MathFunctions.cos, m, r] using hBounds

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
