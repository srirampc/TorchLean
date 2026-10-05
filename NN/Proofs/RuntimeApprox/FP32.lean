/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32
public import NN.Proofs.RuntimeApprox.Core.Tolerance

/-!
# FP32 Runtime-Approximation Bridge

`NN.Proofs.RuntimeApprox.Core.Tolerance` defines a small “close enough” relation:

* `approxR x y t`  (notation: `x ≈[t] y`)

Here we connect that generic notion of tolerance to our `FP32` rounding model.

FloatLib’s `Model.abs_roundAt_sub_le` supplies the per-operation absolute error bound

$$
|\mathrm{approx}-\mathrm{exact}|
\le \varepsilon_{32}(\mathrm{exact}).
$$

We repackage those bounds so downstream proofs can use the uniform `≈[t]` vocabulary.

Two small conventions show up everywhere below:

* We use an **absolute-only tolerance** `ApproxTol.absOnly eps`.
  That means `x ≈[absOnly eps] y` is exactly the usual bound $|y-x|\le\varepsilon$
  (see `approxR_absOnly_iff`). This matches the shape of our `FP32` theorems.

* The `FP32` “epsilon” is **value-dependent**:
  $\varepsilon=\operatorname{ulp}(\mathrm{exact})/2=\varepsilon_{32}(\mathrm{exact})$.
  This is the standard round-to-nearest error model: the ulp is smaller near 0 and grows as the
  magnitude grows. Writing it as an `≈[t]` fact makes it easy to mix with other tolerances without
  inventing yet another approximation relation.
-/

@[expose] public section

open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq


namespace TorchLean.Floats
namespace FP32

open Proofs.RuntimeApprox

/--
Helper: turn a plain absolute-error inequality into an `approxR` with an absolute-only tolerance.

Informally, if $|y-x|\le\varepsilon$ and $\varepsilon\ge 0$, then
`x ≈[absOnly eps] y`.
-/
private theorem approxR_absOnly_of_abs_sub_le {x y eps : ℝ} (heps : 0 ≤ eps)
    (h : abs (y - x) ≤ eps) : approxR x y (ApproxTol.absOnly eps) :=
  (approxR_absOnly_iff (x := x) (y := y) (eps := eps) heps).2 h

/--
Nonnegativity of the FP32 half-ULP scale `Model.epsilonAt FloatFormat.binary32`.

This is needed to use `approxR_absOnly_iff`, which requires $\varepsilon\ge 0$.
-/
private theorem epsilon_nonneg (x : ℝ) : 0 ≤ Model.epsilonAt FloatFormat.binary32 x := by
  -- Unfold to FloatLib’s `ulp` so we can reuse its nonnegativity lemma.
  unfold Model.epsilonAt Model.ulpAt
  exact div_nonneg
    (ulp.nonneg (β := binaryRadix) (fexp := (Model.fexpOf FloatFormat.binary32)) (x := x))
    (by norm_num)

/-! ## Arithmetic (one real op + one rounding step) -/

/--
`FP32` addition, stated as an `≈[t]` fact with `t = absOnly (ulp(exact)/2)`.

Read this as:

- “the exact real sum” is `a.val + b.val`,
- “the rounded result” is `(a + b).val`,
- and the two differ by at most half an ulp of the exact real sum.

Informally,
$\operatorname{val}(a)+\operatorname{val}(b)
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\operatorname{val}(a)+\operatorname{val}(b)))}
\operatorname{val}(a+b)$.
-/
theorem add_approxR (a b : FP32) :
    approxR (a.val + b.val) (a + b).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val + b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (a.val + b.val)

/--
`FP32` subtraction as an `≈[t]` fact, where `t` is the absolute-only half-ulp tolerance at the
exact difference.

Informally,
$\operatorname{val}(a)-\operatorname{val}(b)
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\operatorname{val}(a)-\operatorname{val}(b)))}
\operatorname{val}(a-b)$.
-/
theorem sub_approxR (a b : FP32) :
    approxR (a.val - b.val) (a - b).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val - b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (a.val - b.val)

/--
`FP32` multiplication as an `≈[t]` fact, where `t` is the absolute-only half-ulp tolerance at the
exact product.

Informally,
$\operatorname{val}(a)\operatorname{val}(b)
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\operatorname{val}(a)\operatorname{val}(b)))}
\operatorname{val}(ab)$.
-/
theorem mul_approxR (a b : FP32) :
    approxR (a.val * b.val) (a * b).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val * b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (a.val * b.val)

/--
`FP32` division as an `≈[t]` fact, where `t` is the absolute-only half-ulp tolerance at the exact
quotient.

Informally,
$\frac{\operatorname{val}(a)}{\operatorname{val}(b)}
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\operatorname{val}(a)/\operatorname{val}(b)))}
\operatorname{val}(a/b)$.
-/
theorem div_approxR (a b : FP32) :
    approxR (a.val / b.val) (a / b).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val / b.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (a.val / b.val)

/-! ## Transcendentals (real function + rounding) -/

/--
`FP32` `exp` is `Real.exp` followed by rounding; this is the result as an `≈[t]` statement.

Informally,
$\exp(\operatorname{val}(a))
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\exp(\operatorname{val}(a))))}
\operatorname{val}(\exp a)$.
-/
theorem exp_approxR (a : FP32) :
    approxR (Real.exp a.val) (Numerics.MathFunctions.exp a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.exp a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.exp a.val)

/--
`FP32` `tanh` as an `≈[t]` statement.

Informally,
$\tanh(\operatorname{val}(a))
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\tanh(\operatorname{val}(a))))}
\operatorname{val}(\tanh a)$.
-/
theorem tanh_approxR (a : FP32) :
    approxR (Real.tanh a.val) (Numerics.MathFunctions.tanh a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.tanh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.tanh a.val)

/--
`FP32` `log` as an `≈[t]` statement (using `Real.log` as the exact reference).

Informally,
$\log(\operatorname{val}(a))
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\log(\operatorname{val}(a))))}
\operatorname{val}(\log a)$.
-/
theorem log_approxR (a : FP32) :
    approxR (Real.log a.val) (Numerics.MathFunctions.log a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.log a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.log a.val)

/--
`FP32` `cos` as an `≈[t]` statement.

Informally,
$\cos(\operatorname{val}(a))
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\cos(\operatorname{val}(a))))}
\operatorname{val}(\cos a)$.
-/
theorem cos_approxR (a : FP32) :
    approxR (Real.cos a.val) (Numerics.MathFunctions.cos a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.cos a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.cos a.val)

/--
`FP32` `sin` as an `≈[t]` statement.

Informally,
$\sin(\operatorname{val}(a))
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\sin(\operatorname{val}(a))))}
\operatorname{val}(\sin a)$.
-/
theorem sin_approxR (a : FP32) :
    approxR (Real.sin a.val) (Numerics.MathFunctions.sin a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.sin a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.sin a.val)

/--
`FP32` `sinh` as an `≈[t]` statement.

Informally,
$\sinh(\operatorname{val}(a))
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\sinh(\operatorname{val}(a))))}
\operatorname{val}(\sinh a)$.
-/
theorem sinh_approxR (a : FP32) :
    approxR (Real.sinh a.val) (Numerics.MathFunctions.sinh a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.sinh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.sinh a.val)

/--
`FP32` `cosh` as an `≈[t]` statement.

Informally,
$\cosh(\operatorname{val}(a))
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\cosh(\operatorname{val}(a))))}
\operatorname{val}(\cosh a)$.
-/
theorem cosh_approxR (a : FP32) :
    approxR (Real.cosh a.val) (Numerics.MathFunctions.cosh a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.cosh a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.cosh a.val)

/--
`FP32` `sqrt` as an `≈[t]` statement.

Informally,
$\sqrt{\operatorname{val}(a)}
\approx_{\operatorname{absOnly}(\varepsilon_{32}(\sqrt{\operatorname{val}(a)}))}
\operatorname{val}(\sqrt a)$.
-/
theorem sqrt_approxR (a : FP32) :
    approxR (Real.sqrt a.val) (Numerics.MathFunctions.sqrt a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (Real.sqrt a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (Real.sqrt a.val)

/--
`FP32` `abs` as an `≈[t]` statement.

Informally,
$|\operatorname{val}(a)|
\approx_{\operatorname{absOnly}(\varepsilon_{32}(|\operatorname{val}(a)|))}
\operatorname{val}(|a|)$.
-/
theorem abs_approxR (a : FP32) :
    approxR (abs a.val) (Numerics.MathFunctions.abs a).val
      (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (abs a.val))) := by
  refine approxR_absOnly_of_abs_sub_le (epsilon_nonneg _) ?_
  exact Model.abs_roundAt_sub_le FloatFormat.binary32 (abs a.val)

/-! ## Examples (how this looks in practice) -/

section Examples

open scoped ApproxTol

variable (a b : FP32)

/-- The same `add_approxR` theorem, but written using the `≈[t]` notation. -/
example :
    (a.val + b.val) ≈[
      ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val + b.val))
    ] (a + b).val := by
  simpa using add_approxR (a := a) (b := b)

/--
An `≈[absOnly eps]` goal can always be unpacked back to a plain absolute error inequality.

This is useful when feeding the result into lemmas stated using `abs`, or into `linarith`.
-/
example :
    abs ((a + b).val - (a.val + b.val)) ≤
      Model.epsilonAt FloatFormat.binary32 (a.val + b.val) := by
  have happ :
      approxR (a.val + b.val) (a + b).val
        (ApproxTol.absOnly (Model.epsilonAt FloatFormat.binary32 (a.val + b.val))) :=
    add_approxR (a := a) (b := b)
  have heps :
      0 ≤ Model.epsilonAt FloatFormat.binary32 (a.val + b.val) :=
    epsilon_nonneg (x := a.val + b.val)
  -- `approxR_absOnly_iff` says `≈[absOnly eps]` is exactly `|y - x| ≤ eps`.
  have : abs ((a + b).val - (a.val + b.val)) ≤
      Model.epsilonAt FloatFormat.binary32 (a.val + b.val) :=
    (approxR_absOnly_iff (x := a.val + b.val) (y := (a + b).val)
      (eps := Model.epsilonAt FloatFormat.binary32 (a.val + b.val)) heps).1 happ
  simpa [abs_sub_comm] using this

end Examples

end FP32
end TorchLean.Floats
