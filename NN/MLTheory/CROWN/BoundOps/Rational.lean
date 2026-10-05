/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Spec.Core.Context.Rational

/-!
# Exact rational bound arithmetic

This scoped instance uses exact rational operations for both directed endpoints. Its real
interpretation laws are proved in `NN.Verification.Cert.RationalReflection`.
-/

@[expose] public section

namespace Spec.RationalAlgebraic

open NN.MLTheory.CROWN
open scoped Spec.RationalAlgebraic

/-- Rational certificate arithmetic is exact, including affine reassociation. -/
scoped instance instBoundOpsRat : BoundOps ℚ where
  addDown := (· + ·)
  addUp := (· + ·)
  subDown := (· - ·)
  subUp := (· - ·)
  mulDown := (· * ·)
  mulUp := (· * ·)
  supportsExactAffineReassociation := true

end Spec.RationalAlgebraic
