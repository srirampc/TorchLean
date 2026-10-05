/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Operators.Activations
public import NN.MLTheory.CROWN.Runtime.Ops

/-!
# Trigonometric operator bounds (optional)

This file provides simple IBP and affine transfer rules for trigonometric functions used in
physics-informed objectives (PINNs): `sin`, `cos`, `tan`, and `atan`.

This module is not imported by the default `NN.MLTheory.CROWN.Operators` index because `tan` and
`atan` require an extra scalar-function interface. Import it explicitly when needed.

Note: `tan`/`atan` are not part of `Context` today. The corresponding transfer rules below are
therefore gated behind an extra `TanAtan α` typeclass.

## Soundness status

The transfer rules in this file are **optional** and have different proof status by operator:

- `sin`/`cos` delegate to the conservative 1-Lipschitz enclosure rules in
  `NN.MLTheory.CROWN.Runtime.Ops.IBP.sin/cos`. This avoids endpoint-only periodic reasoning, which
  can miss internal extrema.
- `tan`/`atan` require the extra `TanAtan α` class. Their endpoint rules are named with an explicit
  caller-side monotonicity assumption; for `tan`, the interval must additionally avoid poles.
- Affine transfer rules here are executable relaxation candidates. A downstream theorem should
  provide or assume the corresponding enclosure property for the scalar backend being used.

## References

- PINNs: Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", JCP 2019:
  https://arxiv.org/abs/1711.10561
- CROWN/DeepPoly context: Zhang et al., 2018 (CROWN) https://arxiv.org/abs/1811.00866
- Lipschitz enclosure idea for `sin`/`cos`:
  $\left|\frac{d}{dx}\sin x\right|\le 1$ and
  $\left|\frac{d}{dx}\cos x\right|\le 1$, so a ball or interval can be mapped using a
  Lipschitz radius (as implemented in `Runtime/Ops`).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators.Trigonometric

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Extra trigonometric functions outside the base `Spec.MathFunctions` / `Context` interface. -/
class TanAtan (α : Type) where
  /-- Tangent. -/
  tan : α → α
  /-- Arctangent. -/
  atan : α → α

/-- IBP for `sin`.

This optional operator delegates to the conservative 1-Lipschitz enclosure used by the runtime graph
verifier. We use that rule instead of endpoint-only periodic reasoning, since endpoint checks alone
can miss internal extrema without exact period tracking.
-/
def ibpSin {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  NN.MLTheory.CROWN.Runtime.Ops.IBP.sin xB

/-- IBP for `cos`.

This optional operator delegates to the same conservative 1-Lipschitz enclosure as the runtime graph
verifier.
-/
def ibpCos {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  NN.MLTheory.CROWN.Runtime.Ops.IBP.cos xB

/-- IBP for `tan` under the caller-side monotonic-branch precondition.

This rule is intended for intervals contained in a single branch of `tan` and away from poles. If a
workflow may cross a pole, it should reject the certificate or provide a different enclosure rule.
-/
def ibpTanAssumingMonotoneBranch {n : Nat} [TanAtan α]
    (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let outLo := Tensor.dim fun i =>
    -- In a monotonic region, tan(l) is the lower bound.
    Tensor.scalar (TanAtan.tan (Tensor.unstack xB.lo i).item)
  let outHi := Tensor.dim fun i =>
    -- In a monotonic region, tan(u) is the upper bound.
    Tensor.scalar (TanAtan.tan (Tensor.unstack xB.hi i).item)
  { lo := outLo, hi := outHi }

/--
Endpoint interval propagation for `atan`, under the caller-side assumption that the supplied
`TanAtan.atan` implementation is monotone on every input interval.
-/
def ibpAtanAssumingMonotone {n : Nat} [TanAtan α]
    (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let outLo := Tensor.dim fun i =>
    Tensor.scalar (TanAtan.atan (Tensor.unstack xB.lo i).item)
  let outHi := Tensor.dim fun i =>
    Tensor.scalar (TanAtan.atan (Tensor.unstack xB.hi i).item)
  { lo := outLo, hi := outHi }

/-- Multiply two scalar intervals using all four endpoint products. -/
def scalarIntervalMul (aLo aHi bLo bHi : α) : α × α :=
  let p1 := aLo * bLo
  let p2 := aLo * bHi
  let p3 := aHi * bLo
  let p4 := aHi * bHi
  (Activations.min2 (Activations.min2 p1 p2) (Activations.min2 p3 p4),
    Activations.max2 (Activations.max2 p1 p2) (Activations.max2 p3 p4))

/-- Square a scalar interval. -/
def scalarIntervalSquare (lo hi : α) : α × α :=
  let lo2 := lo * lo
  let hi2 := hi * hi
  if lo < 0 then
    if hi > 0 then
      (0, Activations.max2 lo2 hi2)
    else
      (hi2, lo2)
  else
    (lo2, hi2)

/--
Multiply a scalar interval by the unit interval `[-1, 1]`.

Every trigonometric rule in this file leans on this one operation, because `cos` and `-sin` are both
globally enclosed by `[-1, 1]` and that is the only enclosure available when the input box may span
an interior extremum. Naming it states the intent: the result is the tightest interval containing
`c * x` for every `c ∈ [-1, 1]` and every `x ∈ [lo, hi]`.
-/
@[inline] def unitIntervalMul (lo hi : α) : α × α :=
  scalarIntervalMul (-1) 1 lo hi

/--
Propagate a derivative box through a factor known only to lie in `[-1, 1]`.

Worth reading the note in `derivSin` before using this directly: it is the shared body of the `sin`
and `cos` derivative rules, not a rule of its own.
-/
def unitFactorDeriv {n : Nat} (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let outLo := Tensor.dim fun i =>
    Tensor.scalar (unitIntervalMul (Tensor.unstack dB.lo i).item (Tensor.unstack dB.hi i).item).1
  let outHi := Tensor.dim fun i =>
    Tensor.scalar (unitIntervalMul (Tensor.unstack dB.lo i).item (Tensor.unstack dB.hi i).item).2
  { lo := outLo, hi := outHi }

/--
Derivative propagation for `sin` using the global enclosure `cos(x) ∈ [-1, 1]`.

This deliberately avoids endpoint-only trigonometric bounds, which are unsound whenever an interval
contains an interior extremum.

The chain-rule factors `cos(x)` and `-sin(x)` both lie in `[-1, 1]`. Consequently, `derivSin` and
`derivCos` share the interval multiplication implemented by `unitFactorDeriv`.
-/
def derivSin {n : Nat} (_xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  unitFactorDeriv dB

/-- Derivative propagation for `cos` using the global enclosure `-sin(x) ∈ [-1, 1]`. -/
def derivCos {n : Nat} (_xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  unitFactorDeriv dB

/--
Derivative bounds for arctangent under the standard ordered-field, absolute-value, and arctangent
laws. The `TanAtan` class supplies executable functions but does not itself prove these laws, so the
assumption is kept in the declaration name.
-/
def derivAtanAssumingStandardLaws {n : Nat} (xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  let interval := fun i : Fin n =>
    let xl := (Tensor.unstack xB.lo i).item
    let xu := (Tensor.unstack xB.hi i).item
    let absMax := Activations.max2 (MathFunctions.abs xl) (MathFunctions.abs xu)
    let absMin := if xl < 0 then if xu > 0 then 0 else MathFunctions.abs xu else xl
    let slopeLo := 1 / (1 + absMax * absMax)
    let slopeHi := 1 / (1 + absMin * absMin)
    scalarIntervalMul slopeLo slopeHi (Tensor.unstack dB.lo i).item (Tensor.unstack dB.hi i).item
  { lo := Tensor.dim fun i => Tensor.scalar (interval i).1
    hi := Tensor.dim fun i => Tensor.scalar (interval i).2 }

/--
Second-derivative propagation for `sin` using global `[-1,1]` enclosures for both `cos` and
`-sin`, together with full interval multiplication.
-/
def secondDerivSin {n : Nat} (_xB : Box α (.dim n .scalar))
    (dB d2B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  -- y'' = f''(x) * (dx)² + f'(x) * d²x, where f'(x) = cos(x) and f''(x) = -sin(x)
  let interval := fun i : Fin n =>
    let dxSq := scalarIntervalSquare (Tensor.unstack dB.lo i).item (Tensor.unstack dB.hi i).item
    let first := unitIntervalMul dxSq.1 dxSq.2
    let second :=
      unitIntervalMul (Tensor.unstack d2B.lo i).item (Tensor.unstack d2B.hi i).item
    (first.1 + second.1, first.2 + second.2)
  { lo := Tensor.dim fun i => Tensor.scalar (interval i).1
    hi := Tensor.dim fun i => Tensor.scalar (interval i).2 }

end NN.MLTheory.CROWN.Operators.Trigonometric
