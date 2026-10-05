/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum.Contraction.Loop -- shake: keep

/-!
# Generated kernel utilities

This module constructs direct correctness certificates for compile-time-expanded
finite folds. Shared generated-let construction lives in `Einsum.Index`.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Unroll a finite fold of known length and construct its equality certificate
from the standard successor and zero laws.

Building this proof directly avoids asking the simplifier to normalize the
dependent operand family generated for every einsum input.
-/
def unrollFiniteFold
    (length : Nat) (step initial : Expr) : MetaM (Expr × Expr) := do
  let fold ←
    mkAppM ``Fin.foldl #[mkNatLit length, step, initial]
  match length with
  | 0 =>
      let hFold ← mkAppM ``Fin.foldl_zero #[step, initial]
      let hUnrolled ← mkAppM ``Eq.symm #[hFold]
      let hUnrolled ←
        withTransparency .all <|
          mkExpectedTypeHint hUnrolled (← mkEq initial fold)
      return (initial, hUnrolled)
  | remainingLength + 1 =>
      let coordinateType ←
        mkAppM ``Fin #[mkNatLit (remainingLength + 1)]
      let firstCoordinate ← mkNumeral coordinateType 0
      let nextInitial := step.beta #[initial, firstCoordinate]
      let stateType ← inferType initial
      let remainingCoordinateType ←
        mkAppM ``Fin #[mkNatLit remainingLength]
      let remainingStep ←
        withLocalDeclD `state stateType fun state =>
          withLocalDeclD `operand remainingCoordinateType fun operand => do
            let successor ← mkAppM ``Fin.succ #[operand]
            mkLambdaFVars #[state, operand] <|
              step.beta #[state, successor]
      let (unrolled, hUnrolledRemaining) ←
        unrollFiniteFold remainingLength remainingStep nextInitial
      let remainingFold ←
        mkAppM ``Fin.foldl #[
          mkNatLit remainingLength, remainingStep, nextInitial]
      let hUnrolledRemaining ←
        withTransparency .all <|
          mkExpectedTypeHint hUnrolledRemaining
            (← mkEq unrolled remainingFold)
      let hFoldRemaining ←
        mkAppM ``Fin.foldl_succ #[step, initial]
      let hFoldRemaining ←
        withTransparency .all <|
          mkExpectedTypeHint hFoldRemaining
            (← mkEq fold remainingFold)
      let hRemainingFold ← mkAppM ``Eq.symm #[hFoldRemaining]
      let hUnrolled ←
        mkAppM ``Eq.trans #[hUnrolledRemaining, hRemainingFold]
      return (unrolled, hUnrolled)

end TorchLean.Tensor.Internal.Elab.Impl
