/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Tensor.Internal.Elab.Einsum.Kernel.Affine
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Affine
public meta import Lean.Elab.Tactic

/-!
# Mixed-radix index certificates

Each example checks the optimizer's result and submits its generated equality proof to the kernel.
The swapped sum cases require an explicit commutativity certificate; definitional equality alone
cannot change the summand order.
-/

namespace NN.Tests.Tensor.AffineIndex

open Lean Meta Elab Tactic
open TorchLean.Tensor.Internal.Elab.Impl

public meta section

elab "certify_affine_index" : tactic =>
  liftMetaTactic fun goal => goal.withContext do
    let target ← instantiateMVars (← goal.getType)
    unless target.isAppOfArity ``Eq 3 do
      throwError "expected an equality"
    let args := target.getAppArgs
    let (result, certificate) ← simplifyAffineIndex args[1]!
    unless ← isDefEq result args[2]! do
      throwError "unexpected optimized index: {result}"
    goal.assign certificate
    return []

end

example (digit : Nat) (remainder : Fin 3) :
    (remainder.val + 3 * digit) / 3 = digit := by
  certify_affine_index

example (digit : Nat) (remainder : Fin 3) :
    (3 * digit + remainder.val) / 3 = digit := by
  certify_affine_index

example (digit : Nat) (remainder : Fin 3) :
    (remainder.val + 3 * digit) % 3 = remainder.val := by
  certify_affine_index

example (digit : Nat) (remainder : Fin 3) :
    (3 * digit + remainder.val) % 3 = remainder.val := by
  certify_affine_index

example (radix digit : Nat) (remainder : Fin radix) :
    (radix * digit + remainder.val) / radix = digit := by
  certify_affine_index

example (radix digit : Nat) (remainder : Fin radix) :
    (radix * digit + remainder.val) % radix = remainder.val := by
  certify_affine_index

example (digit value : Nat) :
    (3 * digit + value % 3) / 3 = digit := by
  certify_affine_index

example (digit value : Nat) :
    (3 * digit + value % 3) % 3 = value % 3 := by
  certify_affine_index

end NN.Tests.Tensor.AffineIndex
