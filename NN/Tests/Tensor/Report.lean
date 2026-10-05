/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

meta import NN.Tactic.Einops.Report

/-!
# Shape-check report status

Raw checker applications can fail or remain symbolic. Their reports must not
claim discharged obligations merely because the application was recognized.
-/

open Lean Meta Elab Command
open TorchLean.Tensor.Internal

run_elab
  let pattern ← Elab.Term.elabTerm
    (← `((⟨[], ⟨0, 0⟩⟩ : Syntax.Expression))) none
  let successful ← mkAppM ``Check.checkParseShape #[pattern, toExpr ([] : List Nat)]
  let report ← Report.renderApplication successful
  unless report.startsWith "parse_shape\n" &&
      (report.splitOn "Discharged obligations").length = 2 do
    throwError "successful shape check lost its certified report: {report}"
  let rejected ← mkAppM ``Check.checkParseShape #[pattern, toExpr [2]]
  let report ← Report.renderApplication rejected
  unless report.startsWith "parse_shape (rejected metadata check)\n" &&
      (report.splitOn "Discharged obligations").length = 1 do
    throwError "rejected shape check was reported as certified: {report}"
  withLocalDeclD `shape (mkApp (mkConst ``List [Level.zero]) (mkConst ``Nat)) fun shape => do
    let symbolic ← mkAppM ``Check.checkParseShape #[pattern, shape]
    let report ← Report.renderApplication symbolic
    unless report.startsWith "parse_shape (unverified metadata check)\n" &&
        (report.splitOn "Discharged obligations").length = 1 do
      throwError "unresolved shape check was reported as certified: {report}"
  let report ← Report.renderApplication (toExpr (0 : Nat))
  unless report = "Unsupported einops application" do
    throwError "unrecognized application was treated as a shape check: {report}"
  let named ← mkAppM ``Check.EinsumAxis.named #[toExpr "batch"]
  let ellipsis ← mkAppM ``Check.EinsumAxis.ellipsis #[toExpr (3 : Nat)]
  for (expression, expected) in [(named, "batch"), (ellipsis, "ellipsis[3]")] do
    let some (normalized, description) ← Report.Impl.decodeEinsumAxis expression
      | throwError "concrete einsum axis did not decode"
    unless normalized == expression && description == expected do
      throwError "einsum axis description or normalized expression changed"
  for expression in [toExpr (0 : Nat), mkConst ``Check.EinsumAxis.named,
      mkApp named (toExpr "extra")] do
    unless (← Report.Impl.decodeEinsumAxis expression).isNone do
      throwError "malformed einsum axis decoded successfully"
  withLocalDeclD `name (mkConst ``String) fun name => do
    let symbolic ← mkAppM ``Check.EinsumAxis.named #[name]
    unless (← Report.Impl.decodeEinsumAxis symbolic).isNone do
      throwError "symbolic einsum axis decoded as a concrete name"
  unless Report.Impl.numberedStageLines 7 ["read", "write"] =
      ["    7. read", "    8. write"] &&
      (Report.Impl.numberedStageLines 7 []).isEmpty do
    throwError "lowering stage numbering changed"
