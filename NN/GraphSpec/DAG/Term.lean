/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Primitives.Core
public import NN.GraphSpec.DAG.Semantics

/-!
# DAG Term Combinators

Reusable typed term constructors whose semantics are stated independently of graph syntax.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open Spec TorchLean
open TorchLean.Tensor

namespace Term


/-- Add a list of same-shaped terms, starting from the all-zero tensor. -/
def sum {Γ : List Shape} (s : Shape) (terms : List (Term Γ s)) : Term Γ s :=
  terms.foldl
    (fun total term => Term.op (PrimOp.add s) (.cons total (.cons term .nil)))
    (Term.op (PrimOp.zero s) .nil)

/-- Pure evaluation commutes with `Term.sum`. -/
theorem eval_sum {Γ : List Shape} {s : Shape} {α : Type} [TorchLean.Storage α] [Context α]
    (env : TorchLean.TensorPack α Γ) (terms : List (Term Γ s)) :
    Term.eval env (sum s terms) =
      terms.foldl (fun total term => TorchLean.Tensor.addSpec total (Term.eval env term))
        (Tensor.full s 0) := by
  unfold sum
  exact (List.foldl_hom (Term.eval env)
    (g₁ := fun total term => Term.op (PrimOp.add s) (.cons total (.cons term .nil)))
    (g₂ := fun total term => TorchLean.Tensor.addSpec total (Term.eval env term))
    (l := terms) (init := Term.op (PrimOp.zero s) .nil) (fun _ _ => rfl)).symm



end Term

end DAG
end GraphSpec
end NN
