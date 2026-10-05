/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Syntax
public import NN.GraphSpec.DAG.Syntax

/-!
# Structural conversion of sequential GraphSpec chains to DAG terms

This module embeds each sequential primitive as a DAG primitive and lowers composition to
explicit SSA-style let bindings.
-/

@[expose] public section

namespace NN
namespace GraphSpec

open Spec TorchLean
open TorchLean.Tensor

/-! ## Lowering: sequential chain → DAG term -/

/-!
GraphSpec has two surface syntaxes:

- `NN.GraphSpec.Chain`: a *sequential* DSL (`Chain` + `>>>`), ideal for pure pipelines.
- `NN.GraphSpec.DAG`: a *general* SSA/A-normal-form term language, ideal for sharing/skip
  connections.

The DAG term language is GraphSpec’s “general graph” core: it is the representation that can
express sharing and skip connections.

`Chain` exists because it is the clearest way to write pipelines, and it has its own
direct Spec semantics (`Interp.spec`) and program translation (`Chain.toProgram`).

This lowering is still useful whenever you want to *embed* a sequential pipeline into the DAG world
(e.g. to reuse DAG-only tooling, or to keep a single GraphSpec example surface that can export DAG
models).

The declarations below provide a structural lowering:

- `Chain.toDAGTerm` produces a `DAG.Term (ps ++ [σ]) τ`, i.e. a DAG term whose environment starts
  with the parameter list `ps` and ends with the (single) data input `σ`.
Notes:
- The lowering is *purely structural*: it introduces `let1` binders between stages to make the
  sequential flow explicit in SSA form.
- Each sequential `Primitive ps σ τ` is embedded as a DAG primitive op with inputs `ps ++ [σ]`.
  This embedding is generic: any custom GraphSpec primitive automatically becomes usable in the
  DAG world.
-/

namespace LowerToDAG

/-!
### Lowering internals

The definitions below (`argsOfFn`, `toTerm`, …) implement the structural lowering. The
principal entry point is `Chain.toDAGTerm`.
-/

/-! ### Primitive embedding: `Primitive` → `DAG.PrimOp` -/

/--
Embed a sequential GraphSpec primitive as a DAG primitive op.

The resulting op has input shapes `ps ++ [σ]` (parameters followed by the data input).
 -/
def Primitive.toDAGPrimOp {ps : List Shape} {σ τ : Shape} (p : Primitive ps σ τ) :
    DAG.PrimOp (ps ++ [σ]) τ :=
  { name := p.name
    specFwd := fun {α} _storage _ctx xs =>
      let (params, xs') :=
        TorchLean.TensorPack.split
          (α := α) (ss₁ := ps) (ss₂ := [σ]) xs
      match xs' with
      | .cons x .nil => p.specFwd (α := α) params x
    program := fun {α} _storage _ctx => p.program (α := α)
  }

/-! ### Building well-typed DAG arguments for a primitive call -/

/--
Build a typed `DAG.Args` list from an index-based family of argument terms.

This is the bridge from “arguments as a function of `Fin ins.length`” to the inductive `DAG.Args`
encoding used by `DAG.Term.op`.
-/
def argsOfFn {Γ : List Shape} :
    (ins : List Shape) →
    (∀ i : Fin ins.length, DAG.Term Γ (ins.get i)) →
    DAG.Args Γ ins
  | .nil, _f => .nil
  | .cons s ss, f =>
      -- Head: index 0.
      let head : DAG.Term Γ ((s :: ss).get ⟨0, by simp⟩) := f ⟨0, by simp⟩
      -- Tail: shift indices by 1, and cast the `List.get` result to match `ss.get i`.
      let tail : DAG.Args Γ ss :=
        argsOfFn ss (fun i =>
          DAG.Term.cast (f ⟨i.1 + 1, Nat.succ_lt_succ i.2⟩) List.get_cons_succ')
      .cons (by simpa using head) tail

/--
Reference the `i`th parameter block inside a larger environment layout.

The surrounding environment is split as `pre ++ ps ++ post ++ extra`; this helper returns the term
that points at parameter `i : Fin ps.length` while keeping the full ambient environment explicit.
-/
def mkParamTerm
    {pre ps post extra : List Shape}
    (i : Fin ps.length) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) (ps.get i) := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  let idx : Fin Γ.length := ⟨pre.length + i.val, by
    have := i.isLt
    simp only [Γ, List.length_append]
    omega⟩
  have hGet : Γ.get idx = ps.get i := by
    simp [Γ, idx, List.get_eq_getElem, i.isLt]
  exact DAG.Term.cast (DAG.Term.var (DAG.Var.ofFin idx)) hGet

/--
Lower a unary `Primitive` application into the DAG term language.

Parameters are read from the middle `ps` segment of the ambient environment, in the same order as
the primitive's parameter ABI, and the final data input is supplied by `x`.
-/
def primCall
    {pre ps post extra : List Shape} {σ τ : Shape}
    (p : Primitive ps σ τ)
    (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) τ := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  let op : DAG.PrimOp (ps ++ [σ]) τ := Primitive.toDAGPrimOp (ps := ps) (σ := σ) (τ := τ) p
  let paramsArgs : DAG.Args Γ ps :=
    argsOfFn (Γ := Γ) ps (fun i => mkParamTerm (pre := pre) (ps := ps) (post := post)
      (extra := extra) i)
  let args : DAG.Args Γ (ps ++ [σ]) :=
    DAG.Args.append paramsArgs (.cons x .nil)
  exact (by
    -- Discharge the local `Γ` abbreviation.
    simpa [Γ] using (DAG.Term.op (Γ := Γ) op args))

/-! ### Chain lowering -/

/-- Lower a sequential `Chain` to an SSA-style `DAG.Term`, with parameters read from the
  environment. -/
def toTerm
    {pre ps post extra : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ)
    (x : DAG.Term ((pre ++ ps ++ post) ++ extra) σ) :
    DAG.Term ((pre ++ ps ++ post) ++ extra) τ := by
  let Γ : List Shape := (pre ++ ps ++ post) ++ extra
  match g with
  | .id _ =>
      simpa [Γ] using x
  | .prim p =>
      simpa [Γ] using primCall (pre := pre) (ps := ps) (post := post) (extra := extra) (p := p) x
  | .seq (ps₁ := ps₁) (ps₂ := ps₂) (σ := σ) (τ := τm) (υ := τ) g₁ g₂ =>
      -- Outer env: `(pre ++ (ps₁ ++ ps₂) ++ post) ++ extra`
      let Γ0 : List Shape := (pre ++ (ps₁ ++ ps₂) ++ post) ++ extra
      -- Left subgraph sees post = ps₂ ++ post.
      have hΓ1 :
          (pre ++ ps₁ ++ (ps₂ ++ post)) ++ extra
          =
          Γ0 := by
        simp [Γ0, List.append_assoc]
      let t₁ : DAG.Term Γ0 τm :=
        DAG.Term.castEnv
          (toTerm (pre := pre) (ps := ps₁) (post := ps₂ ++ post) (extra := extra) g₁
            (by
              have x0 : DAG.Term Γ0 σ := by
                simpa [Γ, Γ0] using x
              exact
                DAG.Term.castEnv x0 (by simp [Γ0, List.append_assoc])))
          hΓ1
      -- `let1`-bind and translate the right subgraph.
      let bodyEnv : List Shape := Γ0 ++ [τm]
      -- Bound var in the body env (the last element, at index `Γ0.length`).
      let boundIdx : Fin bodyEnv.length := ⟨Γ0.length, by simp [bodyEnv, List.length_append]⟩
      let boundVar : DAG.Term bodyEnv τm :=
        have hGet : bodyEnv.get boundIdx = τm := by
          simp [bodyEnv, boundIdx]
        DAG.Term.cast (DAG.Term.var (Γ := bodyEnv) (DAG.Var.ofFin boundIdx)) hGet
      -- Translate `g₂` under its own parenthesization, then cast back to `bodyEnv`.
      let rhsEnv : List Shape := ((pre ++ ps₁) ++ ps₂ ++ post) ++ (extra ++ [τm])
      have hRhs : rhsEnv = bodyEnv := by
        simp [rhsEnv, bodyEnv, Γ0, List.append_assoc]
      let boundVar' : DAG.Term rhsEnv τm :=
        DAG.Term.castEnv boundVar hRhs.symm
      let t₂' : DAG.Term rhsEnv τ :=
        toTerm (pre := pre ++ ps₁) (ps := ps₂) (post := post) (extra := extra ++ [τm]) g₂ boundVar'
      let t₂ : DAG.Term bodyEnv τ :=
        DAG.Term.castEnv t₂' hRhs
      let out : DAG.Term Γ0 τ := DAG.Term.let1 t₁ t₂
      -- Discharge the local `Γ` abbreviation.
      simpa [Γ, Γ0] using out

/-! ### Public API -/

/--
Lower a sequential `Chain` to a DAG term with environment `ps ++ [σ]`.
 -/
def Chain.toDAGTerm {ps : List Shape} {σ τ : Shape} (g : Chain ps σ τ) :
    DAG.Term (ps ++ [σ]) τ :=
  let x :
      let Γ : List Shape := ([] ++ ps ++ []) ++ [σ]
      DAG.Term Γ σ := by
    intro Γ
    have hLt : ps.length < Γ.length := by
      simp [Γ, List.length_append]
    let xIdx : Fin Γ.length := ⟨ps.length, hLt⟩
    have hGet0 :
        Γ.get ⟨ps.length, by simp [Γ, List.length_append]⟩ = σ := by
      -- `Γ` is definitional `(([] ++ ps ++ []) ++ [σ])`, so this is the last element.
      simp [Γ]
    have hxIdx : xIdx = ⟨ps.length, by simp [Γ, List.length_append]⟩ := by
      apply Fin.ext
      rfl
    have hGet : Γ.get xIdx = σ := by
      simpa [hxIdx] using hGet0
    exact DAG.Term.cast (DAG.Term.var (Γ := Γ) (DAG.Var.ofFin xIdx)) hGet
  -- `toTerm`’s environment is definitional `([] ++ ps ++ [] ++ [σ])`; normalize to `ps ++ [σ]`.
  by
    simpa [List.nil_append, List.append_nil, List.append_assoc] using
      (toTerm (pre := []) (ps := ps) (post := []) (extra := [σ]) g x)

end LowerToDAG

end GraphSpec
end NN
