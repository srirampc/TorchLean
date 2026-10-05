/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main

/-!
# End-to-end IBP soundness (graph dialect, over `ℝ`)

`NN.MLTheory.CROWN.Proofs.GraphCertSoundness` proves:

> If a per-node IBP certificate is locally consistent (`CertLocalOK`) and the value semantics is
> locally consistent (`SemLocalOK`), then each certified box encloses the corresponding value.

This file supplies a concrete, total evaluator and a concrete, total IBP propagation and proves
they satisfy the local-consistency predicates under the `TopoSorted` assumption (parents have
smaller ids). Combining these results yields an end-to-end theorem.

The final section connects the proof-side pass `runIBP?` to the engine's executable `runIBP`
(`runIBP_eq_runIBP?`) and derives the end-to-end theorem for the engine
(`runIBP_encloses_evalGraphRec`).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

namespace CertSoundness

noncomputable section

/-!
## Total evaluators (Nat-recursive, prefix semantics)

We define fold-by-id evaluators using `Nat.rec` rather than `List.foldl` to keep proofs small and
stable. The resulting arrays coincide with the intended “evaluate in node-id order” semantics.
-/

/-- Evaluate the first `n` nodes of `g` in id order, leaving the rest `none`. -/
def evalGraphPrefix (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) :
    Nat → Array (Option Val)
  | 0 => Array.replicate g.nodes.size none
  | n + 1 =>
      let acc := evalGraphPrefix g ps inputs n
      acc.set! n (evalNode? g.nodes ps inputs acc n)

/-- Evaluate all nodes of `g` in id order, returning the final value array. -/
def evalGraphRec (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) :
    Array (Option Val) :=
  evalGraphPrefix g ps inputs g.nodes.size

/-- Prefix evaluator for the safe IBP checker step (`certStepNode?`). -/
def runIBPPrefix (g : Graph) (ps : ParamStore ℝ) : Nat → Array (Option (FlatBox ℝ))
  | 0 => Array.replicate g.nodes.size none
  | n + 1 =>
      let acc := runIBPPrefix g ps n
      acc.set! n (certStepNode? g.nodes ps acc n)

/-- Run the safe IBP checker step across the full graph, producing a per-node certificate array. -/
def runIBP? (g : Graph) (ps : ParamStore ℝ) : Array (Option (FlatBox ℝ)) :=
  runIBPPrefix g ps g.nodes.size

/-! Prefix size facts. -/

private theorem evalGraphPrefix_size (g : Graph) (ps : ParamStore ℝ)
    (inputs : Std.HashMap Nat Val) :
    ∀ n, (evalGraphPrefix g ps inputs n).size = g.nodes.size := by
  intro n; induction n with
  | zero => simp [evalGraphPrefix]
  | succ n ih =>
      simp [evalGraphPrefix, ih, Array.set!_eq_setIfInBounds]

private theorem runIBPPrefix_size (g : Graph) (ps : ParamStore ℝ) :
    ∀ n, (runIBPPrefix g ps n).size = g.nodes.size := by
  intro n; induction n with
  | zero => simp [runIBPPrefix]
  | succ n ih =>
      simp [runIBPPrefix, ih, Array.set!_eq_setIfInBounds]

/-! Prefix stability: the write at step `n` (in or out of bounds) leaves earlier entries alone. -/

private theorem evalGraphPrefix_succ_get_of_lt
    (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    {n i : Nat} (hi : i < n) :
    (evalGraphPrefix g ps inputs (n + 1))[i]! = (evalGraphPrefix g ps inputs n)[i]! := by
  simp only [evalGraphPrefix]
  exact Array.getElem!_set!_ne _ _ _ _ (Nat.ne_of_gt hi)

private theorem runIBPPrefix_succ_get_of_lt
    (g : Graph) (ps : ParamStore ℝ)
    {n i : Nat} (hi : i < n) :
    (runIBPPrefix g ps (n + 1))[i]! = (runIBPPrefix g ps n)[i]! := by
  simp only [runIBPPrefix]
  exact Array.getElem!_set!_ne _ _ _ _ (Nat.ne_of_gt hi)

private theorem evalGraphPrefix_get_of_lt
    (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    {k n i : Nat} (hkn : k ≤ n) (hi : i < k) :
    (evalGraphPrefix g ps inputs n)[i]! = (evalGraphPrefix g ps inputs k)[i]! := by
  induction n generalizing k with
  | zero =>
      have hk : k = 0 := Nat.eq_zero_of_le_zero hkn
      subst hk
      cases hi
  | succ n ih =>
      rcases Nat.lt_or_eq_of_le hkn with hklt | rfl
      · have hkn' : k ≤ n := Nat.le_of_lt_succ hklt
        have hin : i < n := lt_of_lt_of_le hi hkn'
        exact (evalGraphPrefix_succ_get_of_lt (g := g) (ps := ps) (inputs := inputs) hin) ▸
          ih (hkn := hkn') hi
      · rfl

private theorem runIBPPrefix_get_of_lt
    (g : Graph) (ps : ParamStore ℝ)
    {k n i : Nat} (hkn : k ≤ n) (hi : i < k) :
    (runIBPPrefix g ps n)[i]! = (runIBPPrefix g ps k)[i]! := by
  induction n generalizing k with
  | zero =>
      have hk : k = 0 := Nat.eq_zero_of_le_zero hkn
      subst hk
      cases hi
  | succ n ih =>
      rcases Nat.lt_or_eq_of_le hkn with hklt | rfl
      · have hkn' : k ≤ n := Nat.le_of_lt_succ hklt
        have hin : i < n := lt_of_lt_of_le hi hkn'
        exact (runIBPPrefix_succ_get_of_lt (g := g) (ps := ps) (n := n) (i := i) hin) ▸
          ih (hkn := hkn') hi
      · rfl

/-!
## Congruence of the step functions under `TopoSorted`

If two arrays agree on all parent ids of node `id`, then the step function at `id` evaluates to the
same result.
-/

/-- `certStepNode?` reads the certificate only at the node's parents, through `getBox?`. -/
private theorem certStepNode?_congr (nodes : Array Node) (ps : ParamStore ℝ)
    (cert₁ cert₂ : Array (Option (FlatBox ℝ))) (id : Nat)
    (h : ∀ p ∈ (nodes[id]!).parents, getBox? cert₁ p = getBox? cert₂ p) :
    certStepNode? nodes ps cert₁ id = certStepNode? nodes ps cert₂ id := by
  have hu : ∀ {p}, NN.IR.unaryParent? (nodes[id]!).parents = some p →
      getBox? cert₁ p = getBox? cert₂ p :=
    fun hp => h _ (NN.IR.mem_of_unaryParent?_eq_some hp)
  have hb : ∀ {pq : Nat × Nat}, NN.IR.binaryParents? (nodes[id]!).parents = some pq →
      getBox? cert₁ pq.1 = getBox? cert₂ pq.1 ∧ getBox? cert₁ pq.2 = getBox? cert₂ pq.2 :=
    fun hp => ⟨h _ (NN.IR.fst_mem_of_binaryParents?_eq_some hp),
      h _ (NN.IR.snd_mem_of_binaryParents?_eq_some hp)⟩
  cases hk : (nodes[id]!).kind
  case concat axis =>
    simp only [certStepNode?, hk, array_mapM_congr_of_mem (nodes[id]!).parents h]
  all_goals
    cases hp : NN.IR.unaryParent? (nodes[id]!).parents <;>
    cases hq : NN.IR.binaryParents? (nodes[id]!).parents with
    | none => simp_all [certStepNode?]
    | some pq =>
        obtain ⟨h₁, h₂⟩ := hb hq
        simp_all [certStepNode?]

/-- Under topological order, agreement of the raw entries at the (earlier) parents transfers
`evalNode?` from one graph-sized table to another. -/
private theorem evalNode?_congr_of_parents
    (nodes : Array Node) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (vals₁ vals₂ : Array (Option Val))
    {id : Nat} (hid : id < nodes.size)
    (hsize₁ : vals₁.size = nodes.size)
    (hsize₂ : vals₂.size = nodes.size)
    (hparsLt : ∀ p, p ∈ (nodes[id]!).parents → p < id)
    (hpar : ∀ p, p ∈ (nodes[id]!).parents → vals₁[p]! = vals₂[p]!) :
    evalNode? nodes ps inputs vals₁ id = evalNode? nodes ps inputs vals₂ id := by
  refine evalNode?_congr nodes ps inputs vals₁ vals₂ id fun p hp => ?_
  have hpNodes : p < nodes.size := lt_trans (hparsLt p hp) hid
  have hp1 : p < vals₁.size := by simpa [hsize₁] using hpNodes
  have hp2 : p < vals₂.size := by simpa [hsize₂] using hpNodes
  simpa [getVal?, hp1, hp2] using hpar p hp

/-- The certificate analogue of `evalNode?_congr_of_parents`. -/
private theorem certStepNode?_congr_of_parents
    (nodes : Array Node) (ps : ParamStore ℝ)
    (cert₁ cert₂ : Array (Option (FlatBox ℝ)))
    {id : Nat} (hid : id < nodes.size)
    (hsize₁ : cert₁.size = nodes.size)
    (hsize₂ : cert₂.size = nodes.size)
    (hparsLt : ∀ p, p ∈ (nodes[id]!).parents → p < id)
    (hpar : ∀ p, p ∈ (nodes[id]!).parents → cert₁[p]! = cert₂[p]!) :
    certStepNode? nodes ps cert₁ id = certStepNode? nodes ps cert₂ id := by
  refine certStepNode?_congr nodes ps cert₁ cert₂ id fun p hp => ?_
  have hpNodes : p < nodes.size := lt_trans (hparsLt p hp) hid
  have hp1 : p < cert₁.size := by simpa [hsize₁] using hpNodes
  have hp2 : p < cert₂.size := by simpa [hsize₂] using hpNodes
  simpa [getBox?, hp1, hp2] using hpar p hp

/-!
## Local consistency of the total evaluators
-/

/-- Under topological order, the total evaluator `evalGraphRec` satisfies `SemLocalOK`.

No restriction on node kinds is needed: `evalNode?` consults the table only at parents, which
precede the node, so the value written at step `id + 1` from the prefix equals the value
`evalNode?` reads off the final table. -/
theorem evalGraphRec_SemLocalOK (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (htopo : TopoSorted g) :
    SemLocalOK g ps inputs (evalGraphRec g ps inputs) := by
  refine ⟨by simpa [evalGraphRec] using (evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs)
    g.nodes.size), ?_⟩
  intro id hid
  have hidSz : id < (evalGraphPrefix g ps inputs id).size := by
    simpa [evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs)] using hid
  -- Step `id + 1` writes index `id`; later steps leave it unchanged.
  have hstep :
      (evalGraphPrefix g ps inputs (id + 1))[id]!
        = evalNode? g.nodes ps inputs (evalGraphPrefix g ps inputs id) id := by
    simp only [evalGraphPrefix]
    exact Array.getElem!_set!_self _ _ _ hidSz
  have hstable :
      (evalGraphRec g ps inputs)[id]!
        = (evalGraphPrefix g ps inputs (id + 1))[id]! := by
    have := evalGraphPrefix_get_of_lt (g := g) (ps := ps) (inputs := inputs)
      (k := id + 1) (n := g.nodes.size) (i := id)
      (hkn := Nat.succ_le_of_lt hid) (hi := Nat.lt_succ_self id)
    simpa [evalGraphRec] using this
  -- Parents precede `id`, so the prefix and the final table agree on them.
  have hpar :
      ∀ p, p ∈ (g.nodes[id]!).parents →
        (evalGraphPrefix g ps inputs id)[p]!
          = (evalGraphRec g ps inputs)[p]! := by
    intro p hp
    have := evalGraphPrefix_get_of_lt (g := g) (ps := ps) (inputs := inputs)
      (k := id) (n := g.nodes.size) (i := p)
      (hkn := Nat.le_of_lt hid) (hi := htopo id hid p hp)
    simpa [evalGraphRec] using this.symm
  have hsize₁ : (evalGraphPrefix g ps inputs id).size = g.nodes.size :=
    evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs) id
  have hsize₂ : (evalGraphRec g ps inputs).size = g.nodes.size := by
    simpa [evalGraphRec] using evalGraphPrefix_size (g := g) (ps := ps) (inputs := inputs)
      g.nodes.size
  exact (hstable.trans hstep).trans
    (evalNode?_congr_of_parents (nodes := g.nodes) (ps := ps) (inputs := inputs)
      (vals₁ := evalGraphPrefix g ps inputs id) (vals₂ := evalGraphRec g ps inputs)
      (hid := hid) (hsize₁ := hsize₁) (hsize₂ := hsize₂)
      (hparsLt := fun p hp => htopo id hid p hp) hpar)

/-- The safe step at a prefix agrees with the safe step at the full `runIBP?` array. -/
private theorem certStepNode?_runIBPPrefix_eq (g : Graph) (ps : ParamStore ℝ)
    (htopo : TopoSorted g) {id : Nat} (hid : id < g.nodes.size) :
    certStepNode? g.nodes ps (runIBPPrefix g ps id) id
      = certStepNode? g.nodes ps (runIBP? g ps) id := by
  have hpar :
      ∀ p, p ∈ (g.nodes[id]!).parents →
        (runIBPPrefix g ps id)[p]! = (runIBP? g ps)[p]! := by
    intro p hp
    have := runIBPPrefix_get_of_lt (g := g) (ps := ps)
      (k := id) (n := g.nodes.size) (i := p)
      (hkn := Nat.le_of_lt hid) (hi := htopo id hid p hp)
    simpa [runIBP?] using this.symm
  have hsize₁ : (runIBPPrefix g ps id).size = g.nodes.size :=
    runIBPPrefix_size (g := g) (ps := ps) id
  have hsize₂ : (runIBP? g ps).size = g.nodes.size := by
    simpa [runIBP?] using runIBPPrefix_size (g := g) (ps := ps) g.nodes.size
  exact certStepNode?_congr_of_parents (nodes := g.nodes) (ps := ps)
    (cert₁ := runIBPPrefix g ps id) (cert₂ := runIBP? g ps)
    (hid := hid) (hsize₁ := hsize₁) (hsize₂ := hsize₂)
    (hparsLt := fun p hp => htopo id hid p hp) hpar

/-- Under topological order, the certificate produced by `runIBP?` satisfies `CertLocalOK`. -/
theorem runIBP?_CertLocalOK (g : Graph) (ps : ParamStore ℝ)
    (htopo : TopoSorted g) :
    CertLocalOK (g := g) (ps := ps) (runIBP? g ps) := by
  refine ⟨by simpa [runIBP?] using (runIBPPrefix_size (g := g) (ps := ps) g.nodes.size), ?_⟩
  intro id hid
  have hidSz : id < (runIBPPrefix g ps id).size := by
    simpa [runIBPPrefix_size (g := g) (ps := ps)] using hid
  have hstep :
      (runIBPPrefix g ps (id + 1))[id]!
        = certStepNode? g.nodes ps (runIBPPrefix g ps id) id := by
    simp only [runIBPPrefix]
    exact Array.getElem!_set!_self _ _ _ hidSz
  have hstable :
      (runIBP? g ps)[id]!
        = (runIBPPrefix g ps (id + 1))[id]! := by
    have := runIBPPrefix_get_of_lt (g := g) (ps := ps)
      (k := id + 1) (n := g.nodes.size) (i := id)
      (hkn := Nat.succ_le_of_lt hid) (hi := Nat.lt_succ_self id)
    simpa [runIBP?] using this
  exact (hstable.trans hstep).trans (certStepNode?_runIBPPrefix_eq g ps htopo hid)

/-!
## End-to-end theorem
-/

theorem runIBP?_encloses_evalGraphRec
    (g : Graph) (ps : ParamStore ℝ)
    (inputs : Std.HashMap Nat Val)
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hinputs : InputsEnclosed g ps inputs) :
    ∀ id : Nat, id < g.nodes.size →
      match (runIBP? g ps)[id]!, (evalGraphRec g ps inputs)[id]! with
      | some B, some v => EnclosesBox B v
      | _, _ => True := by
  have hcert : CertLocalOK (g := g) (ps := ps) (runIBP? g ps) :=
    runIBP?_CertLocalOK (g := g) (ps := ps) htopo
  have hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) (evalGraphRec g ps inputs) :=
    evalGraphRec_SemLocalOK (g := g) (ps := ps) (inputs := inputs) htopo
  exact cert_encloses_semantics
    (g := g) (ps := ps)
    (cert := runIBP? g ps)
    (inputs := inputs)
    (vals := evalGraphRec g ps inputs)
    (htopo := htopo)
    (hsupp := hsupp)
    (hcert := hcert)
    (hsem := hsem)
    (hinputs := hinputs)


/-!
## The engine pass `runIBP`

`runIBP` (in `NN.MLTheory.CROWN.Graph.Engine.IBP`) is the executable pass that checkers call.
It differs from the proof-side `runIBP?` above in the following ways.

* It folds `propagateIBPNode` over `List.finRange` instead of recursing on a prefix length.
* It returns an all-`none` array when `crownGraphSemanticsSupported` rejects the graph.
* For `tanh`, `sigmoid`, `sin`, and `cos` it uses the `NonlinearBoundOps` enclosures, whereas
  `certStepNode?` uses the `Runtime.Ops.IBP` boxes. For `sin` and `cos` these are different boxes
  (both sound), so the two passes do not agree on those ops.

Both passes propagate missing parent boxes through `Option`. We identify them on the op set where
the per-node steps coincide (`EngineCore`), using the coverage hypothesis (`IBPCovers`) of the
end-to-end enclosure theorem.
-/

/-- Node kinds on which `propagateIBPNode` and `certStepNode?` compute the same box. -/
def EngineCore (g : Graph) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    match (g.nodes[id]!).kind with
    | .input | .const _ | .detach
    | .add | .sub | .mulElem | .relu
    | .linear | .matmul | .softplus | .safeLog | .concat _ | .conv _ => True
    | _ => False

/-- Every node of a certificate array carries a box. -/
def IBPCovers (g : Graph) (cert : Array (Option (FlatBox ℝ))) : Prop :=
  ∀ id : Nat, id < g.nodes.size → ∃ B : FlatBox ℝ, cert[id]! = some B

/-- The engine-core op set is contained in the op set of the IBP soundness theorem. -/
theorem supported_of_engineCore {g : Graph} (h : EngineCore g) : Supported g := by
  intro id hid
  have h' := h id hid
  revert h'
  cases (g.nodes[id]!).kind <;> simp

/-- The certificate accessor is the checked lookup used by the executable dispatcher. -/
private theorem getBox?_eq_join_getElem? (cert : Array (Option (FlatBox ℝ))) (p : Nat) :
    getBox? cert p = (cert[p]?).join :=
  rfl

/-- A successful unary parent decode pins down the parent array. -/
private theorem parents_eq_of_unaryParent?_eq_some {parents : Array Nat} {p : Nat}
    (h : NN.IR.unaryParent? parents = some p) : parents = #[p] := by
  simp only [NN.IR.unaryParent?] at h
  split at h
  next hsize =>
    obtain ⟨a, rfl⟩ := Array.size_eq_one_iff.mp hsize
    simpa using h
  next => simp at h

/-- A successful binary parent decode pins down both parent occurrences. -/
private theorem parents_eq_of_binaryParents?_eq_some {parents : Array Nat} {p q : Nat}
    (h : NN.IR.binaryParents? parents = some (p, q)) : parents = #[p, q] := by
  rcases parents with ⟨entries⟩
  cases entries with
  | nil => simp [NN.IR.binaryParents?] at h
  | cons a entries =>
      cases entries with
      | nil => simp [NN.IR.binaryParents?] at h
      | cons b entries =>
          cases entries with
          | nil => simpa [NN.IR.binaryParents?] using h
          | cons c entries => simp [NN.IR.binaryParents?] at h

/-- On core operators the checked engine writes the box produced by the certificate step. -/
private theorem propagateIBPNode_eq_set!_of_core
    (nodes : Array Node) (ps : ParamStore ℝ) (boxes : Array (Option (FlatBox ℝ))) (id : Nat)
    (hid : id < nodes.size)
    (hparents : ∀ p ∈ (nodes[id]!).parents, p < nodes.size)
    (hcore : match (nodes[id]!).kind with
      | .input | .const _ | .detach
      | .add | .sub | .mulElem | .relu
      | .linear | .matmul | .softplus | .safeLog | .concat _ | .conv _ => True
      | _ => False)
    (B : FlatBox ℝ) (hstep : certStepNode? nodes ps boxes id = some B) :
    propagateIBPNode (α := ℝ) nodes ps boxes id = boxes.set! id (some B) := by
  have hnode : nodes[id]? = some nodes[id]! := by simp [hid]
  rw [propagateIBPNode, hnode]
  change boxes.set! id (ibpStepNodeAt? nodes ps boxes id nodes[id]!) = boxes.set! id (some B)
  congr 1
  cases hk : (nodes[id]!).kind with
  | input => simpa only [ibpStepNodeAt?, certStepNode?, hk] using hstep
  | const shape =>
      cases hv : ps.constVals[id]? <;>
        simpa only [ibpStepNodeAt?, certStepNode?, hk, hv, Bind.bind, Option.bind] using hstep
  | detach | relu | linear | softplus =>
      cases hp : NN.IR.unaryParent? (nodes[id]!).parents with
      | none => simp [certStepNode?, hk, hp] at hstep
      | some p =>
          cases hb : getBox? boxes p with
          | none => simp [certStepNode?, hk, hp, hb] at hstep
          | some box =>
              simpa [ibpStepNodeAt?, certStepNode?, hk, hp,
                ← getBox?_eq_join_getElem?, hb] using hstep
  | add | sub | mulElem | safeLog =>
      cases hp : NN.IR.binaryParents? (nodes[id]!).parents with
      | none => simp [certStepNode?, hk, hp] at hstep
      | some parents =>
          rcases parents with ⟨p, q⟩
          cases hb : getBox? boxes p with
          | none => simp [certStepNode?, hk, hp, hb] at hstep
          | some left =>
              cases hc : getBox? boxes q with
              | none => simp [certStepNode?, hk, hp, hb, hc] at hstep
              | some right =>
                  simpa [ibpStepNodeAt?, certStepNode?, hk, hp,
                    ← getBox?_eq_join_getElem?, hb, hc] using hstep
  | matmul =>
      cases hp : NN.IR.unaryParent? (nodes[id]!).parents with
      | none =>
          cases hq : NN.IR.binaryParents? (nodes[id]!).parents with
          | none => simp [certStepNode?, hk, hp, hq] at hstep
          | some pair =>
              rcases pair with ⟨p, q⟩
              have hpar := parents_eq_of_binaryParents?_eq_some hq
              have hpLt := hparents p (NN.IR.fst_mem_of_binaryParents?_eq_some hq)
              have hqLt := hparents q (NN.IR.snd_mem_of_binaryParents?_eq_some hq)
              cases hb : getBox? boxes p with
              | none => simp [certStepNode?, hk, hp, hq, hb] at hstep
              | some left =>
                  cases hc : getBox? boxes q with
                  | none => simp [certStepNode?, hk, hp, hq, hb, hc] at hstep
                  | some right =>
                      simp only [certStepNode?, hk, hp, hq, hb, hc] at hstep
                      have hrecord :
                          nodes[id]! = { nodes[id]! with kind := .matmul, parents := #[p, q] } := by
                        rw [← hpar, ← hk]
                      rw [hrecord]
                      change (do
                        let left ← nodes[p]?
                        let right ← nodes[q]?
                        ibpBinaryMatmul? left.outShape right.outShape
                          (← (boxes[p]?).join) (← (boxes[q]?).join)) = some B
                      simpa [← getBox?_eq_join_getElem?, hb, hc, hpLt, hqLt] using hstep
      | some p =>
          have hparents := parents_eq_of_unaryParent?_eq_some hp
          cases hb : getBox? boxes p with
          | none => simp [certStepNode?, hk, hp, hb] at hstep
          | some box =>
              simp only [certStepNode?, hk, hp, hb] at hstep
              have hrecord :
                  nodes[id]! = { nodes[id]! with kind := .matmul, parents := #[p] } := by
                rw [← hparents, ← hk]
              rw [hrecord]
              change ((boxes[p]?).join >>= ibpMatmul id ps) = some B
              simpa [← getBox?_eq_join_getElem?, hb] using hstep
  | concat axis =>
      have hget : getBox? boxes = fun p => (boxes[p]?).join :=
        funext (getBox?_eq_join_getElem? boxes)
      simpa only [ibpStepNodeAt?, certStepNode?, hk, concatNodeBoxes?, concatNodeLayout?,
        hget] using hstep
  | conv configuration =>
      have hget : getBox? boxes = fun p => (boxes[p]?).join :=
        funext (getBox?_eq_join_getElem? boxes)
      simpa only [ibpStepNodeAt?, certStepNode?, hk, hget] using hstep
  | _ => simp [hk] at hcore

/-- Prefix of the engine's fold, by recursion on the number of processed nodes. -/
def runIBPEnginePrefix (g : Graph) (ps : ParamStore ℝ) : Nat → Array (Option (FlatBox ℝ))
  | 0 => Array.replicate g.nodes.size none
  | n + 1 => propagateIBPNode (α := ℝ) g.nodes ps (runIBPEnginePrefix g ps n) n

/-- The engine's fold over node ids is the prefix recursion. -/
private theorem foldl_range_propagateIBPNode (g : Graph) (ps : ParamStore ℝ) :
    ∀ n : Nat,
      (List.range n).foldl
          (fun acc i => propagateIBPNode (α := ℝ) g.nodes ps acc i)
          (Array.replicate g.nodes.size none)
        = runIBPEnginePrefix g ps n := by
  intro n
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [List.range_succ, List.foldl_append, ih]
      rfl

/-- `runIBP` is the engine prefix recursion when the semantic guard accepts the graph. -/
theorem runIBP_eq_runIBPEnginePrefix (g : Graph) (ps : ParamStore ℝ)
    (hguard : crownGraphSemanticsSupported (α := ℝ) g ps = true) :
    runIBP (α := ℝ) g ps = runIBPEnginePrefix g ps g.nodes.size := by
  simp only [runIBP, hguard, ite_true]
  -- The fold in `runIBP` runs over the `Fin`-to-`Nat` coercion of `List.finRange`.
  have h : (do
      let a ← List.finRange g.nodes.size
      pure (a : Nat)) = List.range g.nodes.size := by
    simp only [bind, pure, ← List.map_eq_flatMap]
    exact List.map_coe_finRange_eq_range
  rw [h]
  exact foldl_range_propagateIBPNode g ps g.nodes.size

/-- `runIBP` is all `none` when the semantic guard rejects the graph. -/
theorem runIBP_eq_replicate_none (g : Graph) (ps : ParamStore ℝ)
    (hguard : crownGraphSemanticsSupported (α := ℝ) g ps = false) :
    runIBP (α := ℝ) g ps = Array.replicate g.nodes.size none := by
  simp [runIBP, hguard]

/-- Under coverage, the safe step succeeds at every prefix position. -/
private theorem certStepNode?_runIBPPrefix_some (g : Graph) (ps : ParamStore ℝ)
    (htopo : TopoSorted g) (hcov : IBPCovers g (runIBP? g ps))
    {id : Nat} (hid : id < g.nodes.size) :
    ∃ B : FlatBox ℝ, certStepNode? g.nodes ps (runIBPPrefix g ps id) id = some B := by
  obtain ⟨B, hB⟩ := hcov id hid
  refine ⟨B, ?_⟩
  rw [certStepNode?_runIBPPrefix_eq g ps htopo hid, ← hB]
  exact ((runIBP?_CertLocalOK g ps htopo).2 id hid).symm

/-- The engine prefix and the proof-side prefix coincide on engine-core graphs with coverage. -/
private theorem runIBPEnginePrefix_eq_runIBPPrefix (g : Graph) (ps : ParamStore ℝ)
    (htopo : TopoSorted g) (hcore : EngineCore g) (hcov : IBPCovers g (runIBP? g ps)) :
    ∀ n : Nat, n ≤ g.nodes.size → runIBPEnginePrefix g ps n = runIBPPrefix g ps n := by
  intro n
  induction n with
  | zero => intro _; rfl
  | succ n ih =>
      intro hn
      have hnLt : n < g.nodes.size := hn
      have hprev := ih (Nat.le_of_lt hnLt)
      obtain ⟨B, hB⟩ := certStepNode?_runIBPPrefix_some g ps htopo hcov hnLt
      have hstep := propagateIBPNode_eq_set!_of_core g.nodes ps (runIBPPrefix g ps n) n
        hnLt (fun p hp => lt_trans (htopo n hnLt p hp) hnLt) (hcore n hnLt) B hB
      show propagateIBPNode (α := ℝ) g.nodes ps (runIBPEnginePrefix g ps n) n
        = (runIBPPrefix g ps n).set! n (certStepNode? g.nodes ps (runIBPPrefix g ps n) n)
      rw [hprev, hstep, hB]

/--
The engine's `runIBP` computes exactly the proof-side `runIBP?` on engine-core graphs, provided the
semantic guard accepts the graph and the proof-side pass produced a box at every node.

Coverage supplies a successful certificate step at each prefix of the induction.
-/
theorem runIBP_eq_runIBP? (g : Graph) (ps : ParamStore ℝ)
    (htopo : TopoSorted g) (hcore : EngineCore g)
    (hguard : crownGraphSemanticsSupported (α := ℝ) g ps = true)
    (hcov : IBPCovers g (runIBP? g ps)) :
    runIBP (α := ℝ) g ps = runIBP? g ps := by
  rw [runIBP_eq_runIBPEnginePrefix g ps hguard]
  exact runIBPEnginePrefix_eq_runIBPPrefix g ps htopo hcore hcov g.nodes.size le_rfl

/--
End-to-end soundness of the engine's IBP pass: every box computed by `runIBP` encloses the value
computed by the total evaluator `evalGraphRec` at the same node.

The statement quantifies over node ids at which the engine produced a box `B` and the evaluator a
value `v`, so it cannot hold through a missing entry. The semantic guard needs no hypothesis: if
`crownGraphSemanticsSupported` rejects the graph, `runIBP` produces no boxes and there is nothing
to prove. Coverage of the proof-side pass is needed for the reason given at `runIBP_eq_runIBP?`.
-/
theorem runIBP_encloses_evalGraphRec
    (g : Graph) (ps : ParamStore ℝ)
    (inputs : Std.HashMap Nat Val)
    (htopo : TopoSorted g)
    (hcore : EngineCore g)
    (hcov : IBPCovers g (runIBP? g ps))
    (hinputs : InputsEnclosed g ps inputs) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (B : FlatBox ℝ) (v : Val),
        (runIBP (α := ℝ) g ps)[id]! = some B →
        (evalGraphRec g ps inputs)[id]! = some v →
        EnclosesBox B v := by
  intro id hid B v hB hv
  cases hguard : crownGraphSemanticsSupported (α := ℝ) g ps with
  | false =>
      rw [runIBP_eq_replicate_none g ps hguard] at hB
      simp [hid] at hB
  | true =>
      rw [runIBP_eq_runIBP? g ps htopo hcore hguard hcov] at hB
      have h := runIBP?_encloses_evalGraphRec g ps inputs htopo (supported_of_engineCore hcore)
        hinputs id hid
      simpa [hB, hv] using h

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
