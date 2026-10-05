/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormGraph
public import NN.Proofs.Autograd.Tape.Ops.Norm.CtxVecEval
public import NN.Proofs.Autograd.Tape.Ops.Norm.RowNormalization

/-!
# Evaluating the LayerNorm graph

Closed forms for every saved intermediate of `layerNormGraph`. Each `val*` definition is the value
the corresponding node computes from the packed input `[X, gamma, beta]`, and each `get_*` lemma
identifies the block of `Graph.evalVec` at that stage with its closed form.

Two consequences drive the calculus: the per-row variance block is a mean of squares, so it is
nonnegative, and therefore `var + ε` is positive and the clamped square root is nonzero as soon as
`0 < ε`. Those two facts are exactly the domain hypotheses of the LayerNorm graph theorem.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean

open scoped BigOperators

noncomputable section

namespace LayerNorm

open TapeNodes TapeNodes.MatrixLinear

variable {m n : Nat}

/-- Index of `X` in the bare LayerNorm context. -/
def idxInputX : Idx (ΓLN m n) (MatShape m n) := ⟨⟨0, by simp [ΓLN]⟩, by simp [ΓLN]⟩

/-- Index of `gamma` in the bare LayerNorm context. -/
def idxInputGamma : Idx (ΓLN m n) (VecShape n) := ⟨⟨1, by simp [ΓLN]⟩, by simp [ΓLN]⟩

/-- Index of `beta` in the bare LayerNorm context. -/
def idxInputBeta : Idx (ΓLN m n) (VecShape n) := ⟨⟨2, by simp [ΓLN]⟩, by simp [ΓLN]⟩

/-! ## Closed forms of the intermediates -/

/-- The input matrix block. -/
def valX (xV : CtxVec (ΓLN m n)) : Vec (Spec.Shape.size (MatShape m n)) :=
  CtxVec.get (Γ := ΓLN m n) (s := MatShape m n) idxInputX xV

/-- The scale vector block. -/
def valGamma (xV : CtxVec (ΓLN m n)) : Vec n :=
  getVec (Γ := ΓLN m n) (n := n) idxInputGamma xV

/-- The shift vector block. -/
def valBeta (xV : CtxVec (ΓLN m n)) : Vec n :=
  getVec (Γ := ΓLN m n) (n := n) idxInputBeta xV

/-- Row means of `X`. -/
def valMean (xV : CtxVec (ΓLN m n)) : Vec (Spec.Shape.size (VecShape m)) :=
  castVec (Matmul.vecSize_eq m).symm (rowMeanCLM (m := m) (n := n) (valX xV))

/-- Row means broadcast back to the matrix shape. -/
def valMeanB (xV : CtxVec (ΓLN m n)) : Vec (Spec.Shape.size (MatShape m n)) :=
  broadcastRowCLM (m := m) (n := n) (castVec (Matmul.vecSize_eq m) (valMean xV))

/-- Centered input `X - mean_b`. -/
def valCentered (xV : CtxVec (ΓLN m n)) : Vec (Spec.Shape.size (MatShape m n)) :=
  valX xV - valMeanB xV

/-- Squared centered input. -/
def valCenteredSq (xV : CtxVec (ΓLN m n)) : Vec (Spec.Shape.size (MatShape m n)) :=
  vecOfFun (n := Spec.Shape.size (MatShape m n)) fun i => valCentered xV i * valCentered xV i

/-- Row variances (means of the squared centered entries). -/
def valVar (xV : CtxVec (ΓLN m n)) : Vec (Spec.Shape.size (VecShape m)) :=
  castVec (Matmul.vecSize_eq m).symm (rowMeanCLM (m := m) (n := n) (valCenteredSq xV))

/-- Row variances shifted by `ε`. -/
def valVarEps (xV : CtxVec (ΓLN m n)) (ε : ℝ) : Vec (Spec.Shape.size (VecShape m)) :=
  vecOfFun (n := Spec.Shape.size (VecShape m)) fun i => valVar xV i + ε

/-- Clamped standard deviation `sqrt (max (var + ε) 0)`. -/
def valStd (xV : CtxVec (ΓLN m n)) (ε : ℝ) : Vec (Spec.Shape.size (VecShape m)) :=
  vecOfFun (n := Spec.Shape.size (VecShape m)) fun i => Real.sqrt (max (valVarEps xV ε i) 0)

/-- Inverse standard deviation. -/
def valInvStd (xV : CtxVec (ΓLN m n)) (ε : ℝ) : Vec (Spec.Shape.size (VecShape m)) :=
  vecOfFun (n := Spec.Shape.size (VecShape m)) fun i => (valStd xV ε i)⁻¹

/-- Inverse standard deviation broadcast back to the matrix shape. -/
def valInvStdB (xV : CtxVec (ΓLN m n)) (ε : ℝ) : Vec (Spec.Shape.size (MatShape m n)) :=
  broadcastRowCLM (m := m) (n := n) (castVec (Matmul.vecSize_eq m) (valInvStd xV ε))

/-- Normalized input `centered ⊙ inv_std_b`. -/
def valNorm (xV : CtxVec (ΓLN m n)) (ε : ℝ) : Vec (Spec.Shape.size (MatShape m n)) :=
  vecOfFun (n := Spec.Shape.size (MatShape m n)) fun i => valCentered xV i * valInvStdB xV ε i

/-- Scale vector broadcast over rows. -/
def valGammaB (xV : CtxVec (ΓLN m n)) : Vec (Spec.Shape.size (MatShape m n)) :=
  broadcastColCLM (m := m) (n := n) (valGamma xV)

/-- Scaled normalized input. -/
def valScaled (xV : CtxVec (ΓLN m n)) (ε : ℝ) : Vec (Spec.Shape.size (MatShape m n)) :=
  vecOfFun (n := Spec.Shape.size (MatShape m n)) fun i => valNorm xV ε i * valGammaB xV i

/-- Shift vector broadcast over rows. -/
def valBetaB (xV : CtxVec (ΓLN m n)) : Vec (Spec.Shape.size (MatShape m n)) :=
  broadcastColCLM (m := m) (n := n) (valBeta xV)

/-- LayerNorm output `scaled + beta_b`. -/
def valY (xV : CtxVec (ΓLN m n)) (ε : ℝ) : Vec (Spec.Shape.size (MatShape m n)) :=
  valScaled xV ε + valBetaB xV

/-! ## Stage evaluations -/

/-- Stage `graphMean`: the mean block. -/
theorem get_graphMean (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ [VecShape m]) (s := VecShape m) (idxMean (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := [VecShape m]) (graphMean (m := m) (n := n)) xV) =
      valMean xV := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n) (ss := []) (τ := VecShape m)
    Graph.nil (nodeMean (m := m) (n := n)) xV (idxMean (m := m) (n := n)) rfl).trans ?_
  simp only [nodeMean, rowMean, Node.forwardVec_ofFn]
  rw [Graph.get_evalVec_input (Γ := ΓLN m n) (ss := []) Graph.nil xV
    (idxX (m := m) (n := n) (ss := [])) (idxInputX (m := m) (n := n)) rfl]
  rfl

/-- Stage `graphMeanBroadcast`: the broadcast mean block. -/
theorem get_graphMeanBroadcast (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ [VecShape m, MatShape m n]) (s := MatShape m n)
      (idxMeanB (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n])
        (graphMeanBroadcast (m := m) (n := n)) xV) =
      valMeanB xV := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n) (ss := [VecShape m]) (τ := MatShape m n)
    (graphMean (m := m) (n := n)) (nodeMeanB (m := m) (n := n)) xV (idxMeanB (m := m) (n := n))
    rfl).trans ?_
  simp only [nodeMeanB, broadcastRow, Node.forwardVec_ofFn, getVec]
  rw [get_graphMean]
  rfl

/-- Stage `graphCentered`: the centered block. -/
theorem get_graphCentered (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n]) (s := MatShape m n)
      (idxCentered (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n, MatShape m n])
        (graphCentered (m := m) (n := n)) xV) =
      valCentered xV := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n])
    (τ := MatShape m n) (graphMeanBroadcast (m := m) (n := n)) (nodeCentered (m := m) (n := n)) xV
    (idxCentered (m := m) (n := n)) rfl).trans ?_
  simp only [nodeCentered, sub, Node.forwardVec_ofFn]
  rw [Graph.get_evalVec_input (Γ := ΓLN m n) (ss := [VecShape m, MatShape m n])
    (graphMeanBroadcast (m := m) (n := n)) xV
    (idxX (m := m) (n := n) (ss := [VecShape m, MatShape m n]))
    (idxInputX (m := m) (n := n)) rfl, get_graphMeanBroadcast]
  rfl

/-- Stage `graphCenteredSq`: the squared centered block. -/
theorem get_graphCenteredSq (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n])
      (s := MatShape m n) (idxCenteredSq (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n])
        (graphCenteredSq (m := m) (n := n)) xV) =
      valCenteredSq xV := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n)
    (ss := [VecShape m, MatShape m n, MatShape m n]) (τ := MatShape m n)
    (graphCentered (m := m) (n := n))
    (nodeCenteredSq (m := m) (n := n)) xV (idxCenteredSq (m := m) (n := n)) rfl).trans ?_
  simp only [nodeCenteredSq, mul, Node.forwardVec_ofFn]
  rw [get_graphCentered]
  rfl

/-- Stage `graphVar`: the variance block. -/
theorem get_graphVar (xV : CtxVec (ΓLN m n)) :
    CtxVec.get
      (Γ := ΓLN m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m])
      (s := VecShape m) (idxVar (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m])
        (graphVar (m := m) (n := n)) xV) =
      valVar xV := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n)
    (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n]) (τ := VecShape m)
    (graphCenteredSq (m := m) (n := n)) (nodeVar (m := m) (n := n)) xV (idxVar (m := m) (n := n))
      rfl).trans ?_
  simp only [nodeVar, rowMean, Node.forwardVec_ofFn]
  rw [get_graphCenteredSq]
  rfl

/-- Stage `graphVarEps`: the shifted variance block. -/
theorem get_graphVarEps (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ ssVarEps m n) (s := VecShape m) (idxVarEps (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := ssVarEps m n)
        (graphVarEps (m := m) (n := n) ε) xV) =
      valVarEps xV ε := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n)
    (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m]) (τ := VecShape m)
    (graphVar (m := m) (n := n)) (nodeVarEps (m := m) (n := n) ε) xV (idxVarEps (m := m) (n := n))
    rfl).trans ?_
  simp only [nodeVarEps, elemwise, Node.forwardVec_ofFn]
  rw [get_graphVar]
  rfl

/-- Stage `graphStd`: the standard deviation block. -/
theorem get_graphStd (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ ssStd m n) (s := VecShape m) (idxStd (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := ssStd m n)
        (graphStd (m := m) (n := n) ε) xV) =
      valStd xV ε := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n) (ss := ssVarEps m n) (τ := VecShape m)
    (graphVarEps (m := m) (n := n) ε) (nodeStd (m := m) (n := n)) xV
    (idxStd (m := m) (n := n)) rfl).trans ?_
  simp only [nodeStd, sqrtClamp, elemwise, Node.forwardVec_ofFn]
  rw [get_graphVarEps]
  rfl

/-- Stage `graphInvStd`: the inverse standard deviation block. -/
theorem get_graphInvStd (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m])) (s := VecShape m)
      (idxInvStd (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := ssStd m n ++ [VecShape m])
        (graphInvStd (m := m) (n := n) ε) xV) =
      valInvStd xV ε := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n) (ss := ssStd m n) (τ := VecShape m)
    (graphStd (m := m) (n := n) ε) (nodeInvStd (m := m) (n := n)) xV
    (idxInvStd (m := m) (n := n)) rfl).trans ?_
  simp only [nodeInvStd, inv, elemwise, Node.forwardVec_ofFn]
  rw [get_graphStd]
  rfl

/-- Stage `graphInvStdBroadcast`: the broadcast inverse standard deviation block. -/
theorem get_graphInvStdBroadcast (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m, MatShape m n]))
      (s := MatShape m n) (idxInvStdBroadcast (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := ssStd m n ++ [VecShape m, MatShape m n])
        (graphInvStdBroadcast (m := m) (n := n) ε) xV) =
      valInvStdB xV ε := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n) (ss := ssStd m n ++ [VecShape m])
    (τ := MatShape m n) (graphInvStd (m := m) (n := n) ε) (nodeInvStdB (m := m) (n := n)) xV
    (idxInvStdBroadcast (m := m) (n := n)) rfl).trans ?_
  simp only [nodeInvStdB, broadcastRow, Node.forwardVec_ofFn, getVec]
  rw [get_graphInvStd]
  rfl

/-- The centered block is still available at stage `graphInvStdBroadcast`. -/
theorem get_centeredForNorm (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m, MatShape m n]))
      (s := MatShape m n) (idxCenteredForNorm (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := ssStd m n ++ [VecShape m, MatShape m n])
        (graphInvStdBroadcast (m := m) (n := n) ε) xV) =
      valCentered xV := by
  refine (Graph.get_evalVec_snoc_of_lt (Γ := ΓLN m n) (ss := ssStd m n ++ [VecShape m])
    (τ := MatShape m n) (graphInvStd (m := m) (n := n) ε) (nodeInvStdB (m := m) (n := n)) xV
    (idxCenteredForNorm (m := m) (n := n))
    (Idx.weaken (idxCentered (m := m) (n := n))
      [MatShape m n, VecShape m, VecShape m, VecShape m, VecShape m]) rfl).trans ?_
  refine (Graph.get_evalVec_snoc_of_lt (Γ := ΓLN m n) (ss := ssStd m n)
    (τ := VecShape m) (graphStd (m := m) (n := n) ε) (nodeInvStd (m := m) (n := n)) xV _
    (Idx.weaken (idxCentered (m := m) (n := n))
      [MatShape m n, VecShape m, VecShape m, VecShape m]) rfl).trans ?_
  refine (Graph.get_evalVec_snoc_of_lt (Γ := ΓLN m n) (ss := ssVarEps m n)
    (τ := VecShape m) (graphVarEps (m := m) (n := n) ε) (nodeStd (m := m) (n := n)) xV _
    (Idx.weaken (idxCentered (m := m) (n := n)) [MatShape m n, VecShape m, VecShape m])
    rfl).trans ?_
  refine (Graph.get_evalVec_snoc_of_lt (Γ := ΓLN m n)
    (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n, VecShape m])
    (τ := VecShape m) (graphVar (m := m) (n := n)) (nodeVarEps (m := m) (n := n) ε) xV _
    (Idx.weaken (idxCentered (m := m) (n := n)) [MatShape m n, VecShape m]) rfl).trans ?_
  refine (Graph.get_evalVec_snoc_of_lt (Γ := ΓLN m n)
    (ss := [VecShape m, MatShape m n, MatShape m n, MatShape m n])
    (τ := VecShape m) (graphCenteredSq (m := m) (n := n)) (nodeVar (m := m) (n := n)) xV _
    (Idx.weaken (idxCentered (m := m) (n := n)) [MatShape m n]) rfl).trans ?_
  refine (Graph.get_evalVec_snoc_of_lt (Γ := ΓLN m n)
    (ss := [VecShape m, MatShape m n, MatShape m n])
    (τ := MatShape m n) (graphCentered (m := m) (n := n)) (nodeCenteredSq (m := m) (n := n)) xV _
    (idxCentered (m := m) (n := n)) rfl).trans ?_
  exact get_graphCentered xV

/-- Stage `graphNormalized`: the normalized block. -/
theorem get_graphNormalized (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m, MatShape m n, MatShape m n]))
      (s := MatShape m n)
      (Idx.last (Γ := ΓLN m n) (ss := ssStd m n ++ [VecShape m, MatShape m n])
        (τ := MatShape m n))
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n])
        (graphNormalized (m := m) (n := n) ε) xV) =
      valNorm xV ε := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n]) (τ := MatShape m n)
    (graphInvStdBroadcast (m := m) (n := n) ε) (nodeNorm (m := m) (n := n)) xV _ rfl).trans ?_
  simp only [nodeNorm, mul, Node.forwardVec_ofFn]
  rw [get_centeredForNorm, get_graphInvStdBroadcast]
  rfl

/-- Stage `graphGammaBroadcast`: the broadcast scale block. -/
theorem get_graphGammaBroadcast (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get
      (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]))
      (s := MatShape m n) (idxGammaBroadcast (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n])
        (graphGammaBroadcast (m := m) (n := n) ε) xV) =
      valGammaB xV := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n]) (τ := MatShape m n)
    (graphNormalized (m := m) (n := n) ε) (nodeGammaB (m := m) (n := n)) xV
    (idxGammaBroadcast (m := m) (n := n)) rfl).trans ?_
  simp only [nodeGammaB, broadcastCol, Node.forwardVec_ofFn, getVec]
  rw [Graph.get_evalVec_input (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n])
    (graphNormalized (m := m) (n := n) ε) xV _ (idxInputGamma (m := m) (n := n)) rfl]
  rfl

/-- The normalized block is still available at stage `graphGammaBroadcast`. -/
theorem get_normForScale (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get
      (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n]))
      (s := MatShape m n) (idxNormForScale (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n])
        (graphGammaBroadcast (m := m) (n := n) ε) xV) =
      valNorm xV ε := by
  refine (Graph.get_evalVec_snoc_of_lt (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n]) (τ := MatShape m n)
    (graphNormalized (m := m) (n := n) ε) (nodeGammaB (m := m) (n := n)) xV
    (idxNormForScale (m := m) (n := n))
    (Idx.last (Γ := ΓLN m n) (ss := ssStd m n ++ [VecShape m, MatShape m n])
      (τ := MatShape m n)) rfl).trans ?_
  exact get_graphNormalized ε xV

/-- Stage `graphScaled`: the scaled block. -/
theorem get_graphScaled (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get
      (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
        MatShape m n]))
      (s := MatShape m n)
      (Idx.last (Γ := ΓLN m n)
        (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n])
        (τ := MatShape m n))
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
          MatShape m n])
        (graphScaled (m := m) (n := n) ε) xV) =
      valScaled xV ε := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n])
    (τ := MatShape m n) (graphGammaBroadcast (m := m) (n := n) ε) (nodeScaled (m := m) (n := n))
    xV _ rfl).trans ?_
  simp only [nodeScaled, mul, Node.forwardVec_ofFn]
  rw [get_normForScale, get_graphGammaBroadcast]
  rfl

/-- Stage `graphBetaBroadcast`: the broadcast shift block. -/
theorem get_graphBetaBroadcast (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get
      (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
        MatShape m n, MatShape m n]))
      (s := MatShape m n) (idxBetaBroadcast (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
          MatShape m n, MatShape m n])
        (graphBetaBroadcast (m := m) (n := n) ε) xV) =
      valBetaB xV := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m n])
    (τ := MatShape m n) (graphScaled (m := m) (n := n) ε) (nodeBetaB (m := m) (n := n)) xV
    (idxBetaBroadcast (m := m) (n := n)) rfl).trans ?_
  simp only [nodeBetaB, broadcastCol, Node.forwardVec_ofFn, getVec]
  rw [Graph.get_evalVec_input (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m n])
    (graphScaled (m := m) (n := n) ε) xV _ (idxInputBeta (m := m) (n := n)) rfl]
  rfl

/-- The scaled block is still available at stage `graphBetaBroadcast`. -/
theorem get_scaledForOutput (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get
      (Γ := ΓLN m n ++ (ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
        MatShape m n, MatShape m n]))
      (s := MatShape m n) (idxScaledForOutput (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n)
        (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n,
          MatShape m n, MatShape m n])
        (graphBetaBroadcast (m := m) (n := n) ε) xV) =
      valScaled xV ε := by
  refine (Graph.get_evalVec_snoc_of_lt (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m n])
    (τ := MatShape m n) (graphScaled (m := m) (n := n) ε) (nodeBetaB (m := m) (n := n)) xV
    (idxScaledForOutput (m := m) (n := n))
    (Idx.last (Γ := ΓLN m n)
      (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n])
      (τ := MatShape m n)) rfl).trans ?_
  exact get_graphScaled ε xV

/-- The output block of the full LayerNorm graph. -/
theorem get_idxY (ε : ℝ) (xV : CtxVec (ΓLN m n)) :
    CtxVec.get (Γ := ΓLN m n ++ ssLayerNorm m n) (s := MatShape m n) (idxY (m := m) (n := n))
      (Graph.evalVec (Γ := ΓLN m n) (ss := ssLayerNorm m n)
        (layerNormGraph (m := m) (n := n) ε) xV) =
      valY xV ε := by
  refine (Graph.get_evalVec_snoc_last (Γ := ΓLN m n)
    (ss := ssStd m n ++ [VecShape m, MatShape m n, MatShape m n, MatShape m n, MatShape m n,
      MatShape m n])
    (τ := MatShape m n) (graphBetaBroadcast (m := m) (n := n) ε) (nodeY (m := m) (n := n)) xV
    (idxY (m := m) (n := n)) rfl).trans ?_
  simp only [nodeY, add, Node.forwardVec_ofFn]
  rw [get_scaledForOutput, get_graphBetaBroadcast]
  rfl

/-! ## Positivity of the denominators -/

/-- The variance block is a mean of squares, hence nonnegative. -/
theorem valVar_nonneg (xV : CtxVec (ΓLN m n)) (i : Fin (Spec.Shape.size (VecShape m))) :
    0 ≤ valVar xV i := by
  simp only [valVar, castVec_apply]
  rw [RowNorm.rowMeanCLM_eq, RowNorm.rowMean]
  refine div_nonneg (Finset.sum_nonneg fun j _ => ?_) (Nat.cast_nonneg n)
  simp only [valCenteredSq]
  exact mul_self_nonneg _

/-- With positive `ε`, every shifted variance is positive. -/
theorem valVarEps_pos {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ΓLN m n))
    (i : Fin (Spec.Shape.size (VecShape m))) :
    0 < valVarEps xV ε i := by
  simp only [valVarEps, vecOfFun_apply]
  exact add_pos_of_nonneg_of_pos (valVar_nonneg xV i) hε

/-- With positive `ε`, every clamped standard deviation is positive. -/
theorem valStd_pos {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ΓLN m n))
    (i : Fin (Spec.Shape.size (VecShape m))) :
    0 < valStd xV ε i := by
  simp only [valStd, vecOfFun_apply]
  exact Real.sqrt_pos.2 (lt_max_of_lt_left (valVarEps_pos hε xV i))

/-- The LayerNorm `sqrt` domain hypothesis follows from `0 < ε`. -/
theorem varEps_pos_of_eps_pos {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ΓLN m n)) :
    ∀ i : Fin (Spec.Shape.size (VecShape m)),
      0 < CtxVec.get (Γ := ΓLN m n ++ ssVarEps m n) (s := VecShape m) (idxVarEps (m := m) (n := n))
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssVarEps m n)
          (graphVarEps (m := m) (n := n) ε) xV) i := by
  intro i
  rw [get_graphVarEps]
  exact valVarEps_pos hε xV i

/-- The LayerNorm `inv` domain hypothesis follows from `0 < ε`. -/
theorem std_ne_zero_of_eps_pos {ε : ℝ} (hε : 0 < ε) (xV : CtxVec (ΓLN m n)) :
    ∀ i : Fin (Spec.Shape.size (VecShape m)),
      CtxVec.get (Γ := ΓLN m n ++ ssStd m n) (s := VecShape m) (idxStd (m := m) (n := n))
        (Graph.evalVec (Γ := ΓLN m n) (ss := ssStd m n)
          (graphStd (m := m) (n := n) ε) xV) i ≠ 0 := by
  intro i
  rw [get_graphStd]
  exact (valStd_pos hε xV i).ne'

end LayerNorm

end

end Autograd
end Proofs
