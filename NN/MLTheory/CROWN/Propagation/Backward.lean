/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Models.Mlp

/-!
# Backward CROWN propagation

Backward affine-bound propagation for tighter neural-network certificates.

This module implements the core *backward* CROWN idea: rather than pushing interval/affine bounds
forward layer by layer, start from an output objective and propagate its affine dependence on the
input backwards through the network.

At a high level, for each hidden layer `l` we maintain affine coefficients

`A^(l) x + b^(l)`

that over-approximate the chosen output objective, and update them using diagonal activation
relaxations `Λ^(l)` together with the layer weights/biases.

References / citations:
- Huan Zhang et al., “Efficient Neural Network Robustness Certification with General Activation
  Functions”, NeurIPS 2018.
  https://proceedings.neurips.cc/paper/2018/hash/d04863f100d59b3eb688a11f95b0ae60-Abstract.html
- Singh et al., “An Abstract Domain for Certifying Neural Networks” (DeepPoly), POPL 2019.

We keep this as the compact layer-wise backward propagation surface. The graph-IR verifier lives in
`NN.MLTheory.CROWN.Graph`; both modules use the same mathematical idea, but they serve different
integration points.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Propagation.Backward

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Per-neuron activation relaxation parameters. -/
structure NeuronRelax (α : Type) where
  /-- Slope of the lower affine envelope. -/
  slopeLower : α
  /-- Bias of the lower affine envelope. -/
  biasLower : α
  /-- Slope of the upper affine envelope. -/
  slopeUpper : α
  /-- Bias of the upper affine envelope. -/
  biasUpper : α

/-- Layer-wise relaxation parameters. -/
structure LayerRelax (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of neurons covered by this relaxation record. -/
  dim : Nat
  /-- Per-neuron affine envelope parameters. -/
  params : Tensor (NeuronRelax α) [dim]

/-- Extract relaxation slopes as diagonal matrix (for lower bound). -/
def layerSlopesLower {n : Nat} (relax : LayerRelax α) (h : relax.dim = n) : Tensor α [n, n] :=
  h ▸ Tensor.dim fun i : Fin relax.dim =>
    Tensor.dim fun j : Fin relax.dim =>
      Tensor.scalar (if decide (i.val = j.val) then (relax.params.getScalar i).slopeLower else 0)

/-- Extract relaxation slopes as diagonal matrix (for upper bound). -/
def layerSlopesUpper {n : Nat} (relax : LayerRelax α) (h : relax.dim = n) : Tensor α [n, n] :=
  h ▸ Tensor.dim fun i : Fin relax.dim =>
    Tensor.dim fun j : Fin relax.dim =>
      Tensor.scalar (if decide (i.val = j.val) then (relax.params.getScalar i).slopeUpper else 0)

/-- Extract bias vector (for lower bound). -/
def layerBiasLower {n : Nat} (relax : LayerRelax α) (h : relax.dim = n) : Tensor α [n] :=
  h ▸ Tensor.dim fun i : Fin relax.dim => Tensor.scalar (relax.params.getScalar i).biasLower

/-- Extract bias vector (for upper bound). -/
def layerBiasUpper {n : Nat} (relax : LayerRelax α) (h : relax.dim = n) : Tensor α [n] :=
  h ▸ Tensor.dim fun i : Fin relax.dim => Tensor.scalar (relax.params.getScalar i).biasUpper

/-- Network structure for backward propagation: weights, biases, and pre-computed activation
relaxations, indexed by layer. -/
structure BackwardNetwork (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of layers (not counting input) -/
  numLayers : Nat
  /-- Input dimension -/
  inDim : Nat
  /-- Output dimension -/
  outDim : Nat
  /-- Layer dimensions: dims[i] is output dim of layer i -/
  dims : Array Nat
  /-- Weight matrices: W[i] has shape [dims[i], dims[i-1]] -/
  weights : Array (Σ m n : Nat, Tensor α [m, n])
  /-- Bias vectors: b[i] has shape [dims[i]] -/
  biases : Array (Σ n : Nat, Tensor α [n])
  /-- Per-layer activation relaxations (empty for output layer) -/
  relaxations : Array (LayerRelax α)

/-- Backward state during propagation: the affine forms `A x + b` bounding the objective from
below and from above. -/
structure BackwardState (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Coefficient matrix `A` of the lower affine bound (`output_dim × input_dim`). -/
  coeffLower : Σ m n : Nat, Tensor α [m, n]
  /-- Coefficient matrix `A` of the upper affine bound (`output_dim × input_dim`). -/
  coeffUpper : Σ m n : Nat, Tensor α [m, n]
  /-- Bias vector `b` of the lower affine bound (`output_dim`). -/
  biasLower : Σ n : Nat, Tensor α [n]
  /-- Bias vector `b` of the upper affine bound (`output_dim`). -/
  biasUpper : Σ n : Nat, Tensor α [n]

/-- Helper: matrix multiplication for sigma-typed tensors. -/
def sigmaMatMul (A : Σ m n : Nat, Tensor α [m, n])
    (B : Σ p q : Nat, Tensor α [p, q]) :
    Option (Σ m q : Nat, Tensor α [m, q]) :=
  let ⟨m, n, matA⟩ := A
  let ⟨p, q, matB⟩ := B
  if h : n = p then
    some ⟨m, q, Spec.matMulSpec (α := α) matA (by cases h; exact matB)⟩
  else
    none

/-- Helper: matrix-vector multiplication for sigma-typed tensors. -/
def sigmaMatVecMul (A : Σ m n : Nat, Tensor α [m, n])
    (v : Σ n : Nat, Tensor α [n]) :
    Option (Σ m : Nat, Tensor α [m]) :=
  let ⟨m, n, matA⟩ := A
  let ⟨p, vecV⟩ := v
  if h : n = p then
    some ⟨m, Spec.matVecMulSpec (α := α) matA (by cases h; exact vecV)⟩
  else
    none

/-- Helper: vector addition for sigma-typed tensors. -/
def sigmaVecAdd (v1 v2 : Σ n : Nat, Tensor α [n]) :
    Option (Σ n : Nat, Tensor α [n]) :=
  let ⟨n1, lhsTensor⟩ := v1
  let ⟨n2, rhsTensor⟩ := v2
  if h : n1 = n2 then
    some ⟨n1, Tensor.addSpec lhsTensor (by cases h; exact rhsTensor)⟩
  else
    none

/-- Initialize backward state with identity at output. -/
def initBackwardState (outDim : Nat) : BackwardState α :=
  let identity : Tensor α [outDim, outDim] :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        Tensor.scalar (if decide (i.val = j.val) then 1 else 0)))
  let zero : Tensor α [outDim] :=
    Tensor.full (α := α) (.dim outDim .scalar) 0
  { coeffLower := ⟨outDim, outDim, identity⟩
  , coeffUpper := ⟨outDim, outDim, identity⟩
  , biasLower := ⟨outDim, zero⟩
  , biasUpper := ⟨outDim, zero⟩ }

/-- Process one layer in backward propagation.

    For layer l with weight W, bias b, and relaxation Λ:
    A^(l-1) = A^(l) · Λ · W
    b^(l-1) = A^(l) · (Λ · b + offset) + b^(l)

    Here we handle lower and upper bounds separately. The result is `none` when the weight rows,
    the bias, and the relaxation do not cover the same neurons, or when the running state does not
    compose with this layer.
-/
def backwardOneLayer (state : BackwardState α)
    (W : Σ m n : Nat, Tensor α [m, n])
    (bias : Σ n : Nat, Tensor α [n])
    (relax : LayerRelax α) : Option (BackwardState α) := do
  let ⟨wm, wn, matW⟩ := W
  let ⟨bn, vecB⟩ := bias

  -- Check that the matrix, bias, and relaxation dimensions match.
  if h : wm = relax.dim ∧ bn = wm then
    -- Build diagonal slope matrices for the relaxation dimension
    let slopesLower := layerSlopesLower (n := relax.dim) relax rfl
    let slopesUpper := layerSlopesUpper (n := relax.dim) relax rfl
    let offsetsLower := layerBiasLower (n := relax.dim) relax rfl
    let offsetsUpper := layerBiasUpper (n := relax.dim) relax rfl

    -- A_new = A · diag(slope) · W, with the weight rows indexed by the relaxation dimension.
    let matW' : Tensor α [relax.dim, wn] := h.1 ▸ matW
    let tempLower := Spec.matMulSpec (α := α) slopesLower matW'
    let coeffLowerNext ← sigmaMatMul state.coeffLower ⟨relax.dim, wn, tempLower⟩

    let tempUpper := Spec.matMulSpec (α := α) slopesUpper matW'
    let coeffUpperNext ← sigmaMatMul state.coeffUpper ⟨relax.dim, wn, tempUpper⟩

    -- b_new = A · (diag(slope) · bias + offset) + b, the inner sum taken elementwise.
    let scaledBiasLower : Tensor α [relax.dim] :=
      Tensor.dim (fun i =>
        let offset := offsetsLower.getScalar i
        let vectorIndex : Fin bn := ⟨i.val, by rw [h.2, h.1]; exact i.isLt⟩
        let vectorValue := vecB.getScalar vectorIndex
        let slope := Spec.get2 slopesLower i i
        Tensor.scalar (slope * vectorValue + offset))

    let scaledBiasUpper : Tensor α [relax.dim] :=
      Tensor.dim (fun i =>
        let offset := offsetsUpper.getScalar i
        let vectorIndex : Fin bn := ⟨i.val, by rw [h.2, h.1]; exact i.isLt⟩
        let vectorValue := vecB.getScalar vectorIndex
        let slope := Spec.get2 slopesUpper i i
        Tensor.scalar (slope * vectorValue + offset))

    -- Multiply by current A and add to b
    let shiftLower ← sigmaMatVecMul state.coeffLower ⟨relax.dim, scaledBiasLower⟩
    let biasLowerNext ← sigmaVecAdd shiftLower state.biasLower

    let shiftUpper ← sigmaMatVecMul state.coeffUpper ⟨relax.dim, scaledBiasUpper⟩
    let biasUpperNext ← sigmaVecAdd shiftUpper state.biasUpper

    return {
      coeffLower := coeffLowerNext
      coeffUpper := coeffUpperNext
      biasLower := biasLowerNext
      biasUpper := biasUpperNext
    }
  else
    none

/-- A weight whose row count differs from the relaxation dimension, or a bias whose length differs
from the row count, is rejected by `backwardOneLayer`. -/
theorem backwardOneLayer_eq_none (state : BackwardState α) {wm wn bn : Nat}
    (matW : Tensor α [wm, wn]) (vecB : Tensor α [bn]) (relax : LayerRelax α)
    (h : ¬ (wm = relax.dim ∧ bn = wm)) :
    backwardOneLayer state ⟨wm, wn, matW⟩ ⟨bn, vecB⟩ relax = none := by
  simp [backwardOneLayer, h]

/-- Run the backward propagation through the network, from the output layer to the input.

Layers are visited in decreasing index order. A layer whose weight, bias, and relaxation
dimensions disagree, or that does not compose with the running state, makes the whole run fail
with `none`; a partially propagated state is never returned. A layer index missing from any of the
three arrays is skipped. -/
def runBackward (net : BackwardNetwork α) : Option (BackwardState α) := do
  let mut state := initBackwardState (α := α) net.outDim
  for layerIdx in (List.range net.numLayers).reverse do
    if hlw : layerIdx < net.weights.size then
      if hlb : layerIdx < net.biases.size then
        if hlr : layerIdx < net.relaxations.size then
          state ← backwardOneLayer (α := α) state net.weights[layerIdx] net.biases[layerIdx]
            net.relaxations[layerIdx]
  return state

/-- A one-layer network whose weight has two rows while its relaxation covers one neuron. -/
private noncomputable def mismatchedNetwork : BackwardNetwork ℝ where
  numLayers := 1
  inDim := 1
  outDim := 2
  dims := #[2]
  weights := #[⟨2, 1, Tensor.full (α := ℝ) (.dim 2 (.dim 1 .scalar)) 0⟩]
  biases := #[⟨2, Tensor.full (α := ℝ) (.dim 2 .scalar) 0⟩]
  relaxations := #[{
    dim := 1
    params := Tensor.dim fun _ =>
      Tensor.scalar { slopeLower := 0, biasLower := 0, slopeUpper := 0, biasUpper := 0 } }]

/-- A dimension mismatch between a weight and its relaxation is rejected rather than skipped. -/
private theorem runBackward_mismatchedNetwork_eq_none :
    runBackward mismatchedNetwork = none := rfl

/-- Evaluate backward bounds on an input box.

Given $Ax+b$ with $x\in[\mathrm{lo},\mathrm{hi}]$, compute the corresponding output interval.
-/
def evalBackwardBounds (outDim inDim : Nat) (state : BackwardState α)
    [BoundOps α] (xB : Box α (.dim inDim .scalar)) :
    Option (Box α (.dim outDim .scalar)) :=
  let ⟨mL, nL, coeffLo⟩ := state.coeffLower
  let ⟨mU, nU, coeffHi⟩ := state.coeffUpper
  let ⟨bDimL, biasLo⟩ := state.biasLower
  let ⟨bDimU, biasHi⟩ := state.biasUpper

  if hmL : mL = outDim then
    if hnL : nL = inDim then
      if hmU : mU = outDim then
        if hnU : nU = inDim then
          if hbL : bDimL = outDim then
            if hbU : bDimU = outDim then
              by
                cases hmL; cases hnL; cases hmU; cases hnU; cases hbL; cases hbU
                let bBLower : Box α (.dim outDim .scalar) := { lo := biasLo, hi := biasLo }
                let bBUpper : Box α (.dim outDim .scalar) := { lo := biasHi, hi := biasHi }
                let yLower := NN.MLTheory.CROWN.IBP.linear (α := α) coeffLo xB bBLower
                let yUpper := NN.MLTheory.CROWN.IBP.linear (α := α) coeffHi xB bBUpper
                exact some { lo := yLower.lo, hi := yUpper.hi }
            else
              none
          else
            none
        else
          none
      else
        none
    else
      none
  else
    none

/-- Compute ReLU relaxation parameters from pre-activation bounds.

The inactive test is `u < 0`, so the degenerate interval `u = 0` takes the crossing branch and
evaluates `u / (u - l)` there (dividing zero by zero when `l = 0` as well).
`Runtime.Ops.ReLU.relaxScalar` tests `u > 0` and treats `u = 0` as inactive; the two relaxations
are deliberately not unified. -/
def computeReLURelax (n : Nat) (preB : Box α (.dim n .scalar)) : LayerRelax α :=
  let params := Tensor.dim (fun i : Fin n =>
    let l := preB.lo.getScalar i
    let u := preB.hi.getScalar i
    let relax : NeuronRelax α :=
      if u < 0 then
        -- Inactive: y = 0
        { slopeLower := 0
        , biasLower := 0
        , slopeUpper := 0
        , biasUpper := 0 }
      else if l > 0 then
        -- Active: y = x
        { slopeLower := 1
        , biasLower := 0
        , slopeUpper := 1
        , biasUpper := 0 }
      else
        -- Crossing: lower y ≥ 0, upper y ≤ αx - αl
        let α := u / (u - l)
        { slopeLower := 0  -- Conservative lower
        , biasLower := 0
        , slopeUpper := α
        , biasUpper := -(α * l) }
    Tensor.scalar relax)
  { dim := n, params := params }

end NN.MLTheory.CROWN.Propagation.Backward
