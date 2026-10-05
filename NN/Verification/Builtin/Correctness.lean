/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Lowering.API

/-!
# Correctness

TorchLean→IR correctness helpers.

This file contains no correctness theorem itself. It provides the two executable bridges that the
lowering theorems under `NN.Verification.Builtin.Proved` (in particular
`Proved.Correctness.runForwardIR_eq_evalForward`) are stated about:

- convert a verifier `ParamStore` into an IR `Payload` for `NN.IR.Graph.denote`;
- evaluate a `LoweredIR` graph on a concrete input.
-/

@[expose] public section


namespace NN.Verification.Builtin

open NN.IR

/--
Convert a verifier `ParamStore` into an IR `Payload` for `NN.IR.Graph.denote`.

This is the bridge between the CROWN/LiRPA parameter representation used by the verification
pipeline and the executable IR semantics.
-/
def payloadOfParamStore {α : Type} [TorchLean.Storage α] [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) : Payload α :=
  { const? := fun id =>
      (ps.constVals.get? id).map (fun c =>
        { n := c.n, v := c.v })
    linear? := fun id =>
      (ps.linearWB.get? id).map (fun p =>
        { outDim := p.m, inDim := p.n, W := p.w, b := p.b })
    conv? := ps.convCfg.get?
    batchNormEval? := ps.batchNormEval.get?
    layerNorm? := ps.layerNorm.get? }

/-- Evaluate a `LoweredIR` forward graph on an input tensor, returning a shape-checked tensor. -/
def runForwardIR
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape outShape : Spec.Shape}
    (c : LoweredIR α) (x : TorchLean.Tensor α inShape) :
    Except String (TorchLean.Tensor α outShape) := do
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x
  let out ←
    Graph.denote (α := α) (g := c.graph) (payload := payloadOfParamStore (α := α) c.ps)
      (input := input) (outputId := c.outputId)
  if h : out.shape = outShape then
    pure (h ▸ out.tensor)
  else
    throw <|
      s!"TorchLeanCorrectness: output shape mismatch: " ++
        s!"produced={repr out.shape}, expected={repr outShape}"

end NN.Verification.Builtin
