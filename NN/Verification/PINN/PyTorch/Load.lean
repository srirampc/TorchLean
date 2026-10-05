/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Verification.PINN.Architecture

/-!
# PINN PyTorch Checkpoint Loading

PINN weight import (JSON → typed tensors).

This is the “loading” half of the PINN PyTorch bridge:

- read a PyTorch-style state dictionary encoded as JSON,
- parse each weight/bias matrix into a shape-checked `Tensor Float ...` (shape inferred/checked),
- infer a TorchLean `SequentialPINNArch` from the layer shapes plus optional activation metadata.

The generic JSON helpers (`loadWeights?`, `parseTensor`, `inferMatrixDims`, …) live in
`NN/Runtime/PyTorch/Import/Core.lean`.
-/

@[expose] public section


namespace Import
namespace PINNPyTorch

open Spec TorchLean TorchLean.Tensor Shape
open Lean Data Json

open NN.Verification.PINN
open Import.PyTorch

/-
We keep the state representation explicit:

- `PinnLayer` stores the raw weight/bias tensors as trained in Python,
- `PinnState` pairs those tensors with the inferred sequential PINN architecture.
-/

/-- One fully-connected layer of a PINN (weights + bias) as trained in PyTorch. -/
structure PinnLayer where
  /-- Input dimension. -/
  inDim  : Nat
  /-- Output dimension. -/
  outDim : Nat
  /-- Layer weight matrix, shaped as PyTorch stores `Linear.weight`. -/
  weights : Tensor Float [outDim, inDim]
  /-- Layer bias vector. -/
  bias    : Tensor Float [outDim]

/-- A parsed PINN state dict together with the inferred TorchLean sequential PINN architecture. -/
structure PinnState where
  /-- Inferred sequential fully-connected PINN architecture. -/
  arch   : SequentialPINNArch
  /-- Layer stack. -/
  layers : Array PinnLayer

/-!
## Activation metadata

Python training scripts often record which nonlinearity they used. We treat that as optional
metadata under `meta.activation`. If it is missing we default to `tanh`; an unrecognized value
rejects the checkpoint instead of silently bounding it as a tanh network.
-/

namespace Internal

/--
Parse optional activation metadata from JSON (`meta.activation`).

An absent `meta` object or `activation` field selects `tanh`. A present value must name a
supported nonlinearity (`tanh`, `relu`, or `sin`/`sine`/`siren`); anything else fails the load.
-/
def parseActivation (metaOpt : Option Json) : Option HiddenActivation :=
  match metaOpt with
  | none => some HiddenActivation.tanh
  | some (Json.obj metaObj) =>
    match metaObj.get? "activation" with
    | none => some HiddenActivation.tanh
    | some (Json.str s) =>
      match s.toLower with
      | "tanh" => some HiddenActivation.tanh
      | "relu" => some HiddenActivation.relu
      | "sin" | "sine" | "siren" => some HiddenActivation.sin
      | _ => none
    | some _ => none
  | some _ => none

end Internal

/--
Load a PINN state dict with arbitrary hidden widths.

Expected keys:

- `layers.<i>.weight` and `layers.<i>.bias` for each layer index `i`
- optional `meta.activation` (see `Internal.parseActivation`; unsupported values reject)

Unlike fixed-shape examples, we infer `(outDim, inDim)` for each layer from the JSON matrix shape.
-/
def loadPinnState (j : Json) : Option PinnState := do
  -- Accepts both `{...}` and `{ "params": {...} }`.
  let o ← loadWeights? j

  -- Collect `i` from `layers.<i>.weight` keys, then parse in ascending order.
  let weightIdxs :=
    (o.toList.foldl
      (fun acc kv =>
        match parseIndexedKey "layers." ".weight" kv.fst with
        | some idx => idx :: acc
        | none => acc)
      ([] : List Nat)).toArray.qsort (· < ·)

  match weightIdxs[0]? with
  | none => none
  | some _ =>
    let activation ← Internal.parseActivation (o.get? "meta")
    let layers ←
      weightIdxs.foldlM (fun acc idx => do
        let base := s!"layers.{idx}"
        let wJson ← o.get? (base ++ ".weight")
        let bJson ← o.get? (base ++ ".bias")
        let (outDim, inDim) ← inferMatrixDims wJson
        let weights ← parseTensor (.dim outDim (.dim inDim .scalar)) wJson
        let bias ← parseTensor (.dim outDim .scalar) bJson
        pure (acc.push { inDim := inDim, outDim := outDim, weights := weights, bias := bias })) #[]

    match layers[0]? with
    | none => none
    | some first =>
      let outDims := layers.map (·.outDim)
      let hidden := outDims.extract 0 (outDims.size - 1)
      let outputDim := outDims.back?.getD 0
      let arch : SequentialPINNArch :=
        { inputDim := first.inDim
          hiddenDims := hidden
          outputDim := outputDim
          activation := activation }
      pure { arch := arch, layers := layers }

end PINNPyTorch
end Import
