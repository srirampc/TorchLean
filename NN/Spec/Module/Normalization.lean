/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.Core
public import NN.Spec.Module.Core

/-!
# Normalization module wrappers

This file wraps selected normalization specs as `Spec.Module`s for composition/export.

For `Spec.Module.layerNorm`, the final axis is the feature axis and every preceding axis is treated
as a batch axis. The implementation flattens those leading axes only while applying the
matrix-level normalization semantics.

PyTorch mental picture: `nn.LayerNorm(embedDim)` applied at each timestep, with `weight=gamma` and
`bias=beta`.
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
LayerNorm over the final axis, wrapped as a `Spec.Module`.

The epsilon is passed unchanged to `Spec.layerNorm`; there is no representability check or
fallback when it rounds to zero. Tiny formats may require an explicitly chosen positive finite
epsilon. The Python expression records only the width, omitting the supplied epsilon and affine
parameters.
-/
def layerNorm (leading : Shape) (width : Nat)
  (gamma : Tensor α [width])
  (beta : Tensor α [width])
  (hLeading : 0 < leading.size)
  (hWidth : 0 < width)
  (epsilon : α := TorchLean.normalizationEpsilon) :
  Spec.Module α (leading.appendDim width) (leading.appendDim width) :=
  { forward := fun x =>
      let matrix : Tensor α [leading.size, width] :=
        reshapeSpec x (by simp [Shape.size_appendDim, Shape.size])
      let normalized :=
        Spec.layerNorm (α := α) (seqLen := leading.size) (embedDim := width)
          matrix gamma beta hLeading hWidth epsilon
      reshapeSpec normalized (by simp [Shape.size_appendDim, Shape.size])
    kind := "LayerNorm"
    pythonExpr := s!"nn.LayerNorm({width})" }

end Spec.Module
