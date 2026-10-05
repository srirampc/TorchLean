/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.ShapeChange
public import NN.Spec.Module.Core

/-!
# Flatten module wrapper

`flattenSpec` converts a tensor of shape `s` into a vector of length `Spec.Shape.size s`.

Why is the output length computed at the type level?

- It reflects the spec contract: flattening is just a re-indexing, so the number of elements is
  determined entirely by the input shape.
- It prevents a common class of downstream mistakes (e.g. wiring a linear layer with the wrong
  feature dimension).

Here `s` is the per-sample shape. The `nn.Flatten()` metadata assumes the Python input has an
additional leading batch axis: PyTorch preserves that axis and flattens the sample dimensions.
Applying the same operation directly to an unbatched Python tensor requires
`nn.Flatten(start_dim=0)`.
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

-- Flatten module specification wrapper
/-- Wrap `flattenSpec` as an `Spec.Module` (`s -> (Spec.Shape.size s)`).

The output length comes from the typed input shape; `pythonExpr` follows the external batch-axis
convention described above.
-/
def flatten (α : Type) [TorchLean.Storage α] (s : Shape) :
  Spec.Module α s ([(Spec.Shape.size s)]) :=
{ forward := fun x => flattenSpec x, kind := "Flatten", pythonExpr := "nn.Flatten()" }

end Spec.Module
