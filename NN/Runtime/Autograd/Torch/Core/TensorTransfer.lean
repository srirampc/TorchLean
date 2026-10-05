/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.IEEE754.Native
public import NN.Spec.Core.Tensor.Core

/-!
# Runtime Tensor Transfer

Backend-neutral conversion between executable tensors and the host `Float` representation used at
native runtime boundaries.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean
open FloatLib.Floats

/--
Conversion between an executable scalar type and the host `Float` representation used at native
runtime boundaries.

Native transfer and host inspection are separate operations. A scalar semantics may allow every
element to be inspected as `Float` without claiming that a native tensor has the same arithmetic or
storage representation.
-/
class TensorTransfer (α : Type) [TorchLean.Storage α] where
  /-- Encode a tensor in the host representation used by native runtimes. -/
  toFloatTensor : {s : Shape} → Tensor α s → IO (Tensor Float s)
  /-- Decode a tensor received from a native runtime without changing its shape. -/
  ofFloatTensor : {s : Shape} → Tensor Float s → IO (Tensor α s)
  /-- Convert one scalar to the host representation used by native runtimes. -/
  toFloat : α → IO Float
  /-- Read a tensor for host-side inspection without claiming a native storage representation. -/
  readFloatTensor : {s : Shape} → Tensor α s → IO (Tensor Float s) := toFloatTensor

/-- `Float` transfers preserve the runtime's host representation. -/
instance (priority := 1000) : TensorTransfer Float where
  toFloatTensor := fun tensor => pure tensor
  ofFloatTensor := fun tensor => pure tensor
  toFloat := pure

/-- Native binary32 transfers preserve the runtime's float32 wire representation. -/
instance (priority := 1000) : TensorTransfer Float32 where
  toFloatTensor := fun tensor => pure (TorchLean.Tensor.map Float32.toFloat tensor)
  ofFloatTensor := fun tensor => pure (TorchLean.Tensor.map Float.toFloat32 tensor)
  toFloat := fun x => pure x.toFloat

/--
Host-side conversion for FloatLib's configured IEEE-754 binary32 scalar.

The configured arithmetic uses software kernels. Scalar and tensor readback to `Float` remain
available for reports and checkpoints, but this does not select native CUDA arithmetic. Finite
binary32 values embed exactly in binary64; the native conversion canonicalizes NaN payloads.
-/
instance (priority := 1000) :
    TensorTransfer (ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)) where
  toFloatTensor := fun {_s} _ =>
    throw <| IO.userError
      "torch: configured binary32 supports host readback; select native arithmetic for CUDA"
  ofFloatTensor := fun {_s} _ =>
    throw <| IO.userError
      "torch: configured binary32 supports host readback; select native arithmetic for CUDA"
  toFloat := fun x => pure (ExecFloat.Binary.toFloat32 x).toFloat
  readFloatTensor := fun tensor => pure <|
    TorchLean.Tensor.map (fun x => (ExecFloat.Binary.toFloat32 x).toFloat) tensor

/--
CPU-preserving fallback for scalar types without a native tensor representation.

Add a higher-priority `TensorTransfer α` instance when a scalar type has an honest host `Float`
encoding. CPU execution does not invoke these failing transfer operations.
-/
instance (priority := 10) (α : Type) [TorchLean.Storage α] : TensorTransfer α where
  toFloatTensor := fun {_s} _ =>
    throw <| IO.userError <|
      "torch: this scalar type cannot be converted to a native tensor; " ++
        "use Float for native execution or run this scalar on CPU"
  ofFloatTensor := fun {_s} _ =>
    throw <| IO.userError <|
      "torch: native tensors cannot be converted to this scalar type; " ++
        "use Float for native execution or run this scalar on CPU"
  toFloat := fun _ =>
    throw <| IO.userError <|
      "torch: this scalar type cannot be converted to a native scalar; " ++
        "use Float for native execution or run this scalar on CPU"

end Torch
end Autograd
end Runtime
