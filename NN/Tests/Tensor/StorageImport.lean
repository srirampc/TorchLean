/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Tensor.Storage

/-!
# Standalone storage import

The public storage facade must expose the class and its instances without the tensor umbrella.
-/

example : TorchLean.Storage Float := inferInstance
example : TorchLean.Storage UInt8 := inferInstance
example : TorchLean.Storage Nat := inferInstance
