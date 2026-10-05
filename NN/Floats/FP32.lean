/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
public import FloatLib.Floats.Formats.Flocq

/-!
# TorchLean's rounded-real binary32 specialization

`FP32` specializes FloatLib's `NF` to binary32 precision and gradual underflow, with no upper
exponent bound. Each arithmetic operation rounds its real result to this grid. The underlying
type stores a real value without a representability invariant; use `NF.ofReal` to round an input.

FloatLib supplies the rounding theorems directly: `Model.abs_roundAt_sub_le fmt` bounds one
rounding step, and `round_nearestEven_computed` gives its mantissa/exponent representation.
Encoded values and exceptional arithmetic use `ExecFloat.Binary 8 23`.
-/

@[expose] public section

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean.Floats

/-- Rounded-real binary32 arithmetic, without an upper exponent bound. -/
abbrev FP32 : Type := NF binaryRadix (Model.fexpOf FloatFormat.binary32) nearestEven

/-- Interpret a rounded-real scalar as a real number. -/
abbrev FP32.toReal (x : FP32) : ℝ := x.val

end TorchLean.Floats
