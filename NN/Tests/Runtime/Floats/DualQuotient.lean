/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Dual

/-!
# Nested quotient range regressions

These dyadic witnesses have exact expected coefficients. They exercise both native formats,
including a mixed derivative whose inner contribution underflows before a later multiplication.
The external numerical oracle additionally checks arbitrary coefficient trees against an
independent subset-polynomial recurrence.
-/

@[expose] public section

namespace Tests.Floats.DualQuotient

open TorchLean Runtime.Autograd.Model

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"nested quotient: {label}"

def leaves {α : Type} (x : Dual (Dual (Dual α))) : Array α :=
  #[x.re.re.re, x.re.re.du, x.re.du.re, x.re.du.du,
   x.du.re.re, x.du.re.du, x.du.du.re, x.du.du.du]

def checkScale {α : Type} [Context α] (label : String) (h : α)
    (bits : α → UInt64) : IO Unit := do
  let x : Dual (Dual (Dual α)) :=
    ⟨⟨⟨h, 2 * h⟩, ⟨3 * h, 4 * h⟩⟩, ⟨⟨5 * h, 6 * h⟩, ⟨7 * h, 8 * h⟩⟩⟩
  let y : Dual (Dual (Dual α)) := ⟨⟨⟨h, 0⟩, ⟨0, 0⟩⟩, ⟨⟨0, 0⟩, ⟨0, 0⟩⟩⟩
  let expected : Array α := #[1, 2, 3, 4, 5, 6, 7, 8]
  expect (label ++ "/all coefficients") ((leaves (x / y)).map bits == expected.map bits)
  let identity : Array α := #[1, 0, 0, 0, 0, 0, 0, 0]
  expect (label ++ "/self quotient") ((leaves (x / x)).map bits == identity.map bits)

/-- Compare cold encoders with exact dyadic values derived from IEEE significands and exponents.
Both signs of zero encode as rational zero; infinities and both NaN classes have no encoding. -/
def checkColdEncoders : IO Unit := do
  let finite64 : Array (UInt64 × Rat) :=
    #[(0, 0), (1, 1 / (2 : Rat)^1074),
      (0x000fffffffffffff, ((2 : Rat)^52 - 1) / (2 : Rat)^1074),
      (0x0010000000000000, 1 / (2 : Rat)^1022),
      (0x0010000000000001, ((2 : Rat)^52 + 1) / (2 : Rat)^1074),
      (0x3ff0000000000000, 1), (0x3ff0000000000001, 1 + 1 / (2 : Rat)^52),
      (0x3ff8000000000000, 3 / 2),
      (0x7fefffffffffffff, ((2 : Rat)^53 - 1) * (2 : Rat)^971)]
  for (magnitude, value) in finite64 do
    for negative in #[false, true] do
      let bits := magnitude ||| (if negative then 0x8000000000000000 else 0)
      let encoded := Numeric.QuotientArithmetic.encode (Float.ofBits bits)
      expect s!"binary64/cold encoder/{bits}"
        (encoded.map Numeric.QuotientCoefficients.primal ==
          some (if negative then -value else value))
  for magnitude in (#[0x7ff0000000000000, 0x7ff0000000000001,
      0x7ff8000000000000] : Array UInt64) do
    for sign in (#[0, 0x8000000000000000] : Array UInt64) do
      let bits := magnitude ||| sign
      expect s!"binary64/cold encoder rejects nonfinite/{bits}"
        (Numeric.QuotientArithmetic.encode (Float.ofBits bits)).isNone
  let finite32 : Array (UInt32 × Rat) :=
    #[(0, 0), (1, 1 / (2 : Rat)^149),
      (0x007fffff, ((2 : Rat)^23 - 1) / (2 : Rat)^149),
      (0x00800000, 1 / (2 : Rat)^126),
      (0x00800001, ((2 : Rat)^23 + 1) / (2 : Rat)^149),
      (0x3f800000, 1), (0x3f800001, 1 + 1 / (2 : Rat)^23),
      (0x3fc00000, 3 / 2),
      (0x7f7fffff, ((2 : Rat)^24 - 1) * (2 : Rat)^104)]
  for (magnitude, value) in finite32 do
    for negative in #[false, true] do
      let bits := magnitude ||| (if negative then 0x80000000 else 0)
      let encoded := Numeric.QuotientArithmetic.encode (Float32.ofBits bits)
      expect s!"binary32/cold encoder/{bits}"
        (encoded.map Numeric.QuotientCoefficients.primal ==
          some (if negative then -value else value))
  for magnitude in (#[0x7f800000, 0x7f800001, 0x7fc00000] : Array UInt32) do
    for sign in (#[0, 0x80000000] : Array UInt32) do
      let bits := magnitude ||| sign
      expect s!"binary32/cold encoder rejects nonfinite/{bits}"
        (Numeric.QuotientArithmetic.encode (Float32.ofBits bits)).isNone

/-- Compare all operation flags against an independent bit-encoding reference. -/
def checkOperationFlags {α : Type} (label : String) (values : Array α)
    (bits : α → UInt64) (absMask infinity minNormal : UInt64)
    (ops : Numeric.QuotientArithmetic α) : IO Unit := do
  let magnitude := fun x => bits x &&& absMask
  let range := fun z => magnitude z >= infinity ||
    (magnitude z != 0 && magnitude z < minNormal)
  for x in values do
    for y in values do
      let (sum, addFlags) := ops.addWithFlags x y
      let (difference, subFlags) := ops.subWithFlags x y
      let (product, mulFlags) := ops.mulWithFlags x y
      let (quotient, divFlags) := ops.divWithFlags x y
      expect s!"{label}/add flags/{bits x}/{bits y}" ((0 < addFlags) == range sum)
      expect s!"{label}/sub flags/{bits x}/{bits y}" ((0 < subFlags) == range difference)
      let mulExpected := range product ||
        (magnitude product == 0 && magnitude x != 0 && magnitude y != 0)
      let divExpected := range quotient || magnitude y == 0 ||
        (magnitude quotient == 0 && magnitude x != 0)
      expect s!"{label}/mul flags/{bits x}/{bits y}" ((0 < mulFlags) == mulExpected)
      expect s!"{label}/div flags/{bits x}/{bits y}" ((0 < divFlags) == divExpected)

/-- Check range predicates against independent IEEE boundary encodings. -/
def checkRangePredicates : IO Unit := do
  let boundaries64 : Array UInt64 :=
    #[0, 1, 0x000fffffffffffff, 0x0010000000000000, 0x0010000000000001,
      0x7fefffffffffffff, 0x7ff0000000000000, 0x7ff0000000000001, 0x7ff8000000000000]
  for magnitude in boundaries64 do
    for sign in (#[0, 0x8000000000000000] : Array UInt64) do
      let x := Float.ofBits (magnitude ||| sign)
      expect "binary64/native zero predicate"
        ((x == Float.ofBits 0) == (magnitude == 0))
      let expected := magnitude >= 0x7ff0000000000000 ||
        (magnitude != 0 && magnitude < 0x0010000000000000)
      expect "binary64/native range predicate"
        (Numeric.QuotientArithmetic.Native.rangeLoss x == expected)
  let values64 := boundaries64.flatMap fun magnitude =>
    #[Float.ofBits magnitude, Float.ofBits (magnitude ||| 0x8000000000000000)]
  checkOperationFlags "binary64" values64 Float.toBits
    0x7fffffffffffffff 0x7ff0000000000000 0x0010000000000000 inferInstance
  let boundaries32 : Array UInt32 :=
    #[0, 1, 0x007fffff, 0x00800000, 0x00800001, 0x7f7fffff, 0x7f800000,
      0x7f800001, 0x7fc00000]
  for magnitude in boundaries32 do
    for sign in (#[0, 0x80000000] : Array UInt32) do
      let x := Float32.ofBits (magnitude ||| sign)
      expect "binary32/native zero predicate"
        ((x == Float32.ofBits 0) == (magnitude == 0))
      let expected := magnitude >= 0x7f800000 ||
        (magnitude != 0 && magnitude < 0x00800000)
      expect "binary32/native range predicate"
        (Numeric.QuotientArithmetic.Native.rangeLoss x == expected)

  let values32 := boundaries32.flatMap fun magnitude =>
    #[Float32.ofBits magnitude, Float32.ofBits (magnitude ||| 0x80000000)]
  checkOperationFlags "binary32" values32 (fun x => x.toBits.toUInt64)
    0x7fffffff 0x7f800000 0x00800000 inferInstance

/-- Format-specific witnesses share the same nested quotient and exceptional-value checks. -/
def checkNativeQuotient {α : Type} [Context α] (label : String) (bits : α → UInt64)
    (a b negativeZero nan smallest : α) (expected : Array UInt64)
    (infinityBits : UInt64) : IO Unit := do
  let x : Dual (Dual α) := ⟨⟨a, 0⟩, ⟨0, 0⟩⟩
  let y : Dual (Dual α) := ⟨⟨1, a⟩, ⟨b, 0⟩⟩
  let q := x / y
  expect s!"{label}/rescued mixed coefficient"
    (#[bits q.re.re, bits q.re.du, bits q.du.re, bits q.du.du] == expected)
  let z : Dual α := ⟨negativeZero, b⟩
  let d : Dual α := ⟨b, 0⟩
  let signed := z / d
  expect s!"{label}/native negative zero" (bits signed.re == bits negativeZero)
  expect s!"{label}/negative zero tangent" (signed.du == 1)
  let hidden : Dual (Dual α) := ⟨⟨1, 0⟩, ⟨0, nan⟩⟩
  let unit : Dual (Dual α) := ⟨⟨1, 0⟩, ⟨0, 0⟩⟩
  let invalid := hidden / unit
  expect s!"{label}/hidden NaN remains" (invalid.du.du != invalid.du.du)
  let unbounded : Dual α := ⟨0, 1⟩
  let denominator : Dual α := ⟨smallest, 0⟩
  let infinite := unbounded / denominator
  expect s!"{label}/unrepresentable tangent" (bits infinite.du == infinityBits)

/-- Run exact dyadic witnesses for nested native quotient rules. -/
def run : IO Unit := do
  checkColdEncoders
  checkRangePredicates
  checkScale "binary64/large" (Float.ofBits 0x6bb0000000000000) Float.toBits
  checkScale "binary64/small" (Float.ofBits 0x1430000000000000) Float.toBits
  checkNativeQuotient "binary64" Float.toBits
    (Float.ofBits 0x1a70000000000000) (Float.ofBits 0x7e70000000000000)
    (Float.ofBits 0x8000000000000000) (Float.ofBits 0x7ff8000000000000) (Float.ofBits 1)
    #[0x1a70000000000000, 0x8000000000000000, 0xd8f0000000000000, 0x3380000000000000]
    0x7ff0000000000000
  let bits := fun x : Float32 => x.toBits.toUInt64
  checkScale "binary32/large" (Float32.ofBits 0x71800000) bits
  checkScale "binary32/small" (Float32.ofBits 0x0d800000) bits
  checkNativeQuotient "binary32" bits
    (Float32.ofBits 0x17800000) (Float32.ofBits 0x7b800000)
    (Float32.ofBits 0x80000000) (Float32.ofBits 0x7fc00000) (Float32.ofBits 1)
    #[0x17800000, 0x80000000, 0xd3800000, 0x2c000000] 0x7f800000

end Tests.Floats.DualQuotient
