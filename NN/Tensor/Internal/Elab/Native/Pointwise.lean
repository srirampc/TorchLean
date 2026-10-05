/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Representation.Basic.Pointwise -- shake: keep

/-!
# Certified native pointwise kernels

Packed scalar types can replace the generic output-building loop with one
operation-specific native loop while retaining the ordinary pointwise tensor
definition as their proof-visible semantics. Mixed packed byte/float addition
also fuses scalar promotion into the output loop.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Elab.Impl

/-!
## Reference bodies and their native twins

Each proof-visible kernel has an `@[extern]` twin that calls the reference body.
A `@[csimp]` lemma selects the twin during compilation; its borrow annotations
match the C entry point. Keeping one Lean body prevents the two models from
drifting apart. As with `Array.zipWith`, buffer operations truncate to the
shorter input; tensor callers already have matching lengths by their shapes.
The C implementation's agreement with these models remains an FFI trust boundary.
-/

/-- Proof-visible packed floating-point addition. -/
def floatBufferAdd (left right : FloatArray) : FloatArray :=
  ⟨Array.zipWith (· + ·) left.data right.data⟩

/-- Native packed floating-point addition, compiled to one C loop. -/
@[extern "torchlean_float_array_add"]
def floatBufferAddNative
    (left right : @& FloatArray) : FloatArray :=
  floatBufferAdd left right

/-- Compile packed floating-point addition to one native loop. -/
@[csimp] theorem floatBufferAdd_eq_native :
    @floatBufferAdd = @floatBufferAddNative := rfl

/-- Proof-visible packed floating-point subtraction. -/
def floatBufferSub (left right : FloatArray) : FloatArray :=
  ⟨Array.zipWith (· - ·) left.data right.data⟩

/-- Native packed floating-point subtraction, compiled to one C loop. -/
@[extern "torchlean_float_array_sub"]
def floatBufferSubNative
    (left right : @& FloatArray) : FloatArray :=
  floatBufferSub left right

/-- Compile packed floating-point subtraction to one native loop. -/
@[csimp] theorem floatBufferSub_eq_native :
    @floatBufferSub = @floatBufferSubNative := rfl

/-- Proof-visible packed floating-point multiplication. -/
def floatBufferMul (left right : FloatArray) : FloatArray :=
  ⟨Array.zipWith (· * ·) left.data right.data⟩

/-- Native packed floating-point multiplication, compiled to one C loop. -/
@[extern "torchlean_float_array_mul"]
def floatBufferMulNative
    (left right : @& FloatArray) : FloatArray :=
  floatBufferMul left right

/-- Compile packed floating-point multiplication to one native loop. -/
@[csimp] theorem floatBufferMul_eq_native :
    @floatBufferMul = @floatBufferMulNative := rfl

/-- Proof-visible packed floating-point division. -/
def floatBufferDiv (left right : FloatArray) : FloatArray :=
  ⟨Array.zipWith (· / ·) left.data right.data⟩

/-- Native packed floating-point division, compiled to one C loop. -/
@[extern "torchlean_float_array_div"]
def floatBufferDivNative
    (left right : @& FloatArray) : FloatArray :=
  floatBufferDiv left right

/-- Compile packed floating-point division to one native loop. -/
@[csimp] theorem floatBufferDiv_eq_native :
    @floatBufferDiv = @floatBufferDivNative := rfl

/-- Proof-visible fused byte promotion and floating-point addition. -/
def byteFloatBufferAdd
    (left : ByteArray) (right : FloatArray) : FloatArray :=
  ⟨Array.zipWith (fun x y => x.toFloat + y) left.data right.data⟩

/-- Native fused byte promotion and floating-point addition. -/
@[extern "torchlean_byte_float_array_add"]
def byteFloatBufferAddNative
    (left : @& ByteArray) (right : @& FloatArray) : FloatArray :=
  byteFloatBufferAdd left right

/-- Compile fused byte/float addition to one native loop. -/
@[csimp] theorem byteFloatBufferAdd_eq_native :
    @byteFloatBufferAdd = @byteFloatBufferAddNative := rfl

/-- Proof-visible fused floating-point and promoted-byte addition. -/
def floatByteBufferAdd
    (left : FloatArray) (right : ByteArray) : FloatArray :=
  ⟨Array.zipWith (fun x y => x + y.toFloat) left.data right.data⟩

/-- Native fused floating-point and promoted-byte addition. -/
@[extern "torchlean_float_byte_array_add"]
def floatByteBufferAddNative
    (left : @& FloatArray) (right : @& ByteArray) : FloatArray :=
  floatByteBufferAdd left right

/-- Compile fused float/byte addition to one native loop. -/
@[csimp] theorem floatByteBufferAdd_eq_native :
    @floatByteBufferAdd = @floatByteBufferAddNative := rfl

/--
Add two packed Float tensors in one preallocated native loop.

Lean evaluation uses the ordinary packed-array model. Generated code calls
`torchlean_float_array_add`.
-/
def nativeAdd
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    @Rep Float shape instFloatStorage where
  buffer := floatBufferAdd left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (· + ·) left.data right.data).size = _
    simp

/-- Subtract two packed Float tensors in one preallocated native loop. -/
def nativeSub
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    @Rep Float shape instFloatStorage where
  buffer := floatBufferSub left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (· - ·) left.data right.data).size = _
    simp

/-- Multiply two packed Float tensors in one preallocated native loop. -/
def nativeMul
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    @Rep Float shape instFloatStorage where
  buffer := floatBufferMul left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (· * ·) left.data right.data).size = _
    simp

/-- Divide two packed Float tensors in one preallocated native loop. -/
def nativeDiv
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    @Rep Float shape instFloatStorage where
  buffer := floatBufferDiv left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (· / ·) left.data right.data).size = _
    simp

/--
Promote packed bytes and add them to packed floats in one native output loop.

The proof-visible definition is the canonical promoted pointwise operation.
-/
def nativePromoteLeftAdd
    {shape : Shape}
    (left : @Rep UInt8 shape instUInt8Storage)
    (right : @Rep Float shape instFloatStorage) :
    @Rep Float shape instFloatStorage where
  buffer := byteFloatBufferAdd left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (fun x y => x.toFloat + y) left.data right.data).size = _
    simp

/--
Add packed floats to promoted packed bytes in one native output loop.

Operand order remains explicit so IEEE exceptional behavior matches scalar
`Float` addition exactly.
-/
def nativePromoteRightAdd
    {shape : Shape}
    (left : @Rep Float shape instFloatStorage)
    (right : @Rep UInt8 shape instUInt8Storage) :
    @Rep Float shape instFloatStorage where
  buffer := floatByteBufferAdd left.buffer right.buffer
  size_eq := by
    rw [← instFloatStorage.toArray_size]
    change (Array.zipWith (fun x y => x + y.toFloat) left.data right.data).size = _
    simp

/-- The ordinary array observation of packed addition is pointwise addition. -/
@[simp] theorem nativeAdd_data
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    (nativeAdd left right).data =
      Array.zipWith (· + ·) left.data right.data := by
  change (floatBufferAdd left.buffer right.buffer).data =
    Array.zipWith (· + ·) left.buffer.data right.buffer.data
  rfl

/-- The ordinary array observation of packed subtraction is pointwise subtraction. -/
@[simp] theorem nativeSub_data
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    (nativeSub left right).data =
      Array.zipWith (· - ·) left.data right.data := by
  change (floatBufferSub left.buffer right.buffer).data =
    Array.zipWith (· - ·) left.buffer.data right.buffer.data
  rfl

/-- The ordinary array observation of packed multiplication is pointwise multiplication. -/
@[simp] theorem nativeMul_data
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    (nativeMul left right).data =
      Array.zipWith (· * ·) left.data right.data := by
  change (floatBufferMul left.buffer right.buffer).data =
    Array.zipWith (· * ·) left.buffer.data right.buffer.data
  rfl

/-- The ordinary array observation of packed division is pointwise division. -/
@[simp] theorem nativeDiv_data
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage) :
    (nativeDiv left right).data =
      Array.zipWith (· / ·) left.data right.data := by
  change (floatBufferDiv left.buffer right.buffer).data =
    Array.zipWith (· / ·) left.buffer.data right.buffer.data
  rfl

/-- The ordinary array observation of byte/float addition includes promotion. -/
@[simp] theorem nativePromoteLeftAdd_data
    {shape : Shape}
    (left : @Rep UInt8 shape instUInt8Storage)
    (right : @Rep Float shape instFloatStorage) :
    (nativePromoteLeftAdd left right).data =
      Array.zipWith (fun x y => x.toFloat + y) left.data right.data := by
  change (byteFloatBufferAdd left.buffer right.buffer).data =
    Array.zipWith (fun x y => x.toFloat + y)
      left.buffer.data right.buffer.data
  rfl

/-- The ordinary array observation of float/byte addition includes promotion. -/
@[simp] theorem nativePromoteRightAdd_data
    {shape : Shape}
    (left : @Rep Float shape instFloatStorage)
    (right : @Rep UInt8 shape instUInt8Storage) :
    (nativePromoteRightAdd left right).data =
      Array.zipWith (fun x y => x + y.toFloat) left.data right.data := by
  change (floatByteBufferAdd left.buffer right.buffer).data =
    Array.zipWith (fun x y => x + y.toFloat)
      left.buffer.data right.buffer.data
  rfl

/-- Native packed addition has the canonical coordinate semantics. -/
@[simp, grind =] theorem nativeAdd_apply
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage)
    (coordinate : Coord shape) :
    nativeAdd left right coordinate =
      left coordinate + right coordinate := by
  let index := Coord.linearize coordinate
  change (nativeAdd left right).getFlat index =
    left.getFlat index + right.getFlat index
  rw [← Rep.data_getFlat (nativeAdd left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

/-- Native packed subtraction has the canonical coordinate semantics. -/
@[simp, grind =] theorem nativeSub_apply
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage)
    (coordinate : Coord shape) :
    nativeSub left right coordinate =
      left coordinate - right coordinate := by
  let index := Coord.linearize coordinate
  change (nativeSub left right).getFlat index =
    left.getFlat index - right.getFlat index
  rw [← Rep.data_getFlat (nativeSub left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

/-- Native packed multiplication has the canonical coordinate semantics. -/
@[simp, grind =] theorem nativeMul_apply
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage)
    (coordinate : Coord shape) :
    nativeMul left right coordinate =
      left coordinate * right coordinate := by
  let index := Coord.linearize coordinate
  change (nativeMul left right).getFlat index =
    left.getFlat index * right.getFlat index
  rw [← Rep.data_getFlat (nativeMul left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

/-- Native packed division has the canonical coordinate semantics. -/
@[simp, grind =] theorem nativeDiv_apply
    {shape : Shape}
    (left right : @Rep Float shape instFloatStorage)
    (coordinate : Coord shape) :
    nativeDiv left right coordinate =
      left coordinate / right coordinate := by
  let index := Coord.linearize coordinate
  change (nativeDiv left right).getFlat index =
    left.getFlat index / right.getFlat index
  rw [← Rep.data_getFlat (nativeDiv left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

/-- Fused byte/float addition has the canonical promoted semantics. -/
@[simp, grind =] theorem nativePromoteLeftAdd_apply
    {shape : Shape}
    (left : @Rep UInt8 shape instUInt8Storage)
    (right : @Rep Float shape instFloatStorage)
    (coordinate : Coord shape) :
    nativePromoteLeftAdd left right coordinate =
      (left coordinate).toFloat + right coordinate := by
  let index := Coord.linearize coordinate
  change (nativePromoteLeftAdd left right).getFlat index =
    (left.getFlat index).toFloat + right.getFlat index
  rw [← Rep.data_getFlat (nativePromoteLeftAdd left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

/-- Fused float/byte addition has the canonical promoted semantics. -/
@[simp, grind =] theorem nativePromoteRightAdd_apply
    {shape : Shape}
    (left : @Rep Float shape instFloatStorage)
    (right : @Rep UInt8 shape instUInt8Storage)
    (coordinate : Coord shape) :
    nativePromoteRightAdd left right coordinate =
      left coordinate + (right coordinate).toFloat := by
  let index := Coord.linearize coordinate
  change (nativePromoteRightAdd left right).getFlat index =
    left.getFlat index + (right.getFlat index).toFloat
  rw [← Rep.data_getFlat (nativePromoteRightAdd left right) index,
    ← Rep.data_getFlat left index, ← Rep.data_getFlat right index]
  simp

end TorchLean.Tensor.Internal.Elab.Impl
