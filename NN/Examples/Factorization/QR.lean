/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common

/-!
# QR Factorization

Check whether `Tensor.qr A` reconstructs $A$ as $QR$ and gives orthonormal columns in $Q$.
The checks use `Float` with an explicit tolerance. Reduced QR uses
`Q : Tensor Float [rows, min rows columns]` and
`R : Tensor Float [min rows columns, columns]`.

Run `scripts/lake.sh exe torchlean factorizations`.
The rank-deficient negative control still reconstructs
the input, but fails orthonormality: reconstructing a matrix alone does not establish both QR
properties. This file tests the implementation; it does not prove a real-arithmetic QR theorem.
-/

@[expose] public section

namespace NN.Examples.Factorization.QR

open TorchLean
open TorchLean.Tensor

/-- Full-rank matrix used for the positive QR check. -/
def fullRankMatrix : Tensor Float [3, 3] :=
  [[12, -51, 4],
   [6, 167, -68],
   [-4, 24, -41]]

/-- Compute both factors once. -/
def fullRankFactors := Tensor.qr fullRankMatrix

/-- Reconstruction error $\lVert A-QR\rVert_{\max}$ for any reduced factor shape. -/
def reconstructionError {rows columns : Nat}
    (matrix : Tensor Float [rows, columns]) (factors : Tensor.QRFactors Float rows columns) :
    Float :=
  Tensor.maxAbsDiff matrix <|
    einsum factors.q, factors.r "row contracted, contracted column -> row column"

/-- Orthonormality error $\lVert Q^\mathsf{T}Q-I\rVert_{\max}$. -/
def orthonormalityError {rows columns : Nat}
    (factors : Tensor.QRFactors Float rows columns) : Float :=
  let qtq : Tensor Float [min rows columns, min rows columns] :=
    einsum factors.q, factors.q "contracted row, contracted column -> row column"
  Tensor.maxAbsDiff qtq (Tensor.identity (min rows columns))

/-! ## Wide Matrix

This case also places a dependent column before a later independent column. A reduced
implementation that merely truncates the old square factors loses that later basis direction.
-/

def wideMatrix : Tensor Float [2, 3] :=
  [[1, 2, 0],
   [0, 0, 1]]

/-- Reduced QR of a wide matrix: `Q` is `2 x 2` and `R` is `2 x 3`. -/
def wideFactors := Tensor.qr wideMatrix

/-! ## Negative Control

The following square matrix has one dependent column. This implementation leaves a zero column in
`Q` for the missing basis direction. Gram-Schmidt still reconstructs the matrix, but
$Q^\mathsf{T}Q\ne I$ for the returned factors.
-/

/-- A matrix whose second column is twice its first. -/
def rankDeficientMatrix : Tensor Float [3, 3] :=
  [[1, 2, 0],
   [2, 4, 1],
   [1, 2, 0]]

/--
QR of a rank-deficient matrix. Reconstruction survives; orthonormality does not, because the
dependent column contributes a zero column to `Q`.
-/
def rankDeficientFactors := Tensor.qr rankDeficientMatrix

/-- Run square and wide full-rank checks plus the dependent-column negative control. -/
def check : IO Unit := do
  assertBelow "QR A = Q·R" (reconstructionError fullRankMatrix fullRankFactors)
  assertBelow "QR Qᵀ·Q = I" (orthonormalityError fullRankFactors)
  assertBelow "QR(wide) A = Q·R" (reconstructionError wideMatrix wideFactors)
  assertBelow "QR(wide) Qᵀ·Q = I" (orthonormalityError wideFactors)
  assertBelow "QR(rank-deficient) A = Q·R still reconstructs"
    (reconstructionError rankDeficientMatrix rankDeficientFactors)
  assertAtLeast "QR(rank-deficient) Qᵀ·Q = I correctly fails (zero basis column)"
    (orthonormalityError rankDeficientFactors)

end NN.Examples.Factorization.QR
