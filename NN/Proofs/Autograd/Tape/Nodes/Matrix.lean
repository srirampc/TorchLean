/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Elementwise
public import NN.Proofs.Autograd.Tape.Nodes.Shape

/-!
# Matrix tape nodes

Matrix multiplication, transpose, row/column broadcasting, and row means, with VJP correctness facts
stated at the vectorized tape level.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

-- ---------------------------------------------------------------------------
-- Linear algebra: matrix-matrix multiplication
-- ---------------------------------------------------------------------------

namespace Matmul

open scoped BigOperators

/-- Flattened size of an `m×n` matrix shape: `Spec.Shape.size (.dim m (.dim n .scalar)) = m*n`. -/
abbrev matSize (m n : Nat) : Nat :=
  Spec.Shape.size (.dim m (.dim n .scalar))

/-- Flattened size of a length-`n` vector shape: `Spec.Shape.size (.dim n .scalar) = n`. -/
abbrev vecSize (n : Nat) : Nat :=
  Spec.Shape.size (.dim n .scalar)

@[simp] theorem vecSize_eq (n : Nat) : vecSize n = n := by
  simp [vecSize, Spec.Shape.size]

/-- Convert `(i,j)` coordinates into a flattened index for an `m×n` matrix vectorization. -/
def idxMN {m n : Nat} (i : Fin m) (j : Fin n) : Fin (matSize m n) :=
  let hn : vecSize n = n := vecSize_eq n
  -- `matSize m n` unfolds to `m * vecSize n`, so `finProdFinEquiv` lands in `Fin (matSize m n)`.
  finProdFinEquiv (i, Fin.cast hn.symm j)

/-- The flattened index of entry `(i, j)` is `j + n * i`: row-major layout, as in PyTorch. -/
theorem val_idxMN {m n : Nat} (i : Fin m) (j : Fin n) :
    (idxMN (m := m) (n := n) i j).val = j.val + n * i.val := by
  change j.val + Spec.Shape.size (Shape.dim n Shape.scalar) * i.val = _
  simp [Spec.Shape.size]

/-- Every flattened matrix index is `idxMN` of its row and column. -/
theorem idxMN_divNat_modNat {m n : Nat} (ip : Fin (matSize m n)) :
    idxMN (m := m) (n := n) (ip.divNat (m := m) (n := vecSize n))
      (Fin.cast (vecSize_eq n) (ip.modNat (m := m) (n := vecSize n))) = ip := by
  apply Fin.ext
  change (ip.modNat (m := m) (n := vecSize n)).val +
      vecSize n * (ip.divNat (m := m) (n := vecSize n)).val = ip.val
  exact Nat.mod_add_div _ _

/-- Coordinate `idxMN i j` of a vectorized matrix is the matrix entry `A i j`.

Matrix and attention nodes are proved in the flat `Vec` world, where Mathlib's calculus lives, and
stated about `Tensor ℝ [m, n]`; this lemma connects the two without unfolding the flattening. -/
theorem tensorToVec_idxMN {m n : Nat} (A : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    tensorToVec (t := A) (idxMN (m := m) (n := n) i j) = Spec.get2 A i j := by
  have h1 : 0 < Spec.Shape.size Shape.scalar := by simp [Spec.Shape.size]
  have hrow := tensorToVec_dim_apply (Tensor.unstack A)
    (i, finProdFinEquiv (j, (⟨0, h1⟩ : Fin (Spec.Shape.size Shape.scalar))))
  have hcol := tensorToVec_dim_apply (Tensor.unstack (Tensor.unstack A i))
    (j, (⟨0, h1⟩ : Fin (Spec.Shape.size Shape.scalar)))
  rw [Tensor.dim_unstack] at hrow hcol
  have hidx :
      (finProdFinEquiv (i, finProdFinEquiv (j, (⟨0, h1⟩ : Fin (Spec.Shape.size Shape.scalar)))) :
        Fin (matSize m n)) = idxMN (m := m) (n := n) i j := by
    apply Fin.ext
    rw [val_idxMN]
    change (0 + Spec.Shape.size Shape.scalar * j.val) +
      Spec.Shape.size (Shape.dim n Shape.scalar) * i.val = j.val + n * i.val
    simp [Spec.Shape.size]
  rw [hidx] at hrow
  rw [hrow, hcol, ← Tensor.scalar_item (Tensor.unstack (Tensor.unstack A i) j), tensorToVec_scalar]
  rfl

/-- `Spec.get2` of a `vecToTensor`-constructed matrix reads back the corresponding flattened
entry. -/
private theorem get2_vecToTensor {m n : Nat} (v : Vec (matSize m n)) (i : Fin m) (j : Fin n) :
    Spec.get2 (vecToTensor (s := .dim m (.dim n .scalar)) v) i j =
      v (idxMN (m := m) (n := n) i j) := by
  have htv :
      tensorToVec (t := vecToTensor (s := .dim m (.dim n .scalar)) v)
          (idxMN (m := m) (n := n) i j)
        = v (idxMN (m := m) (n := n) i j) := by
    simp
  exact (tensorToVec_idxMN (A := vecToTensor (s := .dim m (.dim n .scalar)) v) i j).symm.trans htv

/-- A bilinear map on flattened matrices: `(m×n) × (n×p) → (m×p)` on `Vec (Spec.Shape.size ...)`. -/
def matmulVec {m n p : Nat} (a : Vec (matSize m n)) (b : Vec (matSize n p)) : Vec (matSize m p) :=
  vecOfFun (n := matSize m p) fun ip =>
    let hp : vecSize p = p := vecSize_eq p
    let i : Fin m := ip.divNat (m := m) (n := vecSize p)
    let k' : Fin (vecSize p) := ip.modNat (m := m) (n := vecSize p)
    let k : Fin p := Fin.cast hp k'
    ∑ j : Fin n, a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)

/-- Entry `(i, k)` of a matrix product is the usual sum over the contracted index.

The `let`s in the statement are the flat-index arithmetic: a single `Fin (m * p)` is split into a
row by `divNat` and a column by `modNat`. Keeping them in the statement rather than in a side
condition means `simp` can use this lemma on a goal phrased purely in flat coordinates. -/
@[simp] theorem matmulVec_apply {m n p : Nat} (a : Vec (matSize m n)) (b : Vec (matSize n p))
    (ip : Fin (matSize m p)) :
    matmulVec (m := m) (n := n) (p := p) a b ip =
      let hp : vecSize p = p := vecSize_eq p
      let i : Fin m := ip.divNat (m := m) (n := vecSize p)
      let k' : Fin (vecSize p) := ip.modNat (m := m) (n := vecSize p)
      let k : Fin p := Fin.cast hp k'
      ∑ j : Fin n, a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k) := by
  simp [matmulVec]

/-!
Matrix multiplication is developed at the vector level (flattened matrices) to integrate cleanly
with `CtxVec` and the `HasFDerivAt` machinery.

PyTorch analogue: `torch.matmul` / `@` operator on 2D tensors.
https://pytorch.org/docs/stable/generated/torch.matmul.html
-/

/-- For fixed left operand `a`, `matmulCLMRight a` is the linear map `b ↦ a*b`. -/
def matmulCLMRight {m n p : Nat} (a : Vec (matSize m n)) :
    Vec (matSize n p) →L[ℝ] Vec (matSize m p) := by
  classical
  let fLin : Vec (matSize n p) →ₗ[ℝ] Vec (matSize m p) :=
    { toFun := fun b => matmulVec (m := m) (n := n) (p := p) a b
      map_add' := by
        intro b1 b2
        ext ip
        simp [matmulVec, Finset.sum_add_distrib, mul_add]
      map_smul' := by
        intro r b
        ext ip
        simp [matmulVec, Finset.mul_sum, mul_left_comm] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- `matmulCLMRight a` computes `matmulVec a`. -/
@[simp] theorem matmulCLMRight_apply {m n p : Nat} (a : Vec (matSize m n))
    (b : Vec (matSize n p)) :
    matmulCLMRight (m := m) (n := n) (p := p) a b = matmulVec (m := m) (n := n) (p := p) a b :=
  rfl

/-- Continuous bilinear map for matrix multiplication on flattened vectors.

This is `a ↦ matmulCLMRight a`, linear in `a`; continuity is automatic in finite dimension, so no
operator-norm bound is needed. -/
def matmulBilin {m n p : Nat} :
    Vec (matSize m n) →L[ℝ] Vec (matSize n p) →L[ℝ] Vec (matSize m p) := by
  classical
  let fLin : Vec (matSize m n) →ₗ[ℝ] Vec (matSize n p) →L[ℝ] Vec (matSize m p) :=
    { toFun := fun a => matmulCLMRight (m := m) (n := n) (p := p) a
      map_add' := by
        intro a1 a2
        ext b ip
        simp [matmulVec, Finset.sum_add_distrib, add_mul]
      map_smul' := by
        intro r a
        ext b ip
        simp [matmulVec, smul_eq_mul, Finset.mul_sum, mul_assoc] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The bilinear packaging of matrix multiplication computes `matmulVec`.

`matmulBilin` exists only so the differentiability proofs can reuse Mathlib's bilinear-map API; this
lemma is what lets every other proof forget that packaging. -/
@[simp] theorem matmulBilin_apply {m n p : Nat} (a : Vec (matSize m n)) (b : Vec (matSize n p)) :
    matmulBilin (m := m) (n := n) (p := p) a b = matmulVec (m := m) (n := n) (p := p) a b := rfl

/-- `Spec.matMulSpec` agrees with `matmulVec` after flattening both inputs/outputs. -/
theorem forward_eq_matmulVec {m n p : Nat} (aV : Vec (matSize m n)) (bV : Vec (matSize n p)) :
    tensorToVec (t := Spec.matMulSpec (vecToTensor (s := .dim m (.dim n .scalar)) aV)
        (vecToTensor (s := .dim n (.dim p .scalar)) bV))
      =
    matmulVec (m := m) (n := n) (p := p) aV bV := by
  classical
  ext ip
  -- represent `ip` as a row/column pair using `Fin.divNat/modNat` for `m * vecSize p`
  let i : Fin m := ip.divNat (m := m) (n := vecSize p)
  let k' : Fin (vecSize p) := ip.modNat (m := m) (n := vecSize p)
  let hp : vecSize p = p := vecSize_eq p
  let k : Fin p := Fin.cast hp k'
  -- interpret LHS coordinate via `get2` and the matrix entry lemma
  have hL :
      tensorToVec
          (t := Spec.matMulSpec (vecToTensor (s := .dim m (.dim n .scalar)) aV)
            (vecToTensor (s := .dim n (.dim p .scalar)) bV)) ip
        =
      Spec.get2
          (Spec.matMulSpec (vecToTensor (s := .dim m (.dim n .scalar)) aV)
            (vecToTensor (s := .dim n (.dim p .scalar)) bV)) i k := by
    have hip : idxMN (m := m) (n := p) i k = ip := idxMN_divNat_modNat ip
    rw [←hip]
    exact tensorToVec_idxMN
      (A := Spec.matMulSpec (vecToTensor (s := .dim m (.dim n .scalar)) aV)
        (vecToTensor (s := .dim n (.dim p .scalar)) bV))
      i k
  have hEntry :=
    get2_mat_mul_spec
      (A := vecToTensor (s := .dim m (.dim n .scalar)) aV)
      (B := vecToTensor (s := .dim n (.dim p .scalar)) bV)
      (i := i) (j := k)
  have hA : ∀ j : Fin n,
      Spec.get2 (vecToTensor (s := .dim m (.dim n .scalar)) aV) i j =
        aV (idxMN (m := m) (n := n) i j) :=
    fun j => get2_vecToTensor (v := aV) i j
  have hB : ∀ j : Fin n,
      Spec.get2 (vecToTensor (s := .dim n (.dim p .scalar)) bV) j k =
        bV (idxMN (m := n) (n := p) j k) :=
    fun j => get2_vecToTensor (v := bV) j k
  calc
    tensorToVec
        (t := Spec.matMulSpec (vecToTensor (s := .dim m (.dim n .scalar)) aV)
          (vecToTensor (s := .dim n (.dim p .scalar)) bV)) ip
        =
      Spec.get2
          (Spec.matMulSpec (vecToTensor (s := .dim m (.dim n .scalar)) aV)
            (vecToTensor (s := .dim n (.dim p .scalar)) bV)) i k := hL
    _ = ∑ j : Fin n,
          Spec.get2 (vecToTensor (s := .dim m (.dim n .scalar)) aV) i j *
            Spec.get2 (vecToTensor (s := .dim n (.dim p .scalar)) bV) j k := by
          simpa using hEntry
    _ = ∑ j : Fin n, aV (idxMN (m := m) (n := n) i j) * bV (idxMN (m := n) (n := p) j k) := by
          simp [hA, hB]
    _ = matmulVec (m := m) (n := n) (p := p) aV bV ip := by
          rw [matmulVec_apply]

end Matmul

-- ---------------------------------------------------------------------------
-- Linear algebra: matrix transpose
-- ---------------------------------------------------------------------------

namespace MatTranspose

open Matmul

/-- Helper: `matSize m n` is definitionally `m * n`. -/
theorem matSize_eq_mul (m n : Nat) : Matmul.matSize m n = m * n := by
  simp [Matmul.matSize, Spec.Shape.size]

/-- Equivalence implementing matrix transpose on flattened indices. -/
def transposeEquiv (m n : Nat) : Fin (m * n) ≃ Fin (n * m) :=
  (finProdFinEquiv.symm.trans (Equiv.prodComm (Fin m) (Fin n))).trans finProdFinEquiv

/-- The transpose index equivalence is symmetric up to swapping `m` and `n`. -/
private theorem transposeEquiv_symm (m n : Nat) :
    (transposeEquiv m n).symm = transposeEquiv n m := by
  ext k; simp [transposeEquiv, Equiv.prodComm_symm]

/-- Transpose on flattened matrices: `(m×n)` flattened row-major → `(n×m)` flattened row-major. -/
def transposeVec {m n : Nat} (a : Vec (Matmul.matSize m n)) : Vec (Matmul.matSize n m) :=
  castVec (matSize_eq_mul n m).symm
    (ShapeOps.reindexVec (transposeEquiv m n) (castVec (matSize_eq_mul m n) a))

/-- Adjointness of `transposeVec` with respect to the standard inner product on vectors. -/
private theorem inner_transposeVec {m n : Nat} (x : Vec (Matmul.matSize m n))
    (y : Vec (Matmul.matSize n m)) :
    inner ℝ (transposeVec (m := m) (n := n) x) y =
      inner ℝ x (transposeVec (m := n) (n := m) y) := by
  -- Move the size casts across the inner product; what remains is `inner_reindex_left` for
  -- `transposeEquiv m n`, whose inverse is the transpose equivalence with `m` and `n` swapped.
  unfold transposeVec
  rw [inner_castVec_left, ShapeOps.inner_reindex_left, transposeEquiv_symm,
    ← inner_castVec_castVec (h := matSize_eq_mul m n) (x := x)]
  simp

end MatTranspose

/-!
Transpose is implemented as a coordinate permutation on flattened matrices.

PyTorch analogue: `A.transpose(0, 1)` for a 2D tensor.
https://pytorch.org/docs/stable/generated/torch.transpose.html
-/

/-- Tape node computing matrix transpose: `(m×n) ↦ (n×m)`. -/
def matrixTranspose {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) : Node Γ (.dim n (.dim m .scalar)) :=
  Node.ofFn (Γ := Γ) (τ := .dim n (.dim m .scalar))
    (f := fun xV => MatTranspose.transposeVec (m := m) (n := n) (CtxVec.get (Γ := Γ) (s := .dim m
      (.dim n .scalar)) A xV))
    (jvp := fun _xV dxV => MatTranspose.transposeVec (m := m) (n := n) (CtxVec.get (Γ := Γ) (s :=
      .dim m (.dim n .scalar)) A dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A
        (MatTranspose.transposeVec (m := n) (n := m) δV))
    (correct_inner := by
      intro _xV dxV δV
      have hT :=
        MatTranspose.inner_transposeVec (m := m) (n := n)
          (x := CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV) (y := δV)
      have hCtx :=
        CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV
          (MatTranspose.transposeVec (m := n) (n := m) δV)
      calc
        inner ℝ
            (MatTranspose.transposeVec (m := m) (n := n)
              (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)) δV
            =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
              (MatTranspose.transposeVec (m := n) (n := m) δV) := hT
        _ =
            inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A
                (MatTranspose.transposeVec (m := n) (n := m) δV)) := by
            simpa using hCtx.symm)

/-- `NodeFDerivCorrect` for `matrixTranspose` (it is linear/isometric). -/
def matrixTransposeFderiv {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (matrixTranspose (Γ := Γ) (m := m) (n := n) A) := by
  classical
  -- `transposeVec` is a size cast, a coordinate permutation, and a size cast; each is already a
  -- continuous linear map.
  let Tlin : Vec (Matmul.matSize m n) →L[ℝ] Vec (Matmul.matSize n m) :=
    (Graph.castCLM (h := (MatTranspose.matSize_eq_mul n m).symm)).comp
      ((ShapeOps.reindexLin (MatTranspose.transposeEquiv m n)).comp
        (Graph.castCLM (h := MatTranspose.matSize_eq_mul m n)))
  refine
    { deriv := fun _xV => Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hCLM :=
      (Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)).hasFDerivAt (x := xV)
    have hfun :
        (Node.forwardVec (Γ := Γ) (τ := .dim n (.dim m .scalar))
            (matrixTranspose (Γ := Γ) (m := m) (n := n) A)) =
          (fun x : CtxVec Γ =>
            (Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)) x) := by
      funext x
      simp [matrixTranspose, Node.forwardVec_ofFn, ContinuousLinearMap.comp_apply,
        CtxVec.getCLM_apply, Tlin, Graph.castCLM, MatTranspose.transposeVec]
    exact hCLM.congr_of_eventuallyEq hfun.eventuallyEq
  · intro _xV dxV
    simp [matrixTranspose, Node.jvpVec_ofFn, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply,
      Tlin, Graph.castCLM, MatTranspose.transposeVec]

/-- Matrix multiplication node on 2D tensors. -/
def matmul {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    Node Γ (.dim m (.dim p .scalar)) :=
  Node.ofFn (Γ := Γ) (τ := .dim m (.dim p .scalar))
    (f := fun xV =>
      let aT := vecToTensor (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := vecToTensor (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      tensorToVec (t := Spec.matMulSpec aT bT))
    (jvp := fun xV dxV =>
      let aT := vecToTensor (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := vecToTensor (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let daT := vecToTensor (s := .dim m (.dim n .scalar))
        (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
      let dbT := vecToTensor (s := .dim n (.dim p .scalar))
        (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
      tensorToVec (t := addSpec (Spec.matMulSpec daT bT) (Spec.matMulSpec aT dbT)))
    (vjp := fun xV δV =>
      let aT := vecToTensor (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := vecToTensor (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let δT := vecToTensor (s := .dim m (.dim p .scalar)) δV
      let dA := Spec.matMulSpec δT (swapAdjacentAxes bT 0)
      let dB := Spec.matMulSpec (swapAdjacentAxes aT 0) δT
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (tensorToVec (t := dA)) +
        CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (tensorToVec (t := dB)))
    (correct_inner := by
      intro xV dxV δV
      classical
      -- abbreviate tensors
      let aT := vecToTensor (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := vecToTensor (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let daT := vecToTensor (s := .dim m (.dim n .scalar))
        (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
      let dbT := vecToTensor (s := .dim n (.dim p .scalar))
        (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
      let δT := vecToTensor (s := .dim m (.dim p .scalar)) δV
      let dC := addSpec (Spec.matMulSpec daT bT) (Spec.matMulSpec aT dbT)
      let dA := Spec.matMulSpec δT (swapAdjacentAxes bT 0)
      let dB := Spec.matMulSpec (swapAdjacentAxes aT 0) δT

      -- LHS: rewrite `inner` into tensor `dot` using vectorization.
      have hL : inner ℝ (tensorToVec (t := dC)) δV = dot dC δT := by
        simp [dot_eq_inner_tensorToVec, δT, tensorToVec_vecToTensor]

      -- RHS: split the context inner into the two single-slot contributions.
      have hR :
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (tensorToVec (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (tensorToVec (t := dB)))
            =
          dot daT dA + dot dbT dB := by
        have hA' :
            inner ℝ dxV
                (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (tensorToVec (t := dA)))
              =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
              (tensorToVec (t := dA)) := by
          simpa using
            (CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV
              (tensorToVec (t := dA)))
        have hB' :
            inner ℝ dxV
                (CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (tensorToVec (t := dB)))
              =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
              (tensorToVec (t := dB)) := by
          simpa using
            (CtxVec.inner_get_single (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV
              (tensorToVec (t := dB)))
        have hdotA :
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
                (tensorToVec (t := dA)) =
              dot daT dA := by
          simp [dot_eq_inner_tensorToVec, daT, tensorToVec_vecToTensor]
        have hdotB :
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
                (tensorToVec (t := dB)) =
              dot dbT dB := by
          simp [dot_eq_inner_tensorToVec, dbT, tensorToVec_vecToTensor]
        calc
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (tensorToVec (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (tensorToVec (t := dB)))
              =
              inner ℝ dxV
                  (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A
                    (tensorToVec (t := dA))) +
                inner ℝ dxV
                  (CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B
                    (tensorToVec (t := dB))) := by
                simp [inner_add_right]
          _ =
              inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
                  (tensorToVec (t := dA))
                +
                inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
                  (tensorToVec (t := dB)) := by
                simp [hA', hB']
          _ = dot daT dA + dot dbT dB := by
                simp [hdotA, hdotB]

      -- Algebra on dots: unfold `dC` and apply the two matmul adjointness lemmas.
      have hdotC : dot dC δT = dot daT dA + dot dbT dB := by
        have hadd :
            dot dC δT =
              dot (Spec.matMulSpec daT bT) δT + dot (Spec.matMulSpec aT dbT) δT := by
          simpa [dC] using
            (dot_add_left (a := Spec.matMulSpec daT bT) (b := Spec.matMulSpec aT dbT) (c := δT))
        have h1 : dot (Spec.matMulSpec daT bT) δT = dot daT dA := by
          simpa [dA] using (dot_mat_mul_right_adjoint (A := daT) (B := bT) (C := δT))
        have h2 : dot (Spec.matMulSpec aT dbT) δT = dot dbT dB := by
          simpa [dB] using (dot_mat_mul_left_adjoint (A := aT) (B := dbT) (C := δT))
        calc
          dot dC δT
              = dot (Spec.matMulSpec daT bT) δT + dot (Spec.matMulSpec aT dbT) δT := hadd
          _ = dot daT dA + dot dbT dB := by simp [h1, h2]

      -- combine
      calc
        inner ℝ (tensorToVec (t := dC)) δV = dot dC δT := hL
        _ = dot daT dA + dot dbT dB := hdotC
        _ =
            inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (tensorToVec (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (tensorToVec (t := dB))) :=
                  hR.symm
      )

/--
`NodeFDerivCorrect` for the matrix-matrix multiplication node.

This packages the product rule and the dot/adjointness lemmas for `Spec.matMulSpec`.
-/
def matmulFderiv {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    NodeFDerivCorrect (matmul (Γ := Γ) (m := m) (n := n) (p := p) A B) := by
  classical
  let fA : CtxVec Γ → Vec (Matmul.matSize m n) :=
    fun x => CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A x
  let fB : CtxVec Γ → Vec (Matmul.matSize n p) :=
    fun x => CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B x
  let Bmul :
      Vec (Matmul.matSize m n) →L[ℝ] Vec (Matmul.matSize n p) →L[ℝ] Vec (Matmul.matSize m p) :=
    Matmul.matmulBilin (m := m) (n := n) (p := p)

  refine
    { deriv := fun xV =>
        (Bmul.precompR (CtxVec Γ) (fA xV) (CtxVec.getCLM (Γ := Γ) (s := .dim n (.dim p .scalar)) B))
          +
          (Bmul.precompL (CtxVec Γ) (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A) (fB
            xV))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hA :
        HasFDerivAt fA (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A) xV := by
      have h := (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A).hasFDerivAt (x := xV)
      have hfun : fA = fun x => (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A) x := by
        funext x; simp [fA, CtxVec.getCLM_apply]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq
    have hB :
        HasFDerivAt fB (CtxVec.getCLM (Γ := Γ) (s := .dim n (.dim p .scalar)) B) xV := by
      have h := (CtxVec.getCLM (Γ := Γ) (s := .dim n (.dim p .scalar)) B).hasFDerivAt (x := xV)
      have hfun : fB = fun x => (CtxVec.getCLM (Γ := Γ) (s := .dim n (.dim p .scalar)) B) x := by
        funext x; simp [fB, CtxVec.getCLM_apply]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq

    have hbilin :=
      ContinuousLinearMap.hasFDerivAt_of_bilinear (B := Bmul) (hf := hA) (hg := hB)

    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := .dim m (.dim p .scalar)) (matmul (Γ := Γ) (m := m) (n := n)
          (p := p) A B))
          =
        (fun xV : CtxVec Γ => (Bmul (fA xV)) (fB xV)) := by
      funext ctxV
      simp [matmul, Node.forwardVec_ofFn, fA, fB, Bmul, Matmul.forward_eq_matmulVec]
    exact hbilin.congr_of_eventuallyEq hEq.eventuallyEq

  · intro xV dxV
    -- Rewrite the node JVP into the bilinear derivative formula: `tensorToVec` respects matrix
    -- addition, and `tensorToVec (matMulSpec (vecToTensor a) (vecToTensor b))` is `matmulVec a b`.
    ext ip
    -- After expanding, the two bilinear terms may appear in the opposite order.
    simp [matmul, Node.jvpVec_ofFn, fA, fB, Bmul, tensorToVec_addSpec,
      Matmul.forward_eq_matmulVec, ContinuousLinearMap.comp_apply,
      CtxVec.getCLM_apply]
    ring

-- ---------------------------------------------------------------------------
-- Matrix broadcasts and row-wise reductions (linear)
-- ---------------------------------------------------------------------------

namespace MatrixLinear

open Matmul

open scoped BigOperators

/-- Broadcast a vector `v : Vec m` across the last axis to a flattened `(m×n)` matrix. -/
def broadcastRowCLM {m n : Nat} : Vec m →L[ℝ] Vec (matSize m n) := by
  classical
  let fLin : Vec m →ₗ[ℝ] Vec (matSize m n) :=
    { toFun := fun v =>
        vecOfFun (n := matSize m n) fun ip => v (ip.divNat (m := m) (n := vecSize n))
      map_add' := by
        intro v w
        ext ip
        simp [vecOfFun]
      map_smul' := by
        intro a v
        ext ip
        simp [vecOfFun, smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Broadcast a vector `v : Vec n` across the first axis to a flattened `(m×n)` matrix. -/
def broadcastColCLM {m n : Nat} : Vec n →L[ℝ] Vec (matSize m n) := by
  classical
  let hn : vecSize n = n := vecSize_eq n
  let fLin : Vec n →ₗ[ℝ] Vec (matSize m n) :=
    { toFun := fun v =>
        let v' : Vec (vecSize n) := castVec hn.symm v
        vecOfFun (n := matSize m n) fun ip => v' (ip.modNat (m := m) (n := vecSize n))
      map_add' := by
        intro v w
        ext ip
        simp [vecOfFun]
      map_smul' := by
        intro a v
        ext ip
        simp [vecOfFun, smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Row-wise sum: flattened `(m×n)` matrix → vector `m`. -/
def rowSumCLM {m n : Nat} : Vec (matSize m n) →L[ℝ] Vec m := by
  classical
  let fLin : Vec (matSize m n) →ₗ[ℝ] Vec m :=
    { toFun := fun x => vecOfFun (n := m) fun i => ∑ j : Fin (vecSize n), x (finProdFinEquiv (i, j))
      map_add' := by
        intro x y
        apply PiLp.ext
        intro i
        simp only [vecOfFun_ofLp, WithLp.ofLp_add, Pi.add_apply]
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro j _
        rfl
      map_smul' := by
        intro a x
        apply PiLp.ext
        intro i
        simp only [vecOfFun_ofLp, WithLp.ofLp_smul, Pi.smul_apply, RingHom.id_apply,
          smul_eq_mul]
        rw [Finset.mul_sum]
        rfl }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Row-wise mean: flattened `(m×n)` matrix → vector `m`. -/
def rowMeanCLM {m n : Nat} : Vec (matSize m n) →L[ℝ] Vec m :=
  ((1 : ℝ) / (n : ℝ)) • rowSumCLM (m := m) (n := n)

end MatrixLinear

/-- Broadcast a vector `(.dim m .scalar)` across columns to `(.dim m (.dim n .scalar))`. -/
def broadcastRow {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m .scalar)) : Node Γ (.dim m (.dim n .scalar)) :=
  Node.ofFn (Γ := Γ) (τ := .dim m (.dim n .scalar))
    (f := fun xV => MatrixLinear.broadcastRowCLM (m := m) (n := n) (getVec (Γ := Γ) (n := m) idx
      xV))
    (jvp := fun _xV dxV => MatrixLinear.broadcastRowCLM (m := m) (n := n) (getVec (Γ := Γ) (n := m)
      idx dxV))
    (vjp := fun _xV δV =>
      singleVec (Γ := Γ) (n := m) idx ((MatrixLinear.broadcastRowCLM (m := m) (n := n)).adjoint δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        inner_getVec_singleVec (Γ := Γ) (n := m) idx dxV
          ((MatrixLinear.broadcastRowCLM (m := m) (n := n)).adjoint δV)
      have hadj :
          inner ℝ
              (MatrixLinear.broadcastRowCLM (m := m) (n := n) (getVec (Γ := Γ) (n := m) idx dxV))
              δV
            =
          inner ℝ (getVec (Γ := Γ) (n := m) idx dxV)
            ((MatrixLinear.broadcastRowCLM (m := m) (n := n)).adjoint δV) := by
        simpa using
          (ContinuousLinearMap.adjoint_inner_right (A := MatrixLinear.broadcastRowCLM (m := m) (n :=
            n))
            (x := getVec (Γ := Γ) (n := m) idx dxV) (y := δV)).symm
      exact hadj.trans hctx.symm)

/-- `NodeFDerivCorrect` for `broadcastRow` (linear op). -/
def broadcastRowFderiv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m .scalar)) :
    NodeFDerivCorrect (broadcastRow (Γ := Γ) (m := m) (n := n) idx) := by
  classical
  refine
    { deriv := fun _ =>
        (MatrixLinear.broadcastRowCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := m) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Matmul.matSize m n) :=
      (MatrixLinear.broadcastRowCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := m) idx)
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := .dim m (.dim n .scalar))
            (broadcastRow (Γ := Γ) (m := m) (n := n) idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [broadcastRow, D, Node.forwardVec_ofFn, getVecCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    simp [broadcastRow, Node.jvpVec_ofFn, getVecCLM_apply, ContinuousLinearMap.comp_apply]

/-- Broadcast a vector `(.dim n .scalar)` across rows to `(.dim m (.dim n .scalar))`. -/
def broadcastCol {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim n .scalar)) : Node Γ (.dim m (.dim n .scalar)) :=
  Node.ofFn (Γ := Γ) (τ := .dim m (.dim n .scalar))
    (f := fun xV => MatrixLinear.broadcastColCLM (m := m) (n := n) (getVec (Γ := Γ) (n := n) idx
      xV))
    (jvp := fun _xV dxV => MatrixLinear.broadcastColCLM (m := m) (n := n) (getVec (Γ := Γ) (n := n)
      idx dxV))
    (vjp := fun _xV δV =>
      singleVec (Γ := Γ) (n := n) idx ((MatrixLinear.broadcastColCLM (m := m) (n := n)).adjoint δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        inner_getVec_singleVec (Γ := Γ) (n := n) idx dxV
          ((MatrixLinear.broadcastColCLM (m := m) (n := n)).adjoint δV)
      have hadj :
          inner ℝ
              (MatrixLinear.broadcastColCLM (m := m) (n := n) (getVec (Γ := Γ) (n := n) idx dxV))
              δV
            =
          inner ℝ (getVec (Γ := Γ) (n := n) idx dxV)
            ((MatrixLinear.broadcastColCLM (m := m) (n := n)).adjoint δV) := by
        simpa using
          (ContinuousLinearMap.adjoint_inner_right (A := MatrixLinear.broadcastColCLM (m := m) (n :=
            n))
            (x := getVec (Γ := Γ) (n := n) idx dxV) (y := δV)).symm
      exact hadj.trans hctx.symm)

/-- `NodeFDerivCorrect` for `broadcastCol` (linear op). -/
def broadcastColFderiv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim n .scalar)) :
    NodeFDerivCorrect (broadcastCol (Γ := Γ) (m := m) (n := n) idx) := by
  classical
  refine
    { deriv := fun _ =>
        (MatrixLinear.broadcastColCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := n) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Matmul.matSize m n) :=
      (MatrixLinear.broadcastColCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := n) idx)
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := .dim m (.dim n .scalar))
            (broadcastCol (Γ := Γ) (m := m) (n := n) idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [broadcastCol, D, Node.forwardVec_ofFn, getVecCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    simp [broadcastCol, Node.jvpVec_ofFn, getVecCLM_apply, ContinuousLinearMap.comp_apply]

-- ---------------------------------------------------------------------------
-- Shape-preserving reshapes (vector reinterpretation)
-- ---------------------------------------------------------------------------

/-!
Shape-only nodes (`reshape`, `flatten`, and similar) live in `NN.Proofs.Autograd.Tape.Nodes.Shape`
(namespace `TapeNodes.ShapeOps`).
-/

/-- Row-wise mean (reduce last axis): `(.dim m (.dim n .scalar)) → (.dim m .scalar)`. -/
def rowMean {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) : Node Γ (.dim m .scalar) :=
  let outShape : Shape := .dim m .scalar
  let hsz : Spec.Shape.size outShape = m := by simp [outShape, Spec.Shape.size]
  Node.ofFn (Γ := Γ) (τ := outShape)
    (f := fun xV =>
      castVec hsz.symm <|
        MatrixLinear.rowMeanCLM (m := m) (n := n)
          (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx xV))
    (jvp := fun _xV dxV =>
      castVec hsz.symm <|
        MatrixLinear.rowMeanCLM (m := m) (n := n)
          (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV))
    (vjp := fun _xV δV =>
      let δm : Vec m := castVec hsz δV
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) idx
        ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm))
    (correct_inner := by
      intro _xV dxV δV
      classical
      let dxMat : Vec (Matmul.matSize m n) :=
        CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV
      let δm : Vec m := castVec hsz δV
      have hctx :=
        (CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV
          ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm))
      have hLcast :
          inner ℝ (castVec hsz.symm (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat)) δV =
            inner ℝ (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) δm := by
        have hδ : castVec hsz.symm (castVec hsz δV) = δV := by
          simp
        calc
          inner ℝ (castVec hsz.symm (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat)) δV
              =
              inner ℝ (castVec hsz.symm (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat))
                (castVec hsz.symm (castVec hsz δV)) := by
                  simp [hδ]
          _ = inner ℝ (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) (castVec hsz δV) := by
                simpa using
                  (inner_castVec_castVec (h := hsz.symm)
                    (x := MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) (y := castVec hsz δV))
      have hadj :
          inner ℝ (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) δm =
            inner ℝ dxMat ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm) := by
        simpa [δm] using
          (ContinuousLinearMap.adjoint_inner_right (A := MatrixLinear.rowMeanCLM (m := m) (n := n))
            (x := dxMat) (y := δm)).symm
      -- combine
      exact (hLcast.trans hadj).trans hctx.symm)

/-- `NodeFDerivCorrect` for `rowMean` (reduce-mean along the last axis). -/
def rowMeanFderiv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (rowMean (Γ := Γ) (m := m) (n := n) idx) := by
  classical
  let outShape : Shape := .dim m .scalar
  let hsz : Spec.Shape.size outShape = m := by simp [outShape, Spec.Shape.size]
  refine
    { deriv := fun _ =>
        (Graph.castCLM (h := hsz.symm)).comp
          ((MatrixLinear.rowMeanCLM (m := m) (n := n)).comp
            (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) idx))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size outShape) :=
      (Graph.castCLM (h := hsz.symm)).comp
        ((MatrixLinear.rowMeanCLM (m := m) (n := n)).comp
          (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) idx))
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := outShape)
            (rowMean (Γ := Γ) (m := m) (n := n) idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [rowMean, outShape, D, Node.forwardVec_ofFn, CtxVec.getCLM_apply, Graph.castCLM,
        ContinuousLinearMap.comp_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext i
    simp [rowMean, outShape, Node.jvpVec_ofFn, Graph.castCLM, ContinuousLinearMap.comp_apply,
      CtxVec.getCLM_apply]


end TapeNodes

end

end Autograd
end Proofs
