/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA Kernel Coverage: LayerNorm

Compares CPU eager tape vs CUDA eager tape for `layer_norm` (forward + backward).
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace LayerNorm

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

abbrev seqLen : Nat := 2
abbrev embedDim : Nat := 4

theorem seqLen_pos : seqLen > 0 := by decide
theorem embedDim_pos : embedDim > 0 := by decide

def x : Tensor Float [seqLen, embedDim] :=
  (Tensor.from #[
    0.10, 0.20, 0.00, -0.10,
    -0.30, 0.50, 0.20, 0.10
  ]).reshape [seqLen, embedDim] (by dsimp; decide)

def gamma : Tensor Float [embedDim] :=
  (Tensor.from #[1.0, 0.9, 1.1, 1.0]).reshape [embedDim] (by dsimp; decide)

def beta : Tensor Float [embedDim] :=
  (Tensor.from #[0.0, 0.1, -0.1, 0.0]).reshape [embedDim] (by dsimp; decide)

def Internal.checkCase {seqLen embedDim : Nat}
    (seqLen_pos : seqLen > 0) (embedDim_pos : embedDim > 0)
    (label : String) (x : Tensor Float [seqLen, embedDim])
    (gamma beta : Tensor Float [embedDim])
    (seed : Tensor Float [seqLen, embedDim]) : IO Unit := do
  IO.println s!"== layer_norm {label} =="

  let outShape : Shape := [seqLen, embedDim]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "x")
  let (t2, gId) := Tape.leaf (t := t1) gamma (name := some "gamma")
  let (t3, bId) := Tape.leaf (t := t2) beta (name := some "beta")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.layerNorm (α := Float) (t := t3) (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := seqLen_pos) (h_embed_pos := embedDim_pos) xId gId bId)
  let yCpu ← Utils.cpuValue (s := outShape) t4 yId
  let seedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor seed
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := outShape) gradsCpu xId
  let dGammaCpu ← Utils.cpuGrad (s := [embedDim]) gradsCpu gId
  let dBetaCpu ← Utils.cpuGrad (s := [embedDim]) gradsCpu bId

  -- CUDA tape
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "x")
  let (t2c, gIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer gamma)
    (name := some "gamma")
  let (t3c, bIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer beta)
    (name := some "beta")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.layerNorm (t := t3c) (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := seqLen_pos) (h_embed_pos := embedDim_pos) xIdc gIdc bIdc)
  let yCuda ← Utils.cudaValue (s := outShape) t4c yIdc
  let seedCuda := Utils.tensorToAnyBuffer seed
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := outShape) gradsCuda xIdc
  let dGammaCuda ← Utils.cudaGrad (s := [embedDim]) gradsCuda gIdc
  let dBetaCuda ← Utils.cudaGrad (s := [embedDim]) gradsCuda bIdc

  Utils.assertTensorApprox (s := outShape) "layer_norm forward" yCuda yCpu (tol := 3e-3)
  Utils.assertTensorApprox (s := outShape) "layer_norm dx" dxCuda dxCpu (tol := 3e-3)
  Utils.assertTensorApprox (s := [embedDim]) "layer_norm dgamma" dGammaCuda dGammaCpu (tol := 3e-3)
  Utils.assertTensorApprox (s := [embedDim]) "layer_norm dbeta" dBetaCuda dBetaCpu (tol := 3e-3)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: layer_norm ==="
  Internal.checkCase seqLen_pos embedDim_pos "ordinary" x gamma beta
    (Tensor.full [seqLen, embedDim] 1.0)
  let seed : Tensor Float [2, 4] :=
    (Tensor.from #[1.0, -2.0, 0.5, 3.0, 2.0, -1.0, 4.0, -0.5]).reshape [2, 4] (by decide)
  let constant : Tensor Float [2, 4] := Tensor.full [2, 4] 0.25
  Internal.checkCase (by decide) (by decide) "constant variance" constant gamma beta seed
  let nearConstant : Tensor Float [2, 4] :=
    (Tensor.from #[0.25, 0.250001, 0.249999, 0.25, 0.5, 0.500001, 0.499999, 0.5]).reshape
      [2, 4] (by decide)
  Internal.checkCase (by decide) (by decide) "near-zero variance" nearConstant gamma beta seed
  let wide : Tensor Float [2, 8] :=
    (Tensor.from #[0.0, 0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75,
      2.0, -2.25, 2.5, -2.75, 3.0, -3.25, 3.5, -3.75]).reshape [2, 8] (by decide)
  let wideSeed : Tensor Float [2, 8] :=
    (Tensor.from #[1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25, 4.0,
      -1.0, 0.5, 2.0, 3.0, -2.0, 1.0, 4.0, -0.25]).reshape [2, 8] (by decide)
  Internal.checkCase (by decide) (by decide) "eight features" wide
    (Tensor.full [8] 1.0) (Tensor.full [8] 0.0) wideSeed

end LayerNorm
end Cuda
end Tests
