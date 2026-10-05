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
# CUDA Kernel Coverage: Convolution and Pooling

Compares the arbitrary-spatial-axis CPU and CUDA convolution and pooling paths. Two- and three-axis
fixtures exercise the same public operators with different geometry tensors.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace ConvPool

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd

/-- Input channel count used by the convolution and pooling CUDA coverage cases. -/
abbrev inC : Nat := 1
abbrev outC : Nat := 1
abbrev kH : Nat := 2
abbrev kW : Nat := 2
abbrev stride : Nat := 1
abbrev padding : Nat := 0
abbrev inH : Nat := 3
abbrev inW : Nat := 3

abbrev d2 : Nat := 2

def inSpatial2 : TorchLean.Tensor Nat [d2] := [inH, inW]
def kernel2 : TorchLean.Tensor Nat [d2] := [kH, kW]
def stride2 : TorchLean.Tensor Nat [d2] := [stride, stride]
def padding2 : TorchLean.Tensor Nat [d2] := [padding, padding]

def outH : Nat := Spec.Shape.slidingWindowOutDim inH kH stride padding
def outW : Nat := Spec.Shape.slidingWindowOutDim inW kW stride padding

def kernel : Tensor Float [outC, inC, kH, kW] :=
  (Tensor.from #[0.2, -0.1, 0.3, 0.4]).reshape [outC, inC, kH, kW] (by dsimp; decide)

def bias : Tensor Float [outC] :=
  (Tensor.from #[0.05]).reshape [outC] (by dsimp; decide)

def input : Tensor Float [inC, inH, inW] :=
  (Tensor.from #[
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0
  ]).reshape [inC, inH, inW] (by dsimp; decide)

/-!
## Higher-rank runtime cases ($d=3$)

These exercise the new "ND" ConvPool CUDA entrypoints
(`conv`/`max_pool`/`avg_pool`/`smooth_max_pool`) which accept per-axis parameters.
-/

abbrev d3 : Nat := 3
abbrev inD0 : Nat := 3
abbrev inD1 : Nat := 3
abbrev inD2 : Nat := 3

abbrev k0 : Nat := 2
abbrev k1 : Nat := 2
abbrev k2 : Nat := 2

def inSpatial3 : TorchLean.Tensor Nat [d3] :=
  [inD0, inD1, inD2]

def kernel3V : TorchLean.Tensor Nat [d3] :=
  [k0, k1, k2]

def stride3V : TorchLean.Tensor Nat [d3] :=
  [1, 1, 1]

def padding3V : TorchLean.Tensor Nat [d3] :=
  [0, 0, 0]

def outSpatial3 : TorchLean.Tensor Nat [d3] :=
  Spec.convOutSpatial inSpatial3 kernel3V stride3V padding3V

def outShape3 : Shape :=
  Shape.ofList (outC :: Tensor.to outSpatial3 (List Nat))

def kernel3 : Tensor Float
    (Shape.ofList (outC :: inC :: Tensor.to kernel3V (List Nat))) :=
  (Tensor.from #[
    0.2, -0.1,
    0.3, 0.4,
    -0.25, 0.15,
    0.05, -0.35
  ]).reshape [outC, inC, k0, k1, k2] (by dsimp; decide)

def input3 : Tensor Float (Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) :=
  (Tensor.from #[
    1.0,  2.0,  3.0,
    4.0,  5.0,  6.0,
    7.0,  8.0,  9.0,

    10.0, 11.0, 12.0,
    13.0, 14.0, 15.0,
    16.0, 17.0, 18.0,

    19.0, 20.0, 21.0,
    22.0, 23.0, 24.0,
    25.0, 26.0, 27.0
  ]).reshape [inC, inD0, inD1, inD2] (by dsimp; decide)

def runConv3 : IO Unit := do
  IO.println "== conv (d=3) =="

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, kId) := Tape.leaf (t := t0) kernel3 (name := some "kernel")
  let (t2, bId) := Tape.leaf (t := t1) bias (name := some "bias")
  let (t3, xId) := Tape.leaf (t := t2) input3 (name := some "input")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.conv (α := Float) (t := t3)
      (d := d3) (inC := inC) (outC := outC)
      (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (inSpatial := inSpatial3)
      kId bId xId (name := "conv[d=3]"))
  let yCpu ← Utils.cpuValue (s := outShape3) t4 yId
  let seedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full outShape3 (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dKCpu ← Utils.cpuGrad
    (s := Shape.ofList (outC :: inC :: Tensor.to kernel3V (List Nat))) gradsCpu kId
  let dBCpu ← Utils.cpuGrad (s := [outC]) gradsCpu bId
  let dXCpu ← Utils.cpuGrad
    (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, kIdc) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer kernel3)
      (name := some "kernel")
  let (t2c, bIdc) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer bias) (name :=
      some "bias")
  let (t3c, xIdc) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer input3)
      (name := some "input")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.conv (t := t3c)
      (d := d3) (inC := inC) (outC := outC)
      (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      (inSpatial := inSpatial3)
      kIdc bIdc xIdc)
  let yCuda ← Utils.cudaValue (s := outShape3) t4c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := outShape3
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size outShape3)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dKCuda ← Utils.cudaGrad
    (s := Shape.ofList (outC :: inC :: Tensor.to kernel3V (List Nat))) gradsCuda kIdc
  let dBCuda ← Utils.cudaGrad (s := [outC]) gradsCuda bIdc
  let dXCuda ← Utils.cudaGrad
    (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) gradsCuda xIdc

  Utils.assertTensorApprox (s := outShape3) "conv[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox
    (s := Shape.ofList (outC :: inC :: Tensor.to kernel3V (List Nat)))
    "conv[d=3] dKernel" dKCuda dKCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := [outC]) "conv[d=3] dBias" dBCuda dBCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat)))
    "conv[d=3] dInput" dXCuda dXCpu (tol := 1e-2)

def runMaxPool3 : IO Unit := do
  IO.println "== max_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: Tensor.to outSpatial3 (List Nat))

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad
    (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.maxPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad
    (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat)))
    "max_pool[d=3] dx" dxCuda dxCpu (tol := 1e-6)

/-- Max pooling must retain a valid negative infinity and route its gradient to the first winner. -/
def runMaxPoolNegativeInfinity : IO Unit := do
  IO.println "== max_pool negative infinity =="
  let spatial : TorchLean.Tensor Nat [1] := [2]
  let window : TorchLean.Tensor Nat [1] := [2]
  let unitStride : TorchLean.Tensor Nat [1] := [1]
  let noPadding : TorchLean.Tensor Nat [1] := [0]
  let negInf : Float := (-1.0) / 0.0
  let x : Tensor Float [1, 2] := (Tensor.from #[negInf, negInf]).reshape [1, 2] (by dsimp; decide)
  let outputShape : Shape := Shape.ofList [1, 1]
  let inputShape : Shape := Shape.ofList [1, 2]

  let cpu0 : Tape Float := Tape.empty
  let (cpu1, xCpu) := cpu0.leaf x
  let (cpu2, yCpuId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := cpu1) (d := 1) (C := 1)
      (inSpatial := spatial) (kernel := window) (stride := unitStride) (padding := noPadding)
      xCpu)
  let yCpu ← Utils.cpuValue (s := outputShape) cpu2 yCpuId
  let cpuSeed : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor ((Tensor.from #[1.0]).reshape [1, 1] (by dsimp; decide))
  let cpuGrads ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) cpu2 yCpuId cpuSeed)
  let dxCpu ← Utils.cpuGrad (s := inputShape) cpuGrads xCpu

  let (cuda1, xCuda) := Runtime.Autograd.LibTorch.Tape.empty.leaf (Utils.tensorToAnyBuffer x)
  let (cuda2, yCudaId) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.maxPool (t := cuda1) (d := 1) (C := 1)
      (inSpatial := spatial) (kernel := window) (stride := unitStride) (padding := noPadding)
      xCuda)
  let yCuda ← Utils.cudaValue (s := outputShape) cuda2 yCudaId
  let cudaSeed : Runtime.Autograd.LibTorch.AnyBuffer :=
    Utils.tensorToAnyBuffer ((Tensor.from #[1.0]).reshape [1, 1] (by dsimp; decide))
  let cudaGrads ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll cuda2 yCudaId cudaSeed)
  let dxCuda ← Utils.cudaGrad (s := inputShape) cudaGrads xCuda

  let yCpuFlat := Runtime.Autograd.LibTorch.Convert.flattenFloat yCpu
  let yCudaFlat := Runtime.Autograd.LibTorch.Convert.flattenFloat yCuda
  unless yCpuFlat[0]! == negInf && yCudaFlat[0]! == negInf do
    throw <| IO.userError "max_pool must preserve a valid negative-infinity winner"
  let expectedDx : Tensor Float inputShape :=
    (Tensor.from #[1.0, 0.0]).reshape [1, 2] (by dsimp; decide)
  Utils.assertTensorApprox "max_pool negative-infinity CPU gradient" dxCpu expectedDx
  Utils.assertTensorApprox "max_pool negative-infinity CUDA gradient" dxCuda expectedDx

def runSmoothMaxPool3 : IO Unit := do
  IO.println "== smooth_max_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: Tensor.to outSpatial3 (List Nat))
  let beta : Float := 0.5

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad
    (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.smoothMaxPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad
    (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "smooth_max_pool[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat)))
    "smooth_max_pool[d=3] dx" dxCuda dxCpu (tol := 1e-2)

def runAvgPool3 : IO Unit := do
  IO.println "== avg_pool (d=3) =="

  let outSpatial3 := Spec.poolOutSpatialPad inSpatial3 kernel3V stride3V padding3V
  let yShape : Shape := Shape.ofList (inC :: Tensor.to outSpatial3 (List Nat))

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input3 (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.avgPool (α := Float) (t := t1)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad
    (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input3)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.avgPool (t := t1c)
      (d := d3) (C := inC)
      (inSpatial := inSpatial3) (kernel := kernel3V) (stride := stride3V) (padding := padding3V)
      xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad
    (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat))) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "avg_pool[d=3] forward" yCuda yCpu (tol := 1e-2)
  Utils.assertTensorApprox (s := Shape.ofList (inC :: Tensor.to inSpatial3 (List Nat)))
    "avg_pool[d=3] dx" dxCuda dxCpu (tol := 1e-2)

def runConv : IO Unit := do
  IO.println "== conv (d=2) =="

  let yShape : Shape := [outC, outH, outW]

  -- CPU tape
  let t0 : Tape Float := Tape.empty
  let (t1, kId) := Tape.leaf (t := t0) kernel (name := some "kernel")
  let (t2, bId) := Tape.leaf (t := t1) bias (name := some "bias")
  let (t3, xId) := Tape.leaf (t := t2) input (name := some "input")
  let (t4, yId) ← Utils.okOrThrow
    (Tape.conv (α := Float) (t := t3)
      (d := d2) (inC := inC) (outC := outC) (kernel := kernel2)
      (stride := stride2) (padding := padding2) (inSpatial := inSpatial2)
      kId bId xId)
  let yCpu ← Utils.cpuValue (s := yShape) t4 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t4) yId seedCpu)
  let dKCpu ← Utils.cpuGrad (s := [outC, inC, kH, kW]) gradsCpu kId
  let dBCpu ← Utils.cpuGrad (s := [outC]) gradsCpu bId
  let dXCpu ← Utils.cpuGrad (s := [inC, inH, inW]) gradsCpu xId

  -- CUDA tape
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, kIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer kernel)
    (name := some "kernel")
  let (t2c, bIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer bias)
    (name := some "bias")
  let (t3c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t2c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t4c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.conv (t := t3c)
      (d := d2) (inC := inC) (outC := outC) (kernel := kernel2)
      (stride := stride2) (padding := padding2) (inSpatial := inSpatial2)
      kIdc bIdc xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t4c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape,
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t4c) yIdc seedCuda)
  let dKCuda ← Utils.cudaGrad (s := [outC, inC, kH, kW]) gradsCuda kIdc
  let dBCuda ← Utils.cudaGrad (s := [outC]) gradsCuda bIdc
  let dXCuda ← Utils.cudaGrad (s := [inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "conv forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := [outC, inC, kH, kW])
    "conv dKernel" dKCuda dKCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := [outC]) "conv dBias" dBCuda dBCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := [inC, inH, inW])
    "conv dInput" dXCuda dXCpu (tol := 5e-3)

def runMaxPool : IO Unit := do
  IO.println "== max_pool (d=2) =="
  let yShape : Shape := [inC, outH, outW]

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := t1)
      (d := d2) (C := inC) (inSpatial := inSpatial2) (kernel := kernel2)
      (stride := stride2) (padding := padding2) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := [inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.maxPool (t := t1c)
      (d := d2) (C := inC) (inSpatial := inSpatial2) (kernel := kernel2)
      (stride := stride2) (padding := padding2) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape,
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := [inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool forward" yCuda yCpu (tol := 1e-6)
  Utils.assertTensorApprox (s := [inC, inH, inW]) "max_pool dx" dxCuda dxCpu (tol := 1e-6)

def runMaxPoolPadNegative : IO Unit := do
  IO.println "== max_pool padding negative inputs =="

  let inSpatial : TorchLean.Tensor Nat [2] := [1, 1]
  let kernel : TorchLean.Tensor Nat [2] := [2, 2]
  let stride : TorchLean.Tensor Nat [2] := [1, 1]
  let padding : TorchLean.Tensor Nat [2] := [1, 1]
  let x : Tensor Float [1, 1, 1] :=
    (Tensor.from #[-3.0]).reshape [1, 1, 1] (by dsimp; decide)
  let yShape : Shape := [1, 2, 2]
  let expectedY : Tensor Float [1, 2, 2] :=
    (Tensor.from #[-3.0, -3.0, -3.0, -3.0]).reshape [1, 2, 2] (by dsimp; decide)
  let expectedDx : Tensor Float [1, 1, 1] :=
    (Tensor.from #[4.0]).reshape [1, 1, 1] (by dsimp; decide)

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := t1)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := [1, 1, 1]) gradsCpu xId

  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.maxPool (t := t1c)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape,
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := [1, 1, 1]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool negative CPU expected" yCpu expectedY
    (tol := 1e-6)
  Utils.assertTensorApprox (s := yShape) "max_pool negative CUDA expected" yCuda expectedY
    (tol := 1e-6)
  Utils.assertTensorApprox (s := [1, 1, 1])
    "max_pool negative CPU dx" dxCpu expectedDx (tol := 1e-6)
  Utils.assertTensorApprox (s := [1, 1, 1])
    "max_pool negative CUDA dx" dxCuda expectedDx (tol := 1e-6)

def runMaxPool3PadNegative : IO Unit := do
  IO.println "== max_pool (d=3) padding negative inputs =="

  let inSpatial : TorchLean.Tensor Nat [3] := [1, 1, 1]
  let kernel : TorchLean.Tensor Nat [3] := [2, 2, 2]
  let stride : TorchLean.Tensor Nat [3] := [1, 1, 1]
  let padding : TorchLean.Tensor Nat [3] := [1, 1, 1]
  let yShape : Shape := Shape.ofList [1, 2, 2, 2]
  let x : Tensor Float [1, 1, 1, 1] :=
    (Tensor.from #[-3.0]).reshape [1, 1, 1, 1] (by dsimp; decide)
  let expectedY : Tensor Float [1, 2, 2, 2] :=
    (Tensor.from #[-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0]).reshape [1, 2, 2, 2]
      (by dsimp; decide)
  let expectedDx : Tensor Float [1, 1, 1, 1] :=
    (Tensor.from #[8.0]).reshape [1, 1, 1, 1] (by dsimp; decide)

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.maxPool (α := Float) (t := t1)
      (d := 3) (C := 1)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList [1, 1, 1, 1]) gradsCpu xId

  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.maxPool (t := t1c)
      (d := 3) (C := 1)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape,
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList [1, 1, 1, 1]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] pad negative CPU expected" yCpu expectedY
    (tol := 1e-6)
  Utils.assertTensorApprox (s := yShape) "max_pool[d=3] pad negative CUDA expected" yCuda expectedY
    (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList [1, 1, 1, 1])
    "max_pool[d=3] pad negative CPU dx" dxCpu expectedDx (tol := 1e-6)
  Utils.assertTensorApprox (s := Shape.ofList [1, 1, 1, 1])
    "max_pool[d=3] pad negative CUDA dx" dxCuda expectedDx (tol := 1e-6)

def runSmoothMaxPool : IO Unit := do
  IO.println "== smooth_max_pool (d=2) =="
  let yShape : Shape := [inC, outH, outW]
  let beta : Float := 0.5

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := d2) (C := inC) (inSpatial := inSpatial2) (kernel := kernel2)
      (stride := stride2) (padding := padding2) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := [inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.smoothMaxPool (t := t1c)
      (d := d2) (C := inC) (inSpatial := inSpatial2) (kernel := kernel2)
      (stride := stride2) (padding := padding2) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape,
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := [inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "smooth_max_pool forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := [inC, inH, inW])
    "smooth_max_pool dx" dxCuda dxCpu (tol := 5e-3)

/-- Require a CUDA/runtime boundary operation to reject invalid parameters. -/
def expectCudaResultError {α : Type} (label : String) : Except String α → IO Unit
  | .error _ => pure ()
  | .ok _ => throw <| IO.userError s!"{label}: expected rejection"

/-- Check the stable smooth-max formula at scales where $\beta x$ overflows FP32. -/
def runSmoothMaxPoolStabilityCase (beta expectedSign : Float)
    (expectedDx : Tensor Float [1, 1, 2]) : IO Unit := do
  let inSpatial : TorchLean.Tensor Nat [2] := [1, 2]
  let kernel : TorchLean.Tensor Nat [2] := [1, 2]
  let stride : TorchLean.Tensor Nat [2] := [1, 1]
  let padding : TorchLean.Tensor Nat [2] := [0, 0]
  let x : Tensor Float [1, 1, 2] :=
    (Tensor.from #[1e20, -1e20]).reshape [1, 1, 2] (by dsimp; decide)
  let yShape : Shape := [1, 1, 1]

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let gradsCpu ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t2) yId
      (Spec.SomeTensor.ofTensor (Tensor.full yShape 1.0)))
  let dxCpu ← Utils.cpuGrad (s := [1, 1, 2]) gradsCpu xId

  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.smoothMaxPool (t := t1c)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.LibTorch.Buffer.full 1 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := [1, 1, 2]) gradsCuda xIdc

  let yCpuScalar := (Runtime.Autograd.LibTorch.Convert.flattenFloat yCpu).get! 0
  let yCudaScalar := (Runtime.Autograd.LibTorch.Convert.flattenFloat yCuda).get! 0
  Utils.assertApprox "smooth_max_pool large CPU" (yCpuScalar / 1e20) expectedSign 1e-5
  Utils.assertApprox "smooth_max_pool large CUDA" (yCudaScalar / 1e20) expectedSign 1e-5
  Utils.assertTensorApprox "smooth_max_pool large CPU gradient" dxCpu expectedDx 1e-5
  Utils.assertTensorApprox "smooth_max_pool large CUDA gradient" dxCuda expectedDx 1e-5

/-- Check the spatial smooth-max kernel and reference path under the same overflow pressure. -/
def runSpatialSmoothMaxPoolStabilityCase (beta expectedSign : Float)
    (expectedDx : Tensor Float [1, 2]) : IO Unit := do
  let inSpatial : TorchLean.Tensor Nat [1] := [2]
  let kernel : TorchLean.Tensor Nat [1] := [2]
  let stride : TorchLean.Tensor Nat [1] := [1]
  let padding : TorchLean.Tensor Nat [1] := [0]
  let x : Tensor Float [1, 2] :=
    (Tensor.from #[1e20, -1e20]).reshape [1, 2] (by dsimp; decide)
  let yShape : Shape := Shape.ofList [1, 1]

  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x
  let (t2, yId) ← Utils.okOrThrow
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := 1) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xId beta)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let gradsCpu ← Utils.okOrThrow
    (Tape.backwardDenseAll (α := Float) (t := t2) yId
      (Spec.SomeTensor.ofTensor (Tensor.full yShape 1.0)))
  let dxCpu ← Utils.cpuGrad (s := Shape.ofList [1, 2]) gradsCpu xId

  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x)
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.smoothMaxPool (t := t1c)
      (d := 1) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xIdc beta)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape, buf := Runtime.Autograd.LibTorch.Buffer.full 1 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := Shape.ofList [1, 2]) gradsCuda xIdc

  let yCpuScalar := (Runtime.Autograd.LibTorch.Convert.flattenFloat yCpu).get! 0
  let yCudaScalar := (Runtime.Autograd.LibTorch.Convert.flattenFloat yCuda).get! 0
  Utils.assertApprox "smooth_max_pool spatial large CPU" (yCpuScalar / 1e20) expectedSign 1e-5
  Utils.assertApprox "smooth_max_pool spatial large CUDA" (yCudaScalar / 1e20) expectedSign 1e-5
  Utils.assertTensorApprox "smooth_max_pool spatial large CPU gradient" dxCpu expectedDx 1e-5
  Utils.assertTensorApprox "smooth_max_pool spatial large CUDA gradient" dxCuda expectedDx 1e-5

/-- Stable large-magnitude behavior for positive and negative inverse temperatures. -/
def runSmoothMaxPoolStability : IO Unit := do
  IO.println "== smooth max pooling stability =="
  let maxDx2d : Tensor Float [1, 1, 2] :=
    (Tensor.from #[1.0, 0.0]).reshape [1, 1, 2] (by dsimp; decide)
  let minDx2d : Tensor Float [1, 1, 2] :=
    (Tensor.from #[0.0, 1.0]).reshape [1, 1, 2] (by dsimp; decide)
  let maxDxSpatial : Tensor Float [1, 2] :=
    (Tensor.from #[1.0, 0.0]).reshape [1, 2] (by dsimp; decide)
  let minDxSpatial : Tensor Float [1, 2] :=
    (Tensor.from #[0.0, 1.0]).reshape [1, 2] (by dsimp; decide)
  runSmoothMaxPoolStabilityCase 1e20 1.0 maxDx2d
  runSmoothMaxPoolStabilityCase (-1e20) (-1.0) minDx2d
  runSpatialSmoothMaxPoolStabilityCase 1e20 1.0 maxDxSpatial
  runSpatialSmoothMaxPoolStabilityCase (-1e20) (-1.0) minDxSpatial

/-- Invalid inverse temperatures and zero-rank spatial pooling fail before reaching native code. -/
def runSmoothMaxPoolDomainChecks : IO Unit := do
  IO.println "== smooth max pooling domain checks =="
  let inSpatial : TorchLean.Tensor Nat [2] := [1, 2]
  let kernel : TorchLean.Tensor Nat [2] := [1, 2]
  let stride : TorchLean.Tensor Nat [2] := [1, 1]
  let padding : TorchLean.Tensor Nat [2] := [0, 0]
  let x2d : Tensor Float [1, 1, 2] := (Tensor.from #[1.0, 2.0]).reshape [1, 1, 2] (by dsimp; decide)
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) x2d
  expectCudaResultError "CPU smooth-max zero beta"
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xId 0.0)
  expectCudaResultError "CPU smooth-max negative-zero beta"
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xId (-0.0))
  expectCudaResultError "CPU smooth-max infinite beta"
    (Tape.smoothMaxPool (α := Float) (t := t1)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xId (1.0 / 0.0))

  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer x2d)
  for (label, invalidBeta) in
      [("zero", 0.0), ("negative zero", -0.0), ("binary32 overflow", 1e300),
       ("binary32 underflow", 1e-300)] do
    expectCudaResultError s!"CUDA smooth-max {label} beta"
      (Runtime.Autograd.LibTorch.Tape.smoothMaxPool (t := t1c)
        (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
        (stride := stride) (padding := padding) xIdc invalidBeta)

  let empty : TorchLean.Tensor Nat [0] := []
  let scalarInput : Tensor Float [1] := (Tensor.from #[2.0]).reshape [1] (by dsimp; decide)
  let (scalarCpu, scalarCpuId) := Tape.leaf (t := Tape.empty) scalarInput
  expectCudaResultError "CPU smooth-max zero spatial rank"
    (Tape.smoothMaxPool (α := Float) (t := scalarCpu)
      (d := 0) (C := 1) (inSpatial := empty) (kernel := empty) (stride := empty)
      (padding := empty) scalarCpuId 1.0)
  let (scalarCuda, scalarCudaId) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := Runtime.Autograd.LibTorch.Tape.empty)
      (Utils.tensorToAnyBuffer scalarInput)
  expectCudaResultError "CUDA smooth-max zero spatial rank"
    (Runtime.Autograd.LibTorch.Tape.smoothMaxPool (t := scalarCuda)
      (d := 0) (C := 1) (inSpatial := empty) (kernel := empty) (stride := empty)
      (padding := empty) scalarCudaId 1.0)

def runAvgPool : IO Unit := do
  IO.println "== avg_pool (d=2) =="
  let yShape : Shape := [inC, outH, outW]

  -- CPU
  let t0 : Tape Float := Tape.empty
  let (t1, xId) := Tape.leaf (t := t0) input (name := some "input")
  let (t2, yId) ← Utils.okOrThrow
    (Tape.avgPool (α := Float) (t := t1)
      (d := d2) (C := inC) (inSpatial := inSpatial2) (kernel := kernel2)
      (stride := stride2) (padding := padding2) xId)
  let yCpu ← Utils.cpuValue (s := yShape) t2 yId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.full yShape (1.0 : Float))
  let gradsCpu ← Utils.okOrThrow (Tape.backwardDenseAll (α := Float) (t := t2) yId seedCpu)
  let dxCpu ← Utils.cpuGrad (s := [inC, inH, inW]) gradsCpu xId

  -- CUDA
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, xIdc) := Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer input)
    (name := some "input")
  let (t2c, yIdc) ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.avgPool (t := t1c)
      (d := d2) (C := inC) (inSpatial := inSpatial2) (kernel := kernel2)
      (stride := stride2) (padding := padding2) xIdc)
  let yCuda ← Utils.cudaValue (s := yShape) t2c yIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := yShape,
      buf := Runtime.Autograd.LibTorch.Buffer.full (UInt32.ofNat (Spec.Shape.size yShape)) 1.0 }
  let gradsCuda ← Utils.okOrThrow
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t2c) yIdc seedCuda)
  let dxCuda ← Utils.cudaGrad (s := [inC, inH, inW]) gradsCuda xIdc

  Utils.assertTensorApprox (s := yShape) "avg_pool forward" yCuda yCpu (tol := 5e-3)
  Utils.assertTensorApprox (s := [inC, inH, inW]) "avg_pool dx" dxCuda dxCpu (tol := 5e-3)

/-- Every CUDA convolution and pooling operator rejects a zero stride before its FFI. -/
def runZeroStrideChecks : IO Unit := do
  IO.println "== conv/pool zero-stride validation =="
  let unitInput : Tensor Float [1, 1, 1] :=
    (Tensor.from #[2.0]).reshape [1, 1, 1] (by dsimp; decide)
  let unitKernel : Tensor Float [1, 1, 1, 1] :=
    (Tensor.from #[1.0]).reshape [1, 1, 1, 1] (by dsimp; decide)
  let unitBias : Tensor Float [1] := (Tensor.from #[0.0]).reshape [1] (by dsimp; decide)
  let (t1, kernelId) := Runtime.Autograd.LibTorch.Tape.empty.leaf (Utils.tensorToAnyBuffer
    unitKernel)
  let (t2, biasId) := t1.leaf (Utils.tensorToAnyBuffer unitBias)
  let (t3, inputId) := t2.leaf (Utils.tensorToAnyBuffer unitInput)
  let inSpatial : TorchLean.Tensor Nat [2] := [1, 1]
  let kernel : TorchLean.Tensor Nat [2] := [1, 1]
  let zeroStride : TorchLean.Tensor Nat [2] := [0, 0]
  let noPadding : TorchLean.Tensor Nat [2] := [0, 0]
  expectCudaResultError "conv zero stride"
    (Runtime.Autograd.LibTorch.Tape.conv (t := t3)
      (d := 2) (inC := 1) (outC := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := zeroStride) (padding := noPadding) kernelId biasId inputId)
  expectCudaResultError "conv_transpose zero stride"
    (Runtime.Autograd.LibTorch.Tape.convTranspose (t := t3)
      (d := 2) (inC := 1) (outC := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := zeroStride) (padding := noPadding) kernelId biasId inputId)
  expectCudaResultError "max_pool zero stride"
    (Runtime.Autograd.LibTorch.Tape.maxPool (t := t3)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := zeroStride) (padding := noPadding) inputId)
  expectCudaResultError "smooth_max_pool zero stride"
    (Runtime.Autograd.LibTorch.Tape.smoothMaxPool (t := t3)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := zeroStride) (padding := noPadding) inputId 1.0)
  expectCudaResultError "avg_pool zero stride"
    (Runtime.Autograd.LibTorch.Tape.avgPool (t := t3)
      (d := 2) (C := 1) (inSpatial := inSpatial) (kernel := kernel)
      (stride := zeroStride) (padding := noPadding) inputId)

/--
Distinct channels, strided windows and nonuniform cotangents expose channel/axis permutations.
-/
def Internal.runStridedChannels : IO Unit := do
  let spatial : Tensor Nat [1] := [4]
  let window : Tensor Nat [1] := [2]
  let stride : Tensor Nat [1] := [2]
  let padding : Tensor Nat [1] := [0]
  let x : Tensor Float [2, 4] := [[1.0, -2.0, 3.0, 4.0], [0.5, 2.0, -1.0, 3.0]]
  let k : Tensor Float [2, 2, 2] :=
    [[[0.5, -1.0], [2.0, 0.25]], [[-0.5, 0.75], [1.0, -2.0]]]
  let b : Tensor Float [2] := [0.25, -0.5]
  let seed : Tensor Float [2, 2] := [[1.0, -2.0], [0.5, 3.0]]
  let (cpu1, ck) := Tape.empty.leaf k
  let (cpu2, cb) := cpu1.leaf b
  let (cpu3, cx) := cpu2.leaf x
  let (cpu, cy) ← Utils.okOrThrow <| Tape.conv (t := cpu3)
    (inC := 2) (outC := 2) (kernel := window) (stride := stride)
    (padding := padding) (inSpatial := spatial) ck cb cx
  let cpuGradients ← Utils.okOrThrow <|
    Tape.backwardDenseAll cpu cy (Spec.SomeTensor.ofTensor seed)
  let (gpu1, gk) := Runtime.Autograd.LibTorch.Tape.empty.leaf (Utils.tensorToAnyBuffer k)
  let (gpu2, gb) := gpu1.leaf (Utils.tensorToAnyBuffer b)
  let (gpu3, gx) := gpu2.leaf (Utils.tensorToAnyBuffer x)
  let (gpu, gy) ← Utils.okOrThrow <| Runtime.Autograd.LibTorch.Tape.conv (t := gpu3)
    (inC := 2) (outC := 2) (kernel := window) (stride := stride)
    (padding := padding) (inSpatial := spatial) gk gb gx
  let gpuGradients ← Utils.okOrThrow <|
    Runtime.Autograd.LibTorch.Tape.backwardDenseAll gpu gy (Utils.tensorToAnyBuffer seed)
  Utils.assertTensorApprox "strided multichannel conv output"
    (← Utils.cudaValue (s := [2, 2]) gpu gy) (← Utils.cpuValue cpu cy) (tol := 5e-3)
  Utils.assertTensorApprox "strided multichannel conv dKernel"
    (← Utils.cudaGrad (s := [2, 2, 2]) gpuGradients gk)
    (← Utils.cpuGrad cpuGradients ck) (tol := 5e-3)
  Utils.assertTensorApprox "strided multichannel conv dBias"
    (← Utils.cudaGrad (s := [2]) gpuGradients gb)
    (← Utils.cpuGrad cpuGradients cb) (tol := 5e-3)
  Utils.assertTensorApprox "strided multichannel conv dInput"
    (← Utils.cudaGrad (s := [2, 4]) gpuGradients gx)
    (← Utils.cpuGrad cpuGradients cx) (tol := 5e-3)

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: convolution + pooling ==="
  runConv
  runConv3
  Internal.runStridedChannels
  runMaxPool
  runMaxPoolPadNegative
  runMaxPool3
  runMaxPool3PadNegative
  runMaxPoolNegativeInfinity
  runSmoothMaxPool
  runSmoothMaxPool3
  runSmoothMaxPoolStability
  runSmoothMaxPoolDomainChecks
  runAvgPool
  runAvgPool3
  runZeroStrideChecks

end ConvPool
end Cuda
end Tests
