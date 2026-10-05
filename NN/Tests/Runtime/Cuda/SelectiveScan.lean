/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Kernels
public import NN.Tests.Runtime.Cuda.Utils

/-!
# CUDA kernel coverage: diagonal selective scan

This checks the low-level buffer primitive backing the first Mamba/SSM runtime path under
`scripts/lake.sh -Kcuda=true test`.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace SelectiveScan

open Runtime.Autograd.LibTorch

-- Shared buffer assertions check tolerances and reject nonfinite values.
open Tests.Cuda.Utils (floatArray assertFloatArrayApprox)

def run : IO Unit := do
  IO.println "== CUDA selective_scan_diag_fwd =="

  let A := Buffer.ofFloatArray (floatArray #[0.5, 0.25])
  let B := Buffer.ofFloatArray (floatArray #[1.0, 2.0])
  let h0 := Buffer.ofFloatArray (floatArray #[1.0, -1.0])
  let X := Buffer.ofFloatArray (floatArray #[
    2.0, 1.0,
    4.0, -2.0,
    0.0, 3.0
  ])

  let out := Buffer.selectiveScanDiagFwd A B X h0 3 2
  let expected := floatArray #[
    2.5, 1.75,
    5.25, -3.5625,
    2.625, 5.109375
  ]
  assertFloatArrayApprox "selectiveScanDiagFwd" (Buffer.toFloatArray out) expected (tol := 1e-5)

  let dY := Buffer.ofFloatArray (floatArray #[
    1.0, 1.0,
    1.0, 1.0,
    1.0, 1.0
  ])
  let (dA, dB, dX, dH0) := Buffer.selectiveScanDiagBwd A B X h0 out dY 3 2
  assertFloatArrayApprox "selectiveScanDiagBwd.dA" (Buffer.toFloatArray dA)
    (floatArray #[10.75, -2.6875]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dB" (Buffer.toFloatArray dB)
    (floatArray #[9.5, 1.8125]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dX" (Buffer.toFloatArray dX)
    (floatArray #[
      1.75, 2.625,
      1.5, 2.5,
      1.0, 2.0
    ]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagBwd.dH0" (Buffer.toFloatArray dH0)
    (floatArray #[0.875, 0.328125]) (tol := 1e-5)

  let emptyX := Buffer.ofFloatArray (floatArray #[])
  let emptyOut := Buffer.selectiveScanDiagFwd A B emptyX h0 0 2
  if Buffer.size emptyOut != 0 then
    throw <| IO.userError
      s!"selectiveScanDiagFwd empty seq: got size {Buffer.size emptyOut}, expected 0"

  let emptyParam := Buffer.ofFloatArray (floatArray #[])
  let zeroStateOut := Buffer.selectiveScanDiagFwd emptyParam emptyParam emptyX emptyParam 3 0
  if Buffer.size zeroStateOut != 0 then
    throw <| IO.userError
      s!"selectiveScanDiagFwd zero state: got size {Buffer.size zeroStateOut}, expected 0"

  let Avar := Buffer.ofFloatArray (floatArray #[
    0.5, 0.25,
    0.1, 0.75,
    -1.0, 0.2
  ])
  let Bvar := Buffer.ofFloatArray (floatArray #[
    1.0, 2.0,
    0.5, -1.0,
    0.25, 0.5
  ])
  let Xvar := Buffer.ofFloatArray (floatArray #[
    2.0, 1.0,
    4.0, -2.0,
    0.0, 3.0
  ])
  let outVar := Buffer.selectiveScanDiagVarFwd Avar Bvar Xvar h0 3 2
  let expectedVar := floatArray #[
    2.5, 1.75,
    2.25, 3.3125,
    -2.25, 2.1625
  ]
  assertFloatArrayApprox "selectiveScanDiagVarFwd" (Buffer.toFloatArray outVar) expectedVar
    (tol := 1e-5)

  -- Nonuniform cotangents expose time-index shifts in the variable-coefficient VJP.
  let dyVar := Buffer.ofFloatArray (floatArray #[1.0, -2.0, 3.0, 4.0, -1.0, 0.5])
  let (daVar, dbVar, dxVar, dhVar) :=
    Buffer.selectiveScanDiagVarBwd Avar Bvar Xvar h0 outVar dyVar 3 2
  assertFloatArrayApprox "selectiveScanDiagVarBwd.dA" (Buffer.toFloatArray daVar)
    (floatArray #[1.4, -1.075, 10.0, 7.175, -2.25, 1.65625]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagVarBwd.dB" (Buffer.toFloatArray dbVar)
    (floatArray #[2.8, 1.075, 16.0, -8.2, 0.0, 1.5]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagVarBwd.dX" (Buffer.toFloatArray dxVar)
    (floatArray #[1.4, 2.15, 2.0, -4.1, -0.25, 0.25]) (tol := 1e-5)
  assertFloatArrayApprox "selectiveScanDiagVarBwd.dH0" (Buffer.toFloatArray dhVar)
    (floatArray #[0.7, 0.26875]) (tol := 1e-5)

  let (daEmpty, dbEmpty, dxEmpty, dhEmpty) :=
    Buffer.selectiveScanDiagBwd A B emptyX h0 emptyOut emptyX 0 2
  for (label, buffer, expected) in [
      ("dA", daEmpty, floatArray #[0.0, 0.0]),
      ("dB", dbEmpty, floatArray #[0.0, 0.0]),
      ("dX", dxEmpty, floatArray #[]),
      ("dH0", dhEmpty, floatArray #[0.0, 0.0])] do
    assertFloatArrayApprox s!"selectiveScanDiagBwd.empty.{label}"
      (Buffer.toFloatArray buffer) expected (tol := 0.0)

  IO.println "== CUDA selective_scan_diag_fwd: OK =="

end SelectiveScan
end Cuda
end Tests
