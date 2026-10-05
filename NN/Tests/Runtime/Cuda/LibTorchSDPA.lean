/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session
public import NN.Tests.Runtime.Cuda.Attention

/-!
# CUDA Kernel Coverage: Lean Attention

Tests for attention composed in Lean from LibTorch buffer primitives.

An independent two-key reference and finite differences check values and gradients, including a
hard boolean mask with a fully blocked row. TorchLean's global tape owns the attention node and its
backward call.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace LibTorchSDPA

open Spec TorchLean
open Tests.Cuda.Attention (checked)
open Runtime.Autograd.LibTorch

abbrev batch : Nat := 2
abbrev n : Nat := 2
abbrev d : Nat := 2

abbrev s : Shape := [batch, n, d]
abbrev maskShape : Shape := [batch, n, n]

def q : Tensor Float s :=
  (Tensor.from #[
    0.10, -0.20,
    0.30,  0.05,
   -0.15,  0.25,
    0.40, -0.10
  ]).reshape [batch, n, d] (by dsimp; decide)

def k : Tensor Float s :=
  (Tensor.from #[
    0.05,  0.20,
   -0.10,  0.30,
    0.15, -0.25,
    0.35,  0.10
  ]).reshape [batch, n, d] (by dsimp; decide)

def v : Tensor Float s :=
  (Tensor.from #[
    0.20, -0.05,
    0.10,  0.30,
   -0.20,  0.15,
    0.05, -0.10
  ]).reshape [batch, n, d] (by dsimp; decide)

def dOut : Tensor Float s :=
  (Tensor.from #[
    1.00, 0.50,
   -0.25, 0.75,
    0.30, 1.20,
   -0.60, 0.40
  ]).reshape [batch, n, d] (by dsimp; decide)

/-- `1` marks an allowed key. The last query row is fully blocked. -/
def hardMask : Tensor Float maskShape :=
  (Tensor.from #[
    1.0, 0.0,
    1.0, 1.0,
    1.0, 0.0,
    0.0, 0.0
  ]).reshape [batch, n, n] (by dsimp; decide)

/-- For this two-key case, the first probability is the logistic of the score difference. -/
@[no_expose] def referenceForward (qs ks vs : FloatArray) (masked : Bool) (scale : Float) :
    FloatArray := FloatArray.mk <| Id.run do
  let maskValues := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := maskShape) hardMask
  let mut values : Array Float := #[]
  for b in [:batch] do
    for i in [:n] do
      let firstAllowed := !masked || maskValues.get! ((b * n + i) * n) != 0.0
      let secondAllowed := !masked || maskValues.get! ((b * n + i) * n + 1) != 0.0
      let mut difference : Float := 0.0
      for c in [:d] do
        difference := difference + qs.get! ((b * n + i) * d + c) *
          (ks.get! (b * n * d + c) - ks.get! ((b * n + 1) * d + c))
      let (p, r) : Float × Float :=
        match firstAllowed, secondAllowed with
        | true, true =>
            let probability := 1.0 / (1.0 + Float.exp (-scale * difference))
            (probability, 1.0 - probability)
        | true, false => (1.0, 0.0)
        | false, true => (0.0, 1.0)
        | false, false => (0.0, 0.0)
      for c in [:d] do
        values := values.push
          (p * vs.get! (b * n * d + c) + r * vs.get! ((b * n + 1) * d + c))
  return values

@[no_expose] def perturb (xs : FloatArray) (index : Nat) (delta : Float) : FloatArray :=
  FloatArray.mk <| (Array.range xs.size).map fun j =>
    xs.get! j + if j = index then delta else 0.0

@[no_expose] def referenceLoss (qs ks vs : FloatArray) (masked : Bool) (scale : Float) : Float :=
  Id.run do
    let ys := referenceForward qs ks vs masked scale
    let seed := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) dOut
    let mut loss : Float := 0.0
    for i in [:ys.size] do
      loss := loss + ys.get! i * seed.get! i
    return loss

/-- Check all raw Q/K/V gradients against an independent reference. -/
@[no_expose] def checkReference (masked : Bool) (scale : Float)
    (output dq dk dv : Runtime.Autograd.LibTorch.Buffer) : IO Unit := do
  let qs := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) q
  let ks := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) k
  let vs := Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) v
  Tests.Cuda.Utils.assertFloatArrayApprox "attention two-key reference"
    (Runtime.Autograd.LibTorch.Buffer.toFloatArray output)
    (referenceForward qs ks vs masked scale) 1e-4
  let qGrad := Runtime.Autograd.LibTorch.Buffer.toFloatArray dq
  let kGrad := Runtime.Autograd.LibTorch.Buffer.toFloatArray dk
  let vGrad := Runtime.Autograd.LibTorch.Buffer.toFloatArray dv
  let step : Float := 1e-4
  for i in [:qs.size] do
    let expectedQ :=
      (referenceLoss (perturb qs i step) ks vs masked scale -
       referenceLoss (perturb qs i (-step)) ks vs masked scale) / (2.0 * step)
    let expectedK :=
      (referenceLoss qs (perturb ks i step) vs masked scale -
       referenceLoss qs (perturb ks i (-step)) vs masked scale) / (2.0 * step)
    let expectedV :=
      (referenceLoss qs ks (perturb vs i step) masked scale -
       referenceLoss qs ks (perturb vs i (-step)) masked scale) / (2.0 * step)
    Tests.Utils.assertApprox s!"attention finite-difference dQ[{i}]" (qGrad.get! i) expectedQ 1e-4
    Tests.Utils.assertApprox s!"attention finite-difference dK[{i}]" (kGrad.get! i) expectedK 1e-4
    Tests.Utils.assertApprox s!"attention finite-difference dV[{i}]" (vGrad.get! i) expectedV 1e-4

/-- Validation must return an error through the checked Lean API. -/
@[no_expose] def expectError {α : Type} (message : String)
    (action : Unit → Except String α) : IO Unit := do
  match ← IO.lazyPure action with
  | .error _ => pure ()
  | .ok _ => throw <| IO.userError message

/-- Portable tests check planning; CUDA builds also exercise session selection. -/
@[no_expose] def selectAttention (profile : NN.Backend.BackendProfile) :
    IO NN.Backend.KernelCapsule := do
  let options := Runtime.Autograd.Torch.Config.withBackendProfile
    ({} : Runtime.Autograd.Torch.Config) profile
  match Runtime.Autograd.LibTorch.Buffer.runtimeStatus with
  | .notLinked =>
      let planned ← Tests.Cuda.Utils.okOrThrow <|
        options.planBackendOp .attention
      pure planned.capsule
  | _ =>
      let session ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) options
      session.selectedCapsule .attention

/-- A wider head checks uniform attention and a fully blocked query row. -/
@[no_expose] def checkWideHead : IO Unit := do
  let zeros ← Runtime.Autograd.LibTorch.Buffer.zerosIO 256
  let values := Runtime.Autograd.LibTorch.Buffer.ofFloatArray <|
    FloatArray.mk <| (Array.range 256).map fun i => Float.ofNat (i / 32)
  let maskValues := Runtime.Autograd.LibTorch.Buffer.ofFloatArray <|
    FloatArray.mk <| (Array.range 64).map fun i => if i < 8 then 0.0 else 1.0
  let seed := Runtime.Autograd.LibTorch.Buffer.full 256 1.0
  let (output, probabilities) ← checked fun _ =>
    Buffer.attentionForward zeros zeros values (some maskValues) 1 8 32 0.25
  let (dq, dk, dv) ← checked fun _ =>
    Buffer.attentionBackward zeros zeros values probabilities seed 1 8 32 0.25
  let ys := Runtime.Autograd.LibTorch.Buffer.toFloatArray output
  let dqs := Runtime.Autograd.LibTorch.Buffer.toFloatArray dq
  let dks := Runtime.Autograd.LibTorch.Buffer.toFloatArray dk
  let dvs := Runtime.Autograd.LibTorch.Buffer.toFloatArray dv
  for i in [:256] do
    Tests.Utils.assertApprox s!"aligned attention output[{i}]" (ys.get! i)
      (if i < 32 then 0.0 else 3.5) 1e-4
    Tests.Utils.assertApprox s!"aligned attention dQ[{i}]" (dqs.get! i) 0.0 1e-4
    Tests.Utils.assertApprox s!"aligned attention dK[{i}]" (dks.get! i) 0.0 1e-4
    Tests.Utils.assertApprox s!"aligned attention dV[{i}]" (dvs.get! i) 0.875 1e-4
  for buffer in #[zeros, values, maskValues, seed, output, probabilities, dq, dk, dv] do
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: Lean attention ==="
  Tests.Cuda.Attention.checkSavedBuffers
  Tests.Cuda.Attention.checkFiniteInputRegressions

  let qBuf := Runtime.Autograd.LibTorch.Buffer.ofFloatArray
    (Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) q)
  let kBuf := Runtime.Autograd.LibTorch.Buffer.ofFloatArray
    (Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) k)
  let vBuf := Runtime.Autograd.LibTorch.Buffer.ofFloatArray
    (Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) v)
  let dOutBuf := Runtime.Autograd.LibTorch.Buffer.ofFloatArray
    (Runtime.Autograd.LibTorch.Convert.flattenFloat (s := s) dOut)

  let batch32 := UInt32.ofNat batch
  let n32 := UInt32.ofNat n
  let d32 := UInt32.ofNat d
  let scale : Float := 1.0 / Float.sqrt (Float.ofNat d)

  let (libTorchY, probabilities) ← checked fun _ =>
    Buffer.attentionForward qBuf kBuf vBuf none batch32 n32 d32 scale
  let (libTorchDQ, libTorchDK, libTorchDV) ← checked fun _ =>
    Buffer.attentionBackward qBuf kBuf vBuf probabilities dOutBuf batch32 n32 d32 scale
  checkReference false scale libTorchY libTorchDQ libTorchDK libTorchDV

  let maskBuf := Runtime.Autograd.LibTorch.Buffer.ofFloatArray
    (Runtime.Autograd.LibTorch.Convert.flattenFloat (s := maskShape) hardMask)
  let (libTorchMaskedY, maskedProbabilities) ← checked fun _ =>
    Buffer.attentionForward qBuf kBuf vBuf (some maskBuf) batch32 n32 d32 scale
  let (libTorchMaskedDQ, libTorchMaskedDK, libTorchMaskedDV) ← checked fun _ =>
    Buffer.attentionBackward qBuf kBuf vBuf maskedProbabilities dOutBuf batch32 n32 d32 scale
  checkReference true scale libTorchMaskedY libTorchMaskedDQ libTorchMaskedDK libTorchMaskedDV
  checkWideHead

  for testScale in #[0.0, -0.7] do
    let (output, saved) ← checked fun _ =>
      Buffer.attentionForward qBuf kBuf vBuf (some maskBuf) batch32 n32 d32 testScale
    let (dq, dk, dv) ← checked fun _ =>
      Buffer.attentionBackward qBuf kBuf vBuf saved dOutBuf batch32 n32 d32 testScale
    checkReference true testScale output dq dk dv
    for buffer in #[output, saved, dq, dk, dv] do
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer

  -- Exercise capsule routing and the global TorchLean tape with the Lean local VJP.
  let selectedAttention ← selectAttention NN.Backend.BackendProfile.checkedCuda
  unless selectedAttention.sameIdentity NN.Backend.LibTorch.attention do
    throw <| IO.userError <|
      s!"default checkedCuda selected unexpected attention capsule `{selectedAttention.name}`"
  let base0 : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (base1, wqId) := Runtime.Autograd.LibTorch.Tape.leaf (t := base0)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.wq)
  let (base2, wkId) := Runtime.Autograd.LibTorch.Tape.leaf (t := base1)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.wk)
  let (base3, wvId) := Runtime.Autograd.LibTorch.Tape.leaf (t := base2)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.wv)
  let (base4, woId) := Runtime.Autograd.LibTorch.Tape.leaf (t := base3)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.wo)
  let (base5, xId) := Runtime.Autograd.LibTorch.Tape.leaf (t := base4)
    (Tests.Cuda.Utils.tensorToAnyBuffer Tests.Cuda.Attention.x)
  let attentionResult ← Runtime.Autograd.LibTorch.Tape.attention (t := base5)
    (n := Tests.Cuda.Attention.n) (numHeads := Tests.Cuda.Attention.numHeads)
    (dModel := Tests.Cuda.Attention.dModel) (headDim := Tests.Cuda.Attention.headDim)
    (h1 := Tests.Cuda.Attention.n_ne_zero) wqId wkId wvId woId xId
    (mask := some Tests.Cuda.Attention.mask)
  let (tape, yId) ← Tests.Cuda.Utils.okOrThrow attentionResult
  let seed : AnyBuffer :=
    { s := [Tests.Cuda.Attention.n, Tests.Cuda.Attention.dModel], buf := Buffer.full 8 1.0 }
  let gradients ← Tests.Cuda.Utils.okOrThrow <|
    Runtime.Autograd.LibTorch.Tape.backwardDenseAll tape yId seed
  let output ← Tests.Cuda.Utils.cudaValue (s := [2, 4]) tape yId
  let dx ← Tests.Cuda.Utils.cudaGrad (s := [2, 4]) gradients xId

  let (cpu1, cq) := Runtime.Autograd.Tape.leaf (t := Runtime.Autograd.Tape.empty)
    Tests.Cuda.Attention.wq
  let (cpu2, ck) := Runtime.Autograd.Tape.leaf (t := cpu1) Tests.Cuda.Attention.wk
  let (cpu3, cv) := Runtime.Autograd.Tape.leaf (t := cpu2) Tests.Cuda.Attention.wv
  let (cpu4, co) := Runtime.Autograd.Tape.leaf (t := cpu3) Tests.Cuda.Attention.wo
  let (cpu5, cx) := Runtime.Autograd.Tape.leaf (t := cpu4) Tests.Cuda.Attention.x
  let (cpu6, cy) ← Tests.Cuda.Utils.okOrThrow <|
    Runtime.Autograd.Tape.attention (t := cpu5)
      (n := Tests.Cuda.Attention.n) (numHeads := Tests.Cuda.Attention.numHeads)
      (dModel := Tests.Cuda.Attention.dModel) (headDim := Tests.Cuda.Attention.headDim)
      Tests.Cuda.Attention.n_ne_zero cq ck cv co cx (some Tests.Cuda.Attention.mask)
  let expectedOutput ← Tests.Cuda.Utils.cpuValue (s := [2, 4]) cpu6 cy
  let cpuGradients ← Tests.Cuda.Utils.okOrThrow <|
    Runtime.Autograd.Tape.backwardDenseAll cpu6 cy
      (Spec.SomeTensor.ofTensor (Tensor.full [2, 4] (1.0 : Float)))
  let expectedDx ← Tests.Cuda.Utils.cpuGrad (s := [2, 4]) cpuGradients cx
  Tests.Cuda.Utils.assertTensorApprox "attention global tape output" output expectedOutput 1e-4
  Tests.Cuda.Utils.assertTensorApprox "attention global tape dX" dx expectedDx 1e-4

  let short ← Buffer.zerosIO 1
  for (label, qs, ks, vs) in
      #[("Q", short, kBuf, vBuf), ("K", qBuf, short, vBuf), ("V", qBuf, kBuf, short)] do
    expectError s!"attention forward accepted {label} with the wrong size" fun _ =>
      Buffer.attentionForward qs ks vs none batch32 n32 d32 scale
    expectError s!"attention backward accepted {label} with the wrong size" fun _ =>
      Buffer.attentionBackward qs ks vs probabilities dOutBuf batch32 n32 d32 scale
  expectError "attention accepted a dOut buffer with the wrong size" fun _ =>
    Buffer.attentionBackward qBuf kBuf vBuf probabilities short batch32 n32 d32 scale
  expectError "attention accepted probabilities with the wrong size" fun _ =>
    Buffer.attentionBackward qBuf kBuf vBuf short dOutBuf batch32 n32 d32 scale
  expectError "attention accepted a mask buffer with the wrong size" fun _ =>
    Buffer.attentionForward qBuf kBuf vBuf (some short) batch32 n32 d32 scale
  for invalidScale in #[1.0 / 0.0, -1.0 / 0.0, 0.0 / 0.0, 1e40, -1e40] do
    expectError "attention forward accepted a scale outside finite float32" fun _ =>
      Buffer.attentionForward qBuf kBuf vBuf none batch32 n32 d32 invalidScale
    expectError "attention backward accepted a scale outside finite float32" fun _ =>
      Buffer.attentionBackward qBuf kBuf vBuf probabilities dOutBuf batch32 n32 d32 invalidScale
  expectError "attention forward accepted an overflowing shape" fun _ =>
    Buffer.attentionForward qBuf kBuf vBuf none 4294967295 4294967295 4294967295 scale
  expectError "attention backward accepted an overflowing shape" fun _ =>
    Buffer.attentionBackward qBuf kBuf vBuf probabilities dOutBuf
      4294967295 4294967295 4294967295 scale
  for buffer in #[qBuf, kBuf, vBuf, dOutBuf, maskBuf, short, probabilities, maskedProbabilities,
      libTorchY, libTorchDQ, libTorchDK, libTorchDV,
      libTorchMaskedY, libTorchMaskedDQ, libTorchMaskedDK, libTorchMaskedDV, seed.buf] do
    discard <| Buffer.releaseIO buffer
  IO.println "== Lean attention: OK =="

@[no_expose] def main (args : List String) : IO Unit :=
  match args with
  | [] => run
  | _ => throw <| IO.userError "usage: libtorch_sdpa_test"

end LibTorchSDPA
end Cuda
end Tests

@[no_expose] def main (args : List String) : IO Unit :=
  Tests.Cuda.LibTorchSDPA.main args
