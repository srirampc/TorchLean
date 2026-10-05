/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Session.Autograd
public import NN.Runtime.Autograd.Model.Session.Ops

/-! Regression checks for session-reference ownership and CUDA cache cleanup. -/

@[expose] public section

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd
open Runtime.Autograd.Model

namespace Tests.Floats.SessionRefIdentity

def expectFailure {α : Type} (label : String) (action : IO α) : IO Unit := do
  let failed ← try
    let _ ← action
    pure false
  catch _ =>
    pure true
  unless failed do
    throw <| IO.userError s!"{label}: expected failure"

/-- Observe recording and cache counters without evaluating or changing the session. -/
def recordingState (session : Session Float) : IO (Array Nat) := do
  match session.state with
  | .eager state =>
      pure #[(← state.tape.get).nodes.size, (← state.cudaTape.get).nodes.size,
        (← state.nats.get).size, ← state.rngCounter.get, ← state.refGeneration.get,
        (← state.selectedBackends.get).size, (← state.paramsByLeaf.get).size]
  | .typedGraph state =>
      let graph ← state.state.get
      let cache ← state.valueCache.get
      pure #[graph.Γ.length, graph.ss.length, graph.nat.size,
        ← state.referenceGeneration.get, ← state.stateVersion.get,
        if cache.isSome then 1 else 0, (cache.map Prod.fst).getD 0,
        (cache.map (fun entry => entry.2.size)).getD 0,
        (← state.parametersByLeaf.get).size]

/-- Rejected references must fail at the intended guard before recording or cache changes. -/
def expectReferenceFailure {α : Type} (session : Session Float)
    (label expected : String) (action : IO α) : IO Unit := do
  let before ← recordingState session
  let error? ← try
    let _ ← action
    pure none
  catch error =>
    pure (some error.toString)
  unless error? == some expected do
    throw <| IO.userError s!"{label}: expected {expected}, got {error?}"
  unless (← recordingState session) == before do
    throw <| IO.userError s!"{label}: rejection changed recording or cache state"

def checkExecutionMode (execution : Torch.ExecutionMode) : IO Unit := do
  let first ← Session.new (α := Float) (options := { execution := execution })
  let second ← Session.new (α := Float) (options := { execution := execution })
  let firstRef ← Session.input first (Tensor.scalar 1.0)
  let secondRef ← Session.input second (Tensor.scalar 2.0)
  let foreign := "torch: tensor reference belongs to a different session"
  let missing := "torch: tensor reference has no session owner"
  expectReferenceFailure second "cross-session tensor op" foreign <|
    Session.add second firstRef secondRef
  let unowned : Torch.TensorRef Float Shape.scalar := { id := 1000000 }
  expectReferenceFailure second "missing owner before foreign operand" missing <|
    Session.add second unowned firstRef
  expectReferenceFailure second "foreign operand before missing owner" foreign <|
    Session.add second firstRef unowned

  let firstNat ← Session.inputNat first 7
  expectReferenceFailure second "cross-session Nat read"
    "torch: Nat reference belongs to a different session" <| Session.getNat second firstNat

  Session.resetTape first
  let currentRef ← Session.input first (Tensor.scalar 3.0)
  expectReferenceFailure first "foreign owner before stale generation" foreign <|
    Session.add first secondRef currentRef
  expectReferenceFailure first "post-reset tensor op"
    ("torch: stale tensor reference from generation 0; current generation is 1 " ++
      "(resetTape invalidates recorded references)") <|
    Session.add first firstRef currentRef
  expectReferenceFailure first "post-reset Nat read"
    ("torch: stale Nat reference from generation 0; current generation is 1 " ++
      "(resetTape invalidates recorded references)") <| Session.getNat first firstNat

def checkCudaCacheClear : IO Unit := do
  let value ← IO.mkRef (Tensor.scalar 1.0)
  let buffer ← Runtime.Autograd.LibTorch.Buffer.fullIO 1 1.0
  let cudaValue ← IO.mkRef (some { s := Shape.scalar, buf := buffer })
  let hostCurrent ← IO.mkRef true
  let param : Torch.Param Float Shape.scalar :=
    { value, cudaValue, hostCurrent, requiresGrad := true }
  Torch.AnyParam.releaseCachedCudaValue param
  match ← cudaValue.get with
  | none => pure ()
  | some _ => throw <| IO.userError "CUDA cache release did not clear cudaValue"
  Torch.AnyParam.releaseCachedCudaValue param

def run : IO Unit := do
  checkExecutionMode .eager
  checkExecutionMode .typedGraph
  if Runtime.Autograd.LibTorch.Buffer.runtimeStatus == .nativeAvailable then
    checkCudaCacheClear
  for device in [NN.Backend.Device.cuda, .metal, .custom] do
    expectFailure "typed graph unsupported device" <|
      Session.new (α := Float) { execution := .typedGraph, device }

end Tests.Floats.SessionRefIdentity
