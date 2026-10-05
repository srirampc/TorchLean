/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.IBPCert
public import NN.Tensor

/-!
# General input refinement regressions

These checks exercise a dense/ReLU reduction, shared-input multiplication, multiple graph inputs,
and failed branches. They check containment as well as tighter bounds: returning only a convenient
subset of the input region would give narrow but incorrect answers.
-/

public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)

namespace NN.Tests.MLTheory.IBPRefinement

open Spec TorchLean
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

private def require (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw <| IO.userError s!"IBP refinement: {message}"

private def interval (lo hi : Float) : FlatBox Float :=
  { dim := 1, lo := Tensor.ofFn fun _ => lo, hi := Tensor.ofFn fun _ => hi }

private def endpoints (box : FlatBox Float) : Float × Float :=
  (getAtOrZero box.lo [0], getAtOrZero box.hi [0])

private def output (g : Graph) (ps : ParamStore Float) (outId budget : Nat) :
    IO (Float × Float) := do
  let some box := refinedIBPOutput? g ps 0 outId budget
    | throw <| IO.userError "IBP refinement: expected an output box"
  pure (endpoints box)

private def inputNode (id : Nat) : Node :=
  { id, kind := .input, parents := #[], outShape := [1] }

private def checkContainment (bounds : Float × Float) (value : Float) : IO Unit :=
  require (bounds.1 ≤ value && value ≤ bounds.2) s!"lost value {value} from {bounds}"

/-- The shared refinement applies through different operations, without inspecting model names. -/
private def checkGraphs : IO Unit := do
  let ps : ParamStore Float := ({} : ParamStore Float).seedInputBox 0 (interval (-1) 1)
  let dense : Graph := { nodes := #[inputNode 0,
    { id := 1, kind := .linear, parents := #[0], outShape := [2] },
    { id := 2, kind := .relu, parents := #[1], outShape := [2] },
    { id := 3, kind := .sum, parents := #[2], outShape := [] }] }
  let linear : LinParams Float :=
    { m := 2, n := 1, w := [[1], [-1]], b := [0, 0] }
  let densePs := { ps with linearWB := ps.linearWB.insert 1 linear }
  let original ← output dense densePs 3 0
  let refined ← output dense densePs 3 1
  require (original.2 > 1.9 && refined.2 < 1.01) "dense/ReLU sum did not tighten"
  IO.FS.withTempFile fun handle path => do
    handle.putStr "{\"result\":{\"lo\":[-0.01],\"hi\":[1.01]}}"
    handle.flush
    let baselineAccepted ← NN.Verification.Cert.IBPCert.check dense densePs 3 path.toString
    let refinedAccepted ←
      NN.Verification.Cert.IBPCert.check dense densePs 3 path.toString (some (0, 1))
    require (!baselineAccepted && refinedAccepted) "refinement not used by certificate checker"
  let square : Graph := { nodes := #[inputNode 0,
    { id := 1, kind := .mulElem, parents := #[0, 0], outShape := [1] }] }
  let squareOriginal ← output square ps 1 0
  let squareRefined ← output square ps 1 7
  require (squareOriginal.1 < -0.9 && squareRefined.1 > -0.01) "square did not tighten"
  for k in List.range 201 do
    let x := -1 + Float.ofNat k / 100
    checkContainment refined x.abs
    checkContainment squareRefined (x * x)
  let twoInputs : Graph := { nodes := #[inputNode 0, inputNode 1,
    { id := 2, kind := .add, parents := #[0, 1], outShape := [1] }] }
  let sumBounds ← output twoInputs (ps.seedInputBox 1 (interval 10 20)) 2 3
  checkContainment sumBounds 9
  checkContainment sumBounds 21
  require ((refinedIBPOutput? square ps 1 1 3).isNone) "non-input split target accepted"
  require ((refinedIBPOutput? square ps 9 1 3).isNone) "missing input accepted"

/-- The same subdivision code runs on both native and modeled binary32 endpoints. -/
private def checkScalarBackend (α : Type) [TorchLean.Storage α] [Context α]
    [BoundOps α] [NonlinearBoundOps α] [DecidableLE α] (label : String) : IO Unit := do
  let g : Graph := { nodes := #[inputNode 0,
    { id := 1, kind := .mulElem, parents := #[0, 0], outShape := [1] }] }
  let input : FlatBox α :=
    { dim := 1, lo := Tensor.ofFn fun _ => -1, hi := Tensor.ofFn fun _ => 1 }
  let ps := ({} : ParamStore α).seedInputBox 0 input
  let some result := refinedIBPOutput? g ps 0 1 3
    | throw <| IO.userError s!"IBP refinement: {label} returned no box"
  let lo := getAtOrZero result.lo [0]
  let hi := getAtOrZero result.hi [0]
  require (decide (lo > -(1 : α) / 100)) s!"{label} square did not tighten"
  require (decide (lo ≤ (0 : α)) && decide ((1 : α) ≤ hi))
    s!"{label} square lost an endpoint"

/-- Many small products must not disappear from a matrix product's real enclosure.
The exact dyadic sum is representable, while nearest-rounded accumulation loses every small term.
Both matrix ranks use the same directed endpoint policy. -/
private def checkMatmulAccumulation (α : Type) [TorchLean.Storage α] [Context α]
    [BoundOps α] [NonlinearBoundOps α] [DecidableLE α] (small : α) (label : String) : IO Unit := do
  let k := 129
  let a : FlatBox α := FlatBox.ofTensor <|
    Tensor.ofFn (n := k) fun i => if i.val = 0 then 1 else small
  let b : FlatBox α := FlatBox.ofTensor (Tensor.full [k] 1)
  let ps := ({} : ParamStore α).seedInputBox 0 a |>.seedInputBox 1 b
  let exact := (1 : α) + 128 * small
  for batched in [false, true] do
    let aShape : Shape := if batched then [1, 1, k] else [1, k]
    let bShape : Shape := if batched then [1, k, 1] else [k, 1]
    let outShape : Shape := if batched then [1, 1, 1] else [1, 1]
    let g : Graph := { nodes := #[
      { id := 0, kind := .input, parents := #[], outShape := aShape },
      { id := 1, kind := .input, parents := #[], outShape := bShape },
      { id := 2, kind := .matmul, parents := #[0, 1], outShape }] }
    let some box := (runIBP g ps)[2]?.join
      | throw <| IO.userError s!"IBP refinement: {label} missing matrix product"
    require (Refinement.valid box) s!"{label} invalid matrix endpoints"
    require (decide (getAtOrZero box.lo [0] ≤ exact) &&
      decide (exact ≤ getAtOrZero box.hi [0]))
      s!"{label} lost small terms in batched={batched} matrix bounds"

/-- Degenerate domains and failed child enclosures cannot silently lose coverage. -/
private def checkFailures : IO Unit := do
  require ((Refinement.split? (interval 1 1)).isNone) "singleton split"
  require ((Refinement.split? (interval 1 (HostFloat.nextUp 1))).isNone)
    "adjacent endpoints split without an interior value"
  require ((Refinement.split? (interval 1.6e308 1.7e308)).isSome) "finite midpoint overflow"
  require ((Refinement.split? (interval (0 / 0) 1)).isNone) "NaN endpoint split"
  require ((Refinement.split? (interval 2 1)).isNone) "reversed endpoints split"
  let input := interval (-1) 1
  let missesLeft := fun box : FlatBox Float =>
    if (endpoints box).1 < 0 then none else some box
  require ((Refinement.bound missesLeft input 3).isNone) "failed left branch dropped"
  let missesRight := fun box : FlatBox Float =>
    if (endpoints box).2 > 0 then none else some box
  require ((Refinement.bound missesRight input 3).isNone) "failed right branch dropped"
  require ((Refinement.combine? false (interval 0 1) (interval 2 3)).isNone)
    "disjoint bounds treated as a valid intersection"
  let some hull := Refinement.combine? true (interval (-2) (-1)) (interval 1 2)
    | throw <| IO.userError "IBP refinement: missing hull"
  checkContainment (endpoints hull) (-2)
  checkContainment (endpoints hull) 2

/-- Invalid recomputed bounds must fail before comparing against a serialized certificate. -/
private def checkInvalidCertificate : IO Unit := do
  let g : Graph := { nodes := #[inputNode 0] }
  let wrongShape : FlatBox Float :=
    { dim := 0, lo := Tensor.ofFn fun i => Fin.elim0 i, hi := Tensor.ofFn fun i => Fin.elim0 i }
  for box in [interval 2 1, interval (0 / 0) 1, wrongShape] do
    let ps : ParamStore Float := ({} : ParamStore Float).seedInputBox 0 box
    let rejected ← try
      let _ ← NN.Verification.Cert.IBPCert.check g ps 0 "/unused-invalid-ibp-certificate.json"
      pure false
    catch error => pure (error.toString.contains "invalid input bounds")
    require rejected "invalid input interval reached certificate comparison"
    require ((refinedIBPOutput? g ps 0 0 0).isNone) "invalid refined output accepted"

/-- Run the generic subdivision and certificate validity checks. -/
def run : IO Unit := do
  checkGraphs
  checkFailures
  checkScalarBackend Float32 "native Float32"
  checkScalarBackend (Binary 8 23) "ExecFloat.Binary 8 23"
  checkMatmulAccumulation Float (1 / 18014398509481984) "Float"
  checkMatmulAccumulation Float32 (1 / 67108864) "Float32"
  checkMatmulAccumulation (Binary 8 23)
    (ExecFloat.div (1 : Binary 8 23) (67108864 : Binary 8 23)) "ExecFloat.Binary 8 23"
  checkInvalidCertificate
  IO.println "IBP refinement: containment, tightening, and failure checks passed"

end NN.Tests.MLTheory.IBPRefinement
