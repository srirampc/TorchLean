/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Trainer
public import NN.API.Trainer.Constructor
public import NN.API.Trainer.Train.Loop

/-!
# Trainer Run Configuration API Tests

Regression checks for runtime-option conversion and the example-owned command-line adapter.
-/

@[expose] public section

namespace NN.Tests.API.TrainerRun

open TorchLean

def fail {α : Type} (message : String) : IO α :=
  throw <| IO.userError s!"trainer run configuration check failed: {message}"

def parse (args : List String) (base : Trainer.RunConfig := {}) :
    IO (Trainer.RunConfig × List String) :=
  match TorchLean.CLI.Trainer.parse args base with
  | .ok result => pure result
  | .error message => fail message

/--
The example CLI adapter keeps the default singleton update and honors an explicit two-item batch.
With zero input, one SGD step changes only the bias; its independent mean-gradient formula
distinguishes using the first target alone from consuming both source items.
-/
def checkStreamBatchSize : IO Unit := do
  let trainer := Trainer.new (nn.linear 1 1)
    { optimizer := optim.sgd { learningRate := 0.25 }, seed := 17 }
  let sampleAt (index : Nat) : Sample.Supervised Float [1] [1] :=
    { input := [0.0], target := [if index = 0 then 0.0 else 2.0] }
  let before ← trainer.predict ([0.0] : Tensor Float [1])
  for (flags, batchSize, meanTarget) in
      [([], 1, (0.0 : Float)), (["--batch-size", "2"], 2, 1.0)] do
    let (parsed, rest) ← IO.ofExcept <|
      CLI.Training.OptimizerOptions.parse "stream regression"
        (["--steps", "1", "--lr", "0.25"] ++ flags) "unused.json"
    let options := parsed.trainOptions (enableLog := false)
    unless rest.isEmpty && options.samplesPerStep == batchSize do
      fail "example training options dropped the default or explicit batch size"
    let trained ← trainer.trainStream { execution := .typedGraph }
      sampleAt (sampleAt 0) options
    let prediction ← trained.predict ([0.0] : Tensor Float [1])
    let expected := before[0] - 0.25 * (2.0 * (before[0] - meanTarget))
    unless prediction[0].isFinite && Float.abs (prediction[0] - expected) ≤ 1e-6 &&
        trained.trained.report.steps == 1 do
      fail s!"batch size {batchSize} did not apply one mean-gradient update"

def run : IO Unit := do
  checkStreamBatchSize
  let runtime : Runtime.Config :=
    { execution := .typedGraph
      device := .cpu
      showBackend := true }
  let base : Trainer.RunConfig :=
    { optimizer := optim.sgd { learningRate := 0.25 }
      arithmetic := .ieee }
  let converted := Trainer.RunConfig.fromRuntime runtime base
  unless converted.execution == .typedGraph &&
      converted.device == .cpu &&
      converted.showBackend &&
      converted.arithmetic == .ieee &&
      converted.optimizer.describe == base.optimizer.describe do
    fail "fromRuntime did not preserve runtime and trainer-owned fields"

  let (parsed, rest) ← parse
    ["--arithmetic", "ieee", "--execution", "typed-graph",
      "--device", "cpu", "--show-backend", "--steps", "4"]
  unless parsed.arithmetic == .ieee &&
      parsed.execution == .typedGraph &&
      parsed.device == .cpu &&
      parsed.showBackend &&
      rest == ["--steps", "4"] do
    fail "parse did not consume exactly the runtime-owned flags"

  let (roundTrip, trailing) ←
    parse (TorchLean.CLI.Trainer.cliArguments parsed)
  unless roundTrip.arithmetic == parsed.arithmetic &&
      roundTrip.execution == parsed.execution &&
      roundTrip.device == parsed.device &&
      roundTrip.showBackend == parsed.showBackend &&
      trailing.isEmpty do
    fail "cliArguments did not round-trip through parse"

  match Runtime.Training.LogDestination.parse " OFF " with
  | .disabled => pure ()
  | .json _ => fail "LogDestination.parse did not recognize a disabled spelling"

  match Runtime.Training.LogDestination.parse "training.json" with
  | .disabled => fail "LogDestination.parse disabled a JSON path"
  | .json path =>
      unless path == ("training.json" : System.FilePath) do
        fail "LogDestination.parse changed the JSON path"

  match Runtime.Training.LogDestination.resolve (.json "default.json") none with
  | .disabled => fail "LogDestination.resolve disabled the default destination"
  | .json path =>
      unless path == ("default.json" : System.FilePath) do
        fail "LogDestination.resolve changed the default path"

  match Runtime.Training.LogDestination.resolve (.json "default.json") (some "disabled") with
  | .disabled => pure ()
  | .json _ => fail "LogDestination.resolve ignored an explicit disabled value"

  unless Runtime.Training.LogDestination.resolve .disabled none == .disabled do
    fail "LogDestination.resolve did not preserve a disabled default"

  unless Runtime.Training.LogDestination.disabled.path? == none do
    fail "LogDestination.path? returned a path for a disabled destination"

  IO.println "  trainer run configuration: passed"

end NN.Tests.API.TrainerRun
