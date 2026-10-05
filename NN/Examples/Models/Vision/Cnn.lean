/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Real-data CUDA example:
  python3 scripts/datasets/download_example_data.py --cifar10
  scripts/lake.sh -Kcuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
-/

module

public import NN.Examples.Models.Common.RealData

/-!
# CNN Training Example

Runnable `torchlean cnn` example. It trains a small convolutional classifier on a prepared CIFAR-10
minibatch.

The reusable model wiring lives behind the public `TorchLean.nn.models.cnn` constructor. The
command adds the pieces around it: CLI parsing, dataset selection, step-limited loader training, and
TrainLog artifact writing.

```bash
python3 scripts/datasets/download_example_data.py --cifar10
scripts/lake.sh -Kcuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Vision.Cnn

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "cnn"

/-- Default JSON loss-curve path for this command. -/
def defaultLogPath : System.FilePath := Support.trainLogPath "cnn"

/--
Static minibatch size for the compact CIFAR run.

The model owns the batch axis, so this value appears in both the input/output shapes and the
classifier trainer type.
-/
def batchSize : Nat := 1

/-- CIFAR image channels. -/
def inputChannels : Nat := RealData.cifarChannels

/-- Height of the CIFAR crop used by this runnable CNN command. -/
def cropHeight : Nat := 8

/-- Width of the CIFAR crop used by this runnable CNN command. -/
def cropWidth : Nat := 8

/-- CIFAR class count, hence the output-logit width. -/
def classCount : Nat := RealData.cifarClasses

/-- Shared CNN configuration used by both the model and its checked tensor shapes. -/
abbrev modelConfig : nn.models.CNN.Config 2 :=
  { inputChannels := inputChannels
    spatial := [cropHeight, cropWidth]
    stages :=
      [{ block := { convolution :=
           { outChannels := 4
             kernelSize := [3, 3]
             stride := [2, 2]
             padding := [1, 1] } }
         pooling := { kernelSize := [2, 2], stride := [2, 2] } }]
    classCount := classCount }

/-- Input shape: a minibatch of CIFAR images in channel-first layout. -/
abbrev input : Shape := modelConfig.inputShape [batchSize]

/-- Output shape: one row of class logits per image. -/
abbrev output : Shape := modelConfig.outputShape [batchSize]

/--
Small convolutional classifier from the public model API.

The command chooses the CIFAR paths and runtime flags; the model itself stays an ordinary
`nn.Sequential` value built from the public API.
-/
def model : nn.Builder (nn.Sequential input output) :=
  nn.models.cnn modelConfig [batchSize]

/-- Train the CIFAR CNN with the public `Trainer` surface. -/
def train (runtime : Runtime.Config) (flags : Support.Training.Options Support.Npy.Options) :
    IO Trainer.Report :=
  RealData.trainCifarClassifier batchSize cropHeight cropWidth exeName
    "CNN training" model runtime flags

/-- CLI entrypoint for CIFAR CNN training on the selected runtime device. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.npy exeName args
    (fun rest => Support.Training.Options.parse exeName rest defaultLogPath 1 1e-3
      (parseData := RealData.NpyDatasets.parseCifar))
    (Support.banner exeName "CNN training")
    train (fun result => result.printSummary) (target := "class-label")

end NN.Examples.Models.Vision.Cnn
