/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.FloatInstances
-- Re-exported on purpose: `extractScalarOutput` below comes from `UniversalApproximation` through
-- this import, and the executable two-layer theorem compares against `ReLUMlpBridge.mlpEval`.
public import NN.MLTheory.Proofs.ReLU.Bridge.ReLUMlpBridge -- shake: keep

/-!
# Configured binary helpers for approximation theorems

This module contains the backend-generic glue used by the executable universal-approximation
theorems. We keep it independent of any one approximation construction (hinge sums, shallow ReLU
MLPs, convolutional models, or transformer-style models) and record the common semantic operations:

- evaluating a two-layer ReLU MLP over a configured executable binary format,
- mapping executable binary values back to their real denotation, and
- interpreting executable linear-layer parameters as real parameters.

The mathematical role is the standard one in floating-point analysis: separate the exact real
network from the concrete execution, then bridge the two by explicit rounding hypotheses
or verified rounding lemmas.  Useful references for this separation are IEEE Std 754-2019,
Goldberg's survey on floating-point arithmetic, and Higham's treatment of numerical error
analysis.  The ReLU approximation side is connected through `ReLUMlpBridge`, which supplies the
real-valued MLP semantics used by the universal-approximation files.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange


namespace NN.MLTheory.Proofs.UniversalApproximation
namespace BinaryExecCore

open _root_.Spec _root_.TorchLean

noncomputable section

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
Evaluate a two-layer ReLU MLP over a configured executable binary format.

The corresponding real-valued evaluator is `mlpEval` from `ReLUMlpBridge`. Approximation theorems
compare the real denotation `Model.toReal (toModel (mlpEval l1 l2 x))` with that real
evaluator applied to `tensorToReal x`, isolating the floating-point execution error from the
approximation and quantization errors.
-/
def mlpEval {n hidDim : Nat}
    (l1 : LinearSpec Value n hidDim) (l2 : LinearSpec Value hidDim 1)
    (x : Tensor Value [n]) : Value :=
  extractScalarOutput (Examples.mlpForward l1 l2 x)

/--
Interpret a configured binary tensor entrywise using the model's total real denotation.

Exceptional values follow `Model.toReal`; this map does not assert that the entries are finite.
-/
def tensorToReal {s : Shape} (t : Tensor Value s) : Tensor ℝ s :=
  Tensor.map (Model.toReal ∘ ExecFloat.Binary.toModel) t

/--
Interpret executable linear-layer parameters as real-valued parameters entrywise.

This helper is used in three-term bounds:
real approximation error + parameter quantization error + concrete execution error.
-/
def linearSpecToReal {inDim outDim : Nat}
    (m : LinearSpec Value inDim outDim) : LinearSpec ℝ inDim outDim :=
  { weights := tensorToReal m.weights
    bias := tensorToReal m.bias }

end
end BinaryExecCore
end NN.MLTheory.Proofs.UniversalApproximation
