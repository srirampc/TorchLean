/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.Universal.BinaryExecCore

/-!
# Configured binary two-layer ReLU approximation bound

This file proves the reusable three-term error decomposition for executing a
single-hidden-layer ReLU MLP over any configured executable binary format and model codec.

The theorem separates the three mathematically different sources of error:

- **real approximation**: the ideal real-valued ReLU MLP approximates the target,
- **parameter quantization**: the real MLP is close to the real interpretation of the binary
  parameters, and
- **execution**: the executable graph, interpreted back into $\mathbb{R}$, is close to the
  real graph with those interpreted parameters.

This is the finite-dimensional analogue of the hinge-network executable bound in
`UniversalApproximationBinaryExec`.  The decomposition follows the standard numerical-analysis
pattern for floating-point algorithms: prove the real algorithm correct, bound data/parameter
rounding, and bound arithmetic rounding separately.  For background, see IEEE Std 754-2019,
Goldberg (1991), Higham (2002), and the ReLU density literature of Cybenko, Hornik, Leshno, and
Pinkus.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange


namespace NN.MLTheory.Proofs.UniversalApproximation
namespace BinaryExecTwoLayerMLP

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.Proofs.ReLUMlpBridge
open BinaryExecCore

noncomputable section

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-!
## Three-term bound

Read this as:

Given a configured binary input `xI`, let $x_R$ be its real interpretation (`toReal` elementwise).
If:

1. the target $f$ is approximated by a real two-layer ReLU MLP with error
   at most $\varepsilon_{\mathrm{approx}}$,
2. the real MLP is close to the real interpretation of the binary parameters, with error at most
   $\varepsilon_Q$,
3. executing the binary MLP and then mapping to reals is close to the real interpretation
   of those binary parameters, with error at most $\varepsilon_R$,

then the binary execution approximates $f$ within
$\varepsilon_{\mathrm{approx}}+\varepsilon_Q+\varepsilon_R$.
-/

/-- Combine real approximation, parameter quantization, and execution error on a set of inputs.
The execution bound is a hypothesis about `Model.toReal`, including its treatment of exceptional
values; this theorem does not supply a rounding or finiteness bound. -/
theorem relu_mlp_approximation_three_term
    {n hidDim : Nat}
    (D : Set (Tensor Value [n]))
    (f : Tensor ℝ [n] → ℝ)
    (l1R : LinearSpec ℝ n hidDim) (l2R : LinearSpec ℝ hidDim 1)
    (l1I : LinearSpec Value n hidDim) (l2I : LinearSpec Value hidDim 1)
    (εApprox εQ εR : ℝ)
    (hApprox :
      ∀ xI ∈ D,
        let xR : Tensor ℝ [n] := tensorToReal xI
        |f xR - mlpEval (n := n) (hidDim := hidDim) l1R l2R xR| ≤ εApprox)
    (hQ :
      ∀ xI ∈ D,
        let xR : Tensor ℝ [n] := tensorToReal xI
        |mlpEval (n := n) (hidDim := hidDim) l1R l2R xR
          - mlpEval (n := n) (hidDim := hidDim) (linearSpecToReal l1I) (linearSpecToReal l2I)
            xR| ≤ εQ)
    (hR :
      ∀ xI ∈ D,
        let xR : Tensor ℝ [n] := tensorToReal xI
        |(ExecFloat.Binary.toModel (mlpEval (n := n) (hidDim := hidDim) l1I l2I
          xI)).toReal
          - mlpEval (n := n) (hidDim := hidDim) (linearSpecToReal l1I) (linearSpecToReal l2I)
            xR| ≤ εR) :
    ∀ xI ∈ D,
      let xR : Tensor ℝ [n] := tensorToReal xI
      |f xR - (ExecFloat.Binary.toModel (mlpEval (n := n) (hidDim := hidDim) l1I l2I
        xI)).toReal|
        ≤ εApprox + εQ + εR := by
  intro xI hxI
  -- Name the three intermediate values so the final bound reads as a textbook triangle argument.
  set xR : Tensor ℝ [n] := tensorToReal xI
  set yU : ℝ := mlpEval (n := n) (hidDim := hidDim) l1R l2R xR
  set yQ : ℝ := mlpEval (n := n) (hidDim := hidDim) (linearSpecToReal l1I) (linearSpecToReal
    l2I) xR
  set yI : ℝ := (ExecFloat.Binary.toModel (mlpEval (n := n) (hidDim := hidDim) l1I l2I
    xI)).toReal
  -- Apply the approximation, quantization, and execution bounds at this input.
  have h1 : |f xR - yU| ≤ εApprox := hApprox xI hxI
  have h2 : |yU - yQ| ≤ εQ := hQ xI hxI
  have h3 : |yI - yQ| ≤ εR := hR xI hxI
  -- Chain two triangle inequalities: first through the real approximant, then through the
  -- quantized real interpretation of the executable parameters.
  calc
    |f xR - yI| ≤ |f xR - yU| + |yU - yI| := abs_sub_le (f xR) yU yI
    _ ≤ |f xR - yU| + (|yU - yQ| + |yQ - yI|) := add_le_add le_rfl (abs_sub_le yU yQ yI)
    _ = |f xR - yU| + (|yU - yQ| + |yI - yQ|) := by rw [abs_sub_comm yQ yI]
    _ ≤ εApprox + (εQ + εR) := add_le_add h1 (add_le_add h2 h3)
    _ = εApprox + εQ + εR := by ring

end
end BinaryExecTwoLayerMLP
end NN.MLTheory.Proofs.UniversalApproximation
