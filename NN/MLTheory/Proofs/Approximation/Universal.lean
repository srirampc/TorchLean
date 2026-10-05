/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.Universal.BinaryExecCore
public import NN.MLTheory.Proofs.Approximation.Universal.StoneWeierstrass
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximation
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationBinaryExecTwoLayerMlp
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationRounded
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationBinaryExec
public import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationRate

/-!
# Universal-approximation proofs

This entrypoint collects the constructive ReLU approximation results:

- one-dimensional constructive ReLU approximation by hinge sums;
- quantitative width/rate refinements;
- hinge-network error bounds through rounded-real and executable configured binary semantics;
- a two-layer MLP error decomposition over configured executable binary formats; and
- an `n`-dimensional Stone-Weierstrass bridge through coordinate polynomials.

The results connect the exact real-valued construction to TorchLean's spec-level MLP and executable
binary arithmetic.

References:
- Cybenko, "Approximation by superpositions of a sigmoidal function", 1989.
- Leshno, Lin, Pinkus, and Schocken, "Multilayer feedforward networks with a nonpolynomial
  activation function can approximate any function", 1993.
- Pinkus, *Approximation Theory of the MLP Model in Neural Networks*, 1999.
- Yarotsky, "Error bounds for approximations with deep ReLU networks", 2017.
-/

@[expose] public section
