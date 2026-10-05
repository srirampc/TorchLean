/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Binary
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Core
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SafeDivSigmoid
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Softmax
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SoftplusSafeLog
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Unary

/-!
# NF Elementwise Bounds

Elementwise approximation lemmas for the rounded `NF` backend: per-entry error budgets and the
`approxTensor` bounds for arithmetic and activation operators applied pointwise.
-/
