# `NN.Spec.Autograd`

This directory defines TorchLean's spec-level interface for reverse-mode automatic differentiation.
The interface is intentionally small:

- an `OpSpec` is a forward function plus a VJP, and
- composition of `OpSpec`s is the chain rule.

This layer is independent of any particular runtime tape, typed graph, CUDA kernel, or external
provider. It is the mathematical contract that runtime code should implement or reuse so executable
training stays aligned with the spec.

The record contains no proof that its backward function is the derivative of its forward
function. Derivative theorems carry their own scalar and domain hypotheses; selected subgradients
and floating-point backward rules can use the same interface.

PyTorch analogy:

- `OpSpec.forward` is like `torch.autograd.Function.forward`.
- `OpSpec.backward` is like `torch.autograd.Function.backward`, but expressed as a pure VJP that
  receives the input explicitly instead of reading a mutable context.

## Files

- `AutogradSpec.lean`: defines `Spec.OpSpec` and sequential composition (`compose` / `>>>`).
- `Ops.lean`: common `OpSpec`s built from spec tensor primitives: activations, pointwise math,
  reductions, broadcasting-aware wrappers, and loss functions.

## Related Modules

- `NN/Runtime/Autograd/`: executable tape/graph engines and training utilities.
- `NN/GraphSpec/` and `NN/IR/`: typed DAG and op-tagged IR representations used by lowering and
  verification.
- `NN/Proofs/Autograd/`: correctness statements relating runtime execution or tape algebra to
  spec-level derivatives.

Synchronization rule: define the math once in `NN/Spec/*` and have runtime code call those helpers
when possible. That keeps runtime rules, proof statements, and spec rules tied to the same
vocabulary.
