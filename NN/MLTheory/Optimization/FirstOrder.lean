/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context.Real
public import NN.Runtime.Optim.Optimizers

/-!
# First-Order Optimizer Relations

Relations between distinct first-order optimizer updates.

This is the tensor-facing layer: the statements are phrased over `TorchLean.Tensor` and the
executable operator dictionary `Spec.Context`. When we want ordinary algebraic simplification, such
as proving that a zero weight-decay AdamW update is the same parameter update as Adam, we
specialize to `ℝ`, where mathlib provides the ring laws.
-/

@[expose] public section


namespace Optim
open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace AdamW

/--
AdamW reduces to Adam when `weightDecay = 0` (parameter-update equality), over `ℝ`.

`Context α` gives executable operators but no algebraic laws such as `x * 0 = 0`, so the statement
is made over `ℝ`, where Mathlib supplies them.
-/
theorem update_weight_decay_zero_parameters_eq_adam_real {s : Shape}
    (state : State ℝ s) (parameters gradients : Tensor ℝ s) (hwd : state.weightDecay = 0) :
    (update state parameters gradients).parameters =
      (Adam.update
          ({ learningRate := state.learningRate
             beta1 := state.beta1
             beta2 := state.beta2
             epsilon := state.epsilon
             firstMoment := state.firstMoment
             secondMoment := state.secondMoment
             stepCount := state.stepCount } : Adam.State ℝ s)
          parameters gradients).parameters := by
  -- With `weightDecay = 0` the decoupled decay term is `scaleSpec parameters 0`, which is the zero
  -- tensor, and subtracting the zero tensor is the identity.
  have scaleSpec_zero : ∀ {s : Shape} (t : Tensor ℝ s), scaleSpec t (0 : ℝ) = Tensor.full s 0 := by
    intro s t
    apply TorchLean.Tensor.Internal.Rep.ext
    intro coordinate
    simp [Tensor.scaleSpec, Tensor.mapSpec, Tensor.map]
  have subSpec_full_zero : ∀ {s : Shape} (t : Tensor ℝ s), subSpec t (Tensor.full s 0) = t := by
    intro s t
    apply TorchLean.Tensor.Internal.Rep.ext
    intro coordinate
    simp [Tensor.subSpec]
  cases state with
  | mk learningRate beta1 beta2 epsilon weightDecay firstMoment secondMoment stepCount =>
      have hwd' : weightDecay = 0 := by simpa using hwd
      simp [AdamW.update, Adam.update, hwd', scaleSpec_zero, subSpec_full_zero]

end AdamW

end Optim
