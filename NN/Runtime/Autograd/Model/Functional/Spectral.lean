/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Functional.Fourier

/-!
# One-sided spectral convolution

Both execution paths use the same weights, indexed by `[frequency, input channel, output channel]`.
Only the stored nonnegative frequencies are learned. The normalized inverse supplies the conjugate
negative frequencies and ignores imaginary DC and Nyquist coordinates.

Keeping the dense path explicit makes comparisons meaningful: changing `SpectralPath` changes the
implementation, while preserving the parameters, activation, and function being differentiated.
-/

@[expose] public section

namespace Runtime.Autograd.Model.F

open Spec TorchLean

variable {α : Type} [Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]

namespace Fourier

/-- Restore discarded frequencies as zeros before the inverse transform. -/
def padFrequencies {modes frequencies width : Nat} (h : modes ≤ frequencies)
    (x : RefTy m α [modes, width]) : m (RefTy m α [frequencies, width]) := do
  let zeros ← const (Tensor.zeros (α := α) [frequencies - modes, width])
  let padded ← concat x zeros
  pure (by simpa [Nat.add_sub_of_le h] using padded)

/--
One-sided spectral convolution composed from differentiable tensor operations.

The four real matrix products implement ordinary complex multiplication, without conjugating the
weight. Padding occurs after that multiplication, so discarded frequencies have no parameters.
Only the transforms select native primitives; the tape composes their adjoints with the channel
maps and indexing operations.
-/
def spectralConv {grid width modes : Nat} (hgrid : 0 < grid)
    (hmodes : modes ≤ grid / 2 + 1)
    (x : RefTy m α [grid, width])
    (realWeight imagWeight : RefTy m α [modes, width, width])
    (path : SpectralPath := .denseReference) :
    m (RefTy m α [grid, width]) := do
  let channels ← swapAdjacentAtDepth 0 x
  let transformed ← Runtime.Autograd.Model.F.rfft (batch := [width]) hgrid channels path
  let realChannels ← select 2 transformed ⟨0, by change 0 < 2; decide⟩
  let imagChannels ← select 2 transformed ⟨1, by change 1 < 2; decide⟩
  let realFrequencies ← swapAdjacentAtDepth 0 realChannels
  let imagFrequencies ← swapAdjacentAtDepth 0 imagChannels
  let realModes ← slice 0 modes (by omega) realFrequencies
  let imagModes ← slice 0 modes (by omega) imagFrequencies
  let realRows ← reshape (s₂ := [modes, 1, width]) realModes
    (by simp [Shape.size])
  let imagRows ← reshape (s₂ := [modes, 1, width]) imagModes
    (by simp [Shape.size])
  let rr ← matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    realRows realWeight
  let ii ← matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    imagRows imagWeight
  let ri ← matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    realRows imagWeight
  let ir ← matmul (batchA := [modes]) (batchB := [modes]) (batch := [modes])
    imagRows realWeight
  let realProduct ← sub rr ii
  let imagProduct ← add ri ir
  let realMatrix ← reshape (s₂ := [modes, width]) realProduct (by simp [Shape.size])
  let imagMatrix ← reshape (s₂ := [modes, width]) imagProduct (by simp [Shape.size])
  let realPadded ← padFrequencies hmodes realMatrix
  let imagPadded ← padFrequencies hmodes imagMatrix
  let realPlane ← reshape (s₂ := [1, grid / 2 + 1, width]) realPadded
    (by simp [Shape.size])
  let imagPlane ← reshape (s₂ := [1, grid / 2 + 1, width]) imagPadded
    (by simp [Shape.size])
  let planes ← concat realPlane imagPlane
  let frequencyPlanes ← swapAdjacentAtDepth 0 planes
  let frequencyChannels ← swapAdjacentAtDepth 1 frequencyPlanes
  let packed ← swapAdjacentAtDepth 0 frequencyChannels
  let result ← Runtime.Autograd.Model.F.irfft (batch := [width]) hgrid packed path
  swapAdjacentAtDepth 0 result

end Fourier

/--
Apply learned channel maps to the first `modes` nonnegative Fourier bins.

Inputs and outputs have shape `[grid, width]`; real and imaginary weights both have shape
`[modes, width, width]`. `denseReference` and `automatic` can share a checkpoint directly.
Both paths compose the same Lean operations; `automatic` selects native FFT primitives when
available, including their packed-real adjoints.
The arbitrary-rank full-DFT FNO uses a different parameterization and cannot share these weights.
-/
def spectralConv {grid width modes : Nat}
    (_hgrid : 0 < grid) (_hwidth : 0 < width) (hmodes : modes ≤ grid / 2 + 1)
    (x : RefTy m α [grid, width])
    (realWeight imagWeight : RefTy m α [modes, width, width])
    (path : SpectralPath := .automatic) : m (RefTy m α [grid, width]) :=
  Fourier.spectralConv _hgrid hmodes x realWeight imagWeight path

end Runtime.Autograd.Model.F
