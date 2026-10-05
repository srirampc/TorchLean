/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Data.IO.Csv
public import NN.Data.IO.Npy

/-!
# Data-File Loaders

`NN.Data.IO` is the umbrella for TorchLean's deterministic CSV and NPY parsers.

The loader surface has three parts:

- `NN.Data.IO.Parsing` contains parser primitives and shared safety limits.
- `NN.Data.IO.Csv` reads compact numeric CSV tables.
- `NN.Data.IO.Npy` reads the supported NumPy `.npy` subset.

The API data-source layer turns these untyped file payloads into shape-checked tensors.
-/

@[expose] public section
