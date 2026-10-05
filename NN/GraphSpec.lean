/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Lowering
public import NN.GraphSpec.Chain.ToDAG.Model
public import NN.GraphSpec.Chain.ToDAG.Semantics
public import NN.GraphSpec.DAG
public import NN.GraphSpec.Models
public import NN.GraphSpec.Models.MlpDeterministicInit
public import NN.GraphSpec.Models.MlpSpecEquivalence
public import NN.GraphSpec.Primitives.Spatial
public import NN.GraphSpec.Primitives.Embedding
public import NN.GraphSpec.ToSequential
/-!
# Graph Specifications
Curated umbrella import for GraphSpec.
Use this import when working with GraphSpec models, primitives, lowering, and bridge theorems:
```lean
import NN.GraphSpec
```

It gives you:

- the canonical DAG model API (`NN.GraphSpec.DAG.Term`, `NN.GraphSpec.DAG.Model`),
- the sequential authoring syntax (`NN.GraphSpec.Chain` + `>>>`) for chain models and its lowering
  into DAG (`Chain.ToDAG.Model`), with the theorem that the converted term has the direct chain
  interpretation (`Chain.ToDAG.Semantics`),
- the Spec semantics (`NN.GraphSpec.Interp.spec`) and TorchLean lowering
  (`NN.GraphSpec.Chain.toProgram`),
- sequential and DAG primitive packs. The small primitives (`linear`, `relu`, `softmax`) live in
  `Chain.Primitives`; larger packs such as `Primitives.Spatial` (convolution, pooling, flattening,
  batch normalization) and `Primitives.Embedding` sit under `NN/GraphSpec/Primitives/`,
- the GraphSpec example architectures (`NN.GraphSpec.Models`),
- the optional lowering to `Runtime.Autograd.Model.Layers.Seq` when primitives provide `toLayerM?`,
- and the model/primitive bridge theorems that connect GraphSpec syntax to Spec references.

Sequential `Chain` pipelines can be lowered to the canonical `NN.GraphSpec.DAG.Model` via
`NN.GraphSpec.LowerToDAG.Chain.toDAGTerm` and
`NN.GraphSpec.LowerToDAG.Chain.toDAGModelZeroInit`.

Umbrella re-export; the implementation lives in the imported modules.
-/
