/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.PayloadBridge
public import NN.Verification.Builtin.Proved.Correctness.Eval.LoweringPrefix

/-!
# Lowering Pass Payload Insertion

The forward-fragment lowering pass emits an IR node and, when the node needs external data,
records that data in the verifier `ParamStore` at the same fresh node id.  These lemmas pin down
that insertion step for every constructor of the proved forward fragment that touches the store.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin

namespace IRStep

/-- Lowering a literal constant stores its flattened tensor at the fresh IR node id. -/
theorem lowerNode_const_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (wf : Shape.WellFormed s)
    (t : Tensor α s)
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.const wf t) params ps).2.constVals.get? id =
      some (flatOfTensor (α := α) (s := s) wf t) := by
  simp [lowerNode]

/-- Lowering a parameter constant stores the selected parameter tensor at the fresh IR node id. -/
theorem lowerNode_paramConst_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (wf : Shape.WellFormed s)
    (p : Idx paramShapes s)
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.paramConst wf p) params ps).2.constVals.get? id =
      some (flatOfTensor (α := α) (s := s) wf
        (getParam (α := α) (paramShapes := paramShapes) params p)) := by
  simp [lowerNode]

/-- Lowering a linear node stores exactly the selected weight and bias tensors. -/
theorem lowerNode_linear_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape}
    (id inDim outDim : Nat)
    (w : Idx paramShapes (.dim outDim (.dim inDim .scalar)))
    (b : Idx paramShapes (.dim outDim .scalar))
    (x : Idx (Ctx inShape ss) (.dim inDim .scalar))
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := .dim outDim .scalar) id (.linear inDim outDim w b x) params ps).2.linearWB.get? id =
      some
        ({ m := outDim
           n := inDim
           w := getParam (α := α) (paramShapes := paramShapes) params w
           b := getParam (α := α) (paramShapes := paramShapes) params b } :
          NN.MLTheory.CROWN.Graph.LinParams α) := by
  simp [lowerNode]

/-- Lowering a LayerNorm node leaves no LayerNorm payload at the fresh IR node id, so the IR
evaluator falls back to unit affine parameters. -/
theorem lowerNode_layerNorm_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (op : LayerNormOperation s)
    (x : Idx (Ctx inShape ss) s)
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.layerNorm op x) params ps).2.layerNorm.get? id = none := by
  simp [lowerNode]

/-- Lowering a convolution stores the dense kernel and bias as a unit-dilation, symmetric-padding
convolution payload at the fresh IR node id. -/
theorem lowerNode_conv_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {d : Nat}
    (id inC outC : Nat) (kernelShape stride padding inSpatial : TorchLean.Tensor Nat [d])
    (hIn : inC ≠ 0)
    (hKernel : ∀ i : Fin d, kernelShape.getScalar i ≠ 0)
    (hStride : ∀ i : Fin d, stride.getScalar i ≠ 0)
    (hInfer : OpContracts.inferConvOutShape "conv" 0 inC outC
      kernelShape stride padding (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))) =
        .ok (Shape.ofList
          (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernelShape stride padding)
            (List Nat))))
    (kernel : Idx paramShapes (Shape.ofList (outC :: inC :: (Tensor.to kernelShape (List Nat)))))
    (bias : Idx paramShapes (.dim outC .scalar))
    (x : Idx (Ctx inShape ss) (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) id
        (.conv inC outC kernelShape stride padding inSpatial hIn hKernel hStride hInfer
          kernel bias x) params ps).2.convCfg.get? id =
      some (loweredConvParams (α := α) inC outC kernelShape stride padding inSpatial hKernel
        hStride (getParam (α := α) (paramShapes := paramShapes) params kernel)
        (getParam (α := α) (paramShapes := paramShapes) params bias)) := by
  simp [lowerNode, loweredConvParams]

/-- The lowered IR node for a literal constant is the corresponding payload-backed `const` node. -/
theorem lowerNode_const_node
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (wf : Shape.WellFormed s)
    (t : Tensor α s)
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.const wf t) params ps).1 =
      { id := id, parents := #[], kind := .const s, outShape := s } := by
  rfl

/-- The lowered IR node for a parameter constant is the corresponding payload-backed `const`
node. -/
theorem lowerNode_paramConst_node
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (id : Nat)
    (wf : Shape.WellFormed s)
    (p : Idx paramShapes s)
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := s) id (.paramConst wf p) params ps).1 =
      { id := id, parents := #[], kind := .const s, outShape := s } := by
  rfl

/-- The lowered IR node for a linear source node has one activation parent and external payload. -/
theorem lowerNode_linear_node
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape}
    (id inDim outDim : Nat)
    (w : Idx paramShapes (.dim outDim (.dim inDim .scalar)))
    (b : Idx paramShapes (.dim outDim .scalar))
    (x : Idx (Ctx inShape ss) (.dim inDim .scalar))
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    (lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := .dim outDim .scalar) id (.linear inDim outDim w b x) params ps).1 =
      { id := id, parents := #[x.id], kind := .linear, outShape := .dim outDim .scalar } := by
  rfl

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
