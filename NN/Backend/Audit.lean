/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Planner

/-!
# Backend Plan Audits

Inspection data for contract-carrying kernel plans.

The planner chooses capsules. The audit layer records what that choice means for trust boundaries:
which provider was selected, which device it targets, how its shape, layout, value, and VJP
obligations are supported, and whether a capsule has the `trustedExternal` classification. A
`checked` LibTorch capsule still executes foreign code. These classifications distinguish the
declared evidence, not whether Lean has proved the implementation. Audits are stored inside
numerical certificates and compared on replay, so they must have decidable equality.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Audit row for one selected backend kernel. -/
structure KernelAudit where
  op : BackendOp
  capsuleName : String
  provider : Provider
  device : Device
  trustLevel : TrustLevel
  vjpMode : VJPMode
  shapeContract : ContractDescriptor
  layoutContract : ContractDescriptor
  valueContract : ContractDescriptor
  vjpContract : ContractDescriptor
  numericalPolicy : NumericalPolicy
  deriving DecidableEq, Repr

namespace KernelAudit

/-- Build an audit row from a selected planner kernel. -/
def ofPlannedKernel (k : PlannedKernel) : KernelAudit :=
  { op := k.op
    capsuleName := k.capsule.name
    provider := k.capsule.provider
    device := k.capsule.device
    trustLevel := k.capsule.trustLevel
    vjpMode := k.capsule.vjpMode
    shapeContract := k.capsule.shapeContract
    layoutContract := k.capsule.layoutContract
    valueContract := k.capsule.valueContract
    vjpContract := k.capsule.vjpContract
    numericalPolicy := k.capsule.numericalPolicy }

/-- The four contract descriptors of the selected kernel, paired with the field each one fills. -/
def contracts (a : KernelAudit) : Array (ContractObligation × ContractDescriptor) :=
  #[(.shape, a.shapeContract), (.layout, a.layoutContract),
    (.value, a.valueContract), (.vjp, a.vjpContract)]

end KernelAudit

/-- Audit view of selected kernel capsules. -/
structure KernelPlanAudit where
  kernels : Array KernelAudit
  deriving DecidableEq, Repr

namespace KernelPlanAudit

/-- Capsule names selected by the plan, in plan order. -/
def capsuleNames (a : KernelPlanAudit) : Array String :=
  a.kernels.map (·.capsuleName)

/-- Whether any selected capsule has the `trustedExternal` evidence classification. -/
def hasTrustedExternal (a : KernelPlanAudit) : Bool :=
  a.kernels.any fun kernel => kernel.trustLevel == .trustedExternal

end KernelPlanAudit

namespace KernelPlan

/-- Audit a selected kernel plan. -/
def audit (p : KernelPlan) : KernelPlanAudit :=
  { kernels := p.kernels.map KernelAudit.ofPlannedKernel }

/-- Whether any selected capsule has the `trustedExternal` evidence classification. -/
def hasTrustedExternal (p : KernelPlan) : Bool :=
  p.kernels.any fun kernel => kernel.capsule.trustLevel == .trustedExternal

/-- Operation names whose selected capsules are trusted external. -/
def trustedExternalOps (p : KernelPlan) : Array String :=
  (p.kernels.filter fun kernel => kernel.capsule.trustLevel == .trustedExternal).map (·.op.name)

end KernelPlan

end Backend
end NN
