#!/usr/bin/env python3
"""Export a deterministic GRU-gate interval certificate."""
from typing import Any

from common import (
    centered_box,
    mul_down,
    mul_up,
    sigmoid_range_interval,
    tanh_range_interval,
    write_json,
)

# GRU gate graph:
# input -> gate linear -> sigmoid; input -> candidate linear -> tanh; multiply both branches.

n = 3


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3]`."""
    input_center = [float(i + 1) for i in range(n)]
    return centered_box(input_center, eps)


def ibp_mul_elem(
    x_lo: list[float],
    x_hi: list[float],
    y_lo: list[float],
    y_hi: list[float],
) -> tuple[list[float], list[float]]:
    """Propagate multiplication using all four outward-rounded endpoint products.

    This mirrors `box_mul_elem`: lower and upper products must be rounded in their respective
    directions before selecting the extrema.
    """
    lo = []
    hi = []
    for lx, ux, ly, uy in zip(x_lo, x_hi, y_lo, y_hi):
        lo.append(
            min(
                mul_down(lx, ly),
                mul_down(lx, uy),
                mul_down(ux, ly),
                mul_down(ux, uy),
            )
        )
        hi.append(
            max(
                mul_up(lx, ly),
                mul_up(lx, uy),
                mul_up(ux, ly),
                mul_up(ux, uy),
            )
        )
    return lo, hi


def run_ibp() -> dict[str, Any]:
    """Compute the GRU-gate certificate payload consumed by Lean."""
    x_lo, x_hi = seed_input_box(0.5)
    # These conservative activation ranges are independent of the preceding affine bounds.
    sigmoid_lo, sigmoid_hi = sigmoid_range_interval(n)
    tanh_lo, tanh_hi = tanh_range_interval(n)
    output_lo, output_hi = ibp_mul_elem(sigmoid_lo, sigmoid_hi, tanh_lo, tanh_hi)
    return {
        "graph": "gru_gate_workflow_v1",
        "input_box": {"id": 0, "dim": n, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 5, "dim": n, "lo": output_lo, "hi": output_hi},
    }


def main():
    """Write the GRU-gate certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/gru_gate_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
