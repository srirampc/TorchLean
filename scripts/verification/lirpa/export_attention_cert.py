#!/usr/bin/env python3
"""Export a deterministic attention-softmax interval certificate."""

from typing import Any

from common import centered_box, matmul_interval, softmax_range_interval, write_json

# Attention-softmax graph:
# input -> score projection -> softmax -> value projection.

nIn = 4
nScores = 5
nOut = 3


def seed_value_weight():
    """Return the deterministic value matrix used by the exported bound."""
    value_weight = [[float(2 + (i + j)) for j in range(nScores)] for i in range(nOut)]
    return value_weight


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3, 4]`."""
    input_center = [float(i + 1) for i in range(nIn)]
    return centered_box(input_center, eps)


def run_ibp() -> dict[str, Any]:
    """Compute the attention certificate payload consumed by Lean."""
    value_weight = seed_value_weight()
    x_lo, x_hi = seed_input_box(0.5)
    # The conservative softmax enclosure depends only on row length, not score bounds.
    prob_lo, prob_hi = softmax_range_interval(nScores)
    output_lo, output_hi = matmul_interval(value_weight, prob_lo, prob_hi)
    return {
        "graph": "attention_softmax_workflow_v1",
        "input_box": {"id": 0, "dim": nIn, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 3, "dim": nOut, "lo": output_lo, "hi": output_hi},
    }


def main():
    """Write the attention-softmax certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/attention_softmax_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")


if __name__ == "__main__":
    main()
