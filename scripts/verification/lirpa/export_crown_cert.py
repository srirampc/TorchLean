#!/usr/bin/env python3
"""Export a tiny transformer-encoder-style interval certificate."""
from typing import Any

from common import centered_box, layernorm_range_interval, write_json

# Tiny transformer-encoder-like graph IBP, mirroring
# `NN.Verification.LiRPA.TransformerEncoder`.

nModel = 4


def seed_input_box(eps: float = 0.5):
    """Return the input interval box centered at `[1, 2, 3, 4]`."""
    input_center = [float(i + 1) for i in range(nModel)]
    return centered_box(input_center, eps)


def run_ibp() -> dict[str, Any]:
    """Compute the transformer-like certificate payload consumed by Lean."""
    x_lo, x_hi = seed_input_box(0.5)
    # The final LayerNorm rule uses only row width. Earlier affine/residual bounds do not
    # affect this conservative result and are not serialized in the certificate.
    output_lo, output_hi = layernorm_range_interval(nModel)

    return {
        "graph": "transformer_encoder_workflow_v1",
        "input_box": {"id": 0, "dim": nModel, "lo": x_lo, "hi": x_hi},
        "result": {"node_id": 10, "dim": nModel, "lo": output_lo, "hi": output_hi}
    }


def main():
    """Write the transformer-like certificate to the bundled examples directory."""
    cert = run_ibp()
    out_path = "NN/Examples/Verification/LiRPA/transformer_encoder_cert.json"
    out = write_json(out_path, cert)
    print(f"Wrote certificate to {out}")

if __name__ == "__main__":
    main()
