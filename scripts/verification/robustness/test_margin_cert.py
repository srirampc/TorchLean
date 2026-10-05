"""Check certificate producer validation without changing its arithmetic schedule."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import export_margin_cert as exporter


class MarginCertificateTests(unittest.TestCase):
    def run_export(self, weights, example, eps="0.125"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights.json").write_text(json.dumps(weights))
            (root / "data.json").write_text(json.dumps({"examples": [example]}))
            argv = ["export", "--weights", str(root / "weights.json"),
                    "--dataset", str(root / "data.json"), "--out", str(root / "out.json"),
                    f"--eps={eps}"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                exporter.main()
            return json.loads((root / "out.json").read_text())

    def test_exact_bounds_and_first_tie(self):
        weights = {"layers.0.weight": [[2.0, -1.0], [-2.0, 1.0]], "layers.0.bias": [1.0, 0.0]}
        result = self.run_export(weights, {"x": [0.5, 0.25], "y": 0})
        self.assertEqual(result["examples"][0]["logits_lo"], [1.375, -1.125])
        self.assertEqual(result["examples"][0]["logits_hi"], [2.125, -0.375])
        self.assertEqual(exporter.argmax([1.0, 1.0]), 0)

    def test_reject_malformed_numeric_inputs(self):
        weights = {"layers.0.weight": [[1.0], [0.0]], "layers.0.bias": [0.0, 0.0]}
        valid = {"x": [0.5], "y": 0}
        for eps in ("nan", "inf", "-0.1"):
            with self.assertRaises(SystemExit):
                self.run_export(weights, valid, eps)
        for example in ({"x": [float("nan")], "y": 0}, {"x": [1.1], "y": 0},
                        {"x": [0.5], "y": 0.5}, {"x": [0.5], "y": 0, "id": -1}):
            with self.assertRaises(SystemExit):
                self.run_export(weights, example)
        for bad in ({"layers.0.weight": [[1.0]], "layers.0.bias": [0.0]},
                    {"layers.0.weight": [[1.0], []], "layers.0.bias": [0.0, 0.0]},
                    {"layers.0.weight": [[float("inf")], [0.0]], "layers.0.bias": [0.0, 0.0]}):
            with self.assertRaises(SystemExit):
                self.run_export(bad, valid)


if __name__ == "__main__":
    unittest.main()
