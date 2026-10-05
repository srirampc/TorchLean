"""Regressions for sample counts, expression shapes, and PINN input validation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import import_burgers_shock_mat as burgers
import pinn_common as common
import train_pinn_1d as one
import train_pinn_2d as two

torch = common.torch


def args():
    return SimpleNamespace(
        steps=1, hidden_widths=[2], activation="tanh",
        collocation_points=3, initial_points=2, boundary_points=1, data_points=0,
        weight_ic=1.0, weight_bc=1.0, weight_data=1.0, nu=0.01, const=[],
        dataset_json=None, pde_expr="uxx + uyy", ic_expr="0", bc_expr="0",
        data_expr=None, out_ckpt="unused.pt", out_json="unused.json",
    )


class PinnTrainingTests(unittest.TestCase):
    def test_tensor_shape_and_scalar_gradient(self):
        like = torch.zeros(3, 1)
        self.assertEqual(common.ensure_tensor(torch.arange(3), like).shape, like.shape)
        scalar = torch.tensor(2.0, requires_grad=True)
        common.ensure_tensor(scalar, like).sum().backward()
        self.assertEqual(scalar.grad.item(), 3.0)

    def test_invalid_inputs_before_model_allocation(self):
        for field, value in (("steps", -1), ("collocation_points", 0),
                             ("boundary_points", 0), ("data_points", -1),
                             ("weight_bc", float("nan")), ("nu", float("inf"))):
            settings = args()
            setattr(settings, field, value)
            with self.subTest(field=field), patch.object(one, "build_model") as build:
                with self.assertRaises(ValueError):
                    one.train(settings)
                build.assert_not_called()
        for raw in ("u_x=2", "x=1", "a.b=1", "alpha=nan"):
            with self.assertRaises(ValueError):
                common.parse_const_flags([raw])
        self.assertEqual(common.parse_const_flags(["alpha=-2"]), {"alpha": -2.0})
        for value in (float("inf"), 1e100):
            with self.assertRaises(ValueError):
                common.PinnDataset._read_entries([{"x": value}], ["x"], torch.device("cpu"))

    def test_mixed_derivative_aliases_train(self):
        settings = args()
        settings.pde_expr = "u_xt + utx + u_tx + uxt"
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(one, "export_model") as export, contextlib.redirect_stdout(io.StringIO()):
            one.train(settings)
        self.assertTrue(all(torch.isfinite(p).all() for p in export.call_args.args[0].parameters()))

    def test_small_boundary_counts_and_disabled_dataset_data(self):
        for count in (1, 2, 3):
            settings = args()
            settings.boundary_points = count
            settings.dataset_json = "mock.json"
            dataset = common.PinnDataset(torch.device("cpu"))
            dataset.sections["data"] = torch.tensor([[0.0, 0.0, 2.0]])
            sizes = []
            original_eval = two.eval_pinn_expr

            def observe(expr, **env):
                if expr == settings.bc_expr:
                    sizes.append(env["x"].shape[0])
                return original_eval(expr, **env)

            with patch.object(torch.cuda, "is_available", return_value=False), \
                 patch.object(common.PinnDataset, "load", return_value=dataset), \
                 patch.object(dataset, "sample_columns", wraps=dataset.sample_columns) as sample, \
                 patch.object(two, "eval_pinn_expr", side_effect=observe), \
                 patch.object(two, "export_model"), contextlib.redirect_stdout(io.StringIO()):
                two.train(settings)
            self.assertEqual(sizes, [count])
            self.assertNotIn("data", [call.args[0] for call in sample.call_args_list])

    def test_full_grid_includes_initial_and_boundary_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data.json"
            x, t = np.array([-1, 0, 1]), np.array([0, 1])
            u = np.arange(6).reshape(3, 2)
            with patch.object(burgers, "_load_mat", return_value=(x, t, u)), \
                 patch("sys.argv", ["import", "--mat", "unused.mat", "--out", str(out), "--full-grid"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                burgers.main()
            payload = json.loads(out.read_text())
            self.assertEqual(len(payload["data"]), 6)
            self.assertEqual(len(payload["collocation"]), 1)
            self.assertEqual([row["u"] for row in payload["data"]], list(range(6)))


if __name__ == "__main__":
    unittest.main()
