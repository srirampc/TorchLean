"""Regression checks for the external pendulum bound producer."""

import importlib.util
import math
from pathlib import Path
import unittest


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "NN/MLTheory/CROWN/Tactics/crown_verifier.py"
)
SPEC = importlib.util.spec_from_file_location("crown_verifier", SOURCE)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class PendulumBoundsTests(unittest.TestCase):
    def check_samples(self, lower, upper, samples):
        lo, hi = VERIFIER.pendulum_ibp([lower, -0.5], [upper, 0.75])
        for theta in samples:
            for omega in [-0.5, 0.0, 0.75]:
                actual = -9.81 * math.sin(theta) - 0.1 * omega
                self.assertLessEqual(lo[1], actual)
                self.assertLessEqual(actual, hi[1])
                self.assertLessEqual(lo[0], omega)
                self.assertLessEqual(omega, hi[0])

    def test_interior_extrema(self):
        self.check_samples(1.0, 2.0, [1.0, math.pi / 2, 2.0])
        self.check_samples(-2.0, -1.0, [-2.0, -math.pi / 2, -1.0])
        self.check_samples(-4.0, 4.0, [-4.0, -math.pi / 2, 0.0, math.pi / 2, 4.0])

    def test_central_branch_keeps_endpoint_precision(self):
        lo, hi = VERIFIER.pendulum_ibp([-0.25, 0.0], [0.5, 0.0])
        self.assertEqual(lo[1], -9.81 * math.sin(0.5))
        self.assertEqual(hi[1], -9.81 * math.sin(-0.25))
        self.check_samples(-0.25, 0.5, [-0.25, 0.0, 0.25, 0.5])

    def test_large_angles_use_global_range(self):
        self.check_samples(1.0e100, 2.0e100, [1.0e100, 1.5e100, 2.0e100])


if __name__ == "__main__":
    unittest.main()
