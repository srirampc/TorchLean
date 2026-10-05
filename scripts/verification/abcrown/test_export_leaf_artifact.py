"""Regression tests for caller-supplied property boxes during artifact export."""

import copy
import sys
from types import MappingProxyType
import unittest
from unittest import mock

from export_leaf_artifact import (
    ArtifactExportError, build_abcrown_leaf_artifact, write_abcrown_leaf_artifact,
)


class RootOverrideTests(unittest.TestCase):
    def setUp(self):
        self.leaf = {
            "x_L": [-0.5, -0.5], "x_U": [0.5, 0.5],
            "lower_bounds": [1.0], "thresholds": [0.0],
        }
        self.raw = {"domains": [self.leaf]}
        self.artifact = build_abcrown_leaf_artifact(self.raw)
        self.inputs = [self.leaf, [self.leaf], self.raw, self.artifact]

    def test_override_on_every_input_shape(self):
        for raw in self.inputs:
            with self.subTest(raw=raw):
                original = copy.deepcopy(raw)
                result = build_abcrown_leaf_artifact(raw, root_lo=[-1, -1], root_hi=[1, 1])
                self.assertEqual(result["root"], {"lo": [-1, -1], "hi": [1, 1]})
                self.assertEqual(result["leaves"], self.artifact["leaves"])
                self.assertEqual(raw, original)

    def test_partial_override_is_rejected(self):
        for raw in self.inputs:
            for bounds in [{"root_lo": [-1, -1]}, {"root_hi": [1, 1]}]:
                with self.subTest(raw=raw, bounds=bounds):
                    with self.assertRaisesRegex(ArtifactExportError, "requires both bounds"):
                        build_abcrown_leaf_artifact(raw, **bounds)

    def test_partial_embedded_root_is_rejected(self):
        for bounds in [{"root_lo": [-1, -1]}, {"root_hi": [1, 1]},
                       {"input_lo": [-1, -1]}, {"x_U": [1, 1]}]:
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ArtifactExportError, "requires both bounds"):
                    build_abcrown_leaf_artifact({**self.raw, **bounds})

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_missing_output_path_is_rejected(self):
        with self.assertRaisesRegex(ArtifactExportError, "no output path supplied"):
            write_abcrown_leaf_artifact(root_lo=[-1, -1], root_hi=[1, 1],
                                       leaves=[self.leaf])

    def test_no_override_preserves_artifact(self):
        self.assertEqual(build_abcrown_leaf_artifact(self.artifact), self.artifact)
        self.assertEqual(build_abcrown_leaf_artifact(self.raw), self.artifact)

    def test_invalid_override_is_rejected(self):
        for lo, hi in [([0], [1]), ([2, 2], [1, 1]),
                       ([float("nan"), 0], [1, 1]), ([-1, -1], [1])]:
            with self.subTest(lo=lo, hi=hi):
                with self.assertRaises(ArtifactExportError):
                    build_abcrown_leaf_artifact(self.artifact, root_lo=lo, root_hi=hi)

    def test_overflowed_witness_margin_is_rejected(self):
        leaf = {**self.leaf, "lower_bounds": [sys.float_info.max],
                "thresholds": [-sys.float_info.max]}
        with self.assertRaisesRegex(ArtifactExportError, "witness margin.*finite"):
            build_abcrown_leaf_artifact([leaf])

    def test_unrepresentable_input_number_is_exporter_error(self):
        leaf = {**self.leaf, "lower_bounds": [10**400]}
        with self.assertRaises(ArtifactExportError):
            build_abcrown_leaf_artifact([leaf])

    def test_sequence_and_mapping_inputs_follow_public_signature(self):
        leaf = MappingProxyType(self.leaf)
        result = build_abcrown_leaf_artifact((leaf,))
        self.assertEqual(result, self.artifact)


if __name__ == "__main__":
    unittest.main()
