"""Round-trip converter artifacts without downloads or optional model packages."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import torchlean_data_convert as convert


class DataConvertTests(unittest.TestCase):
    def invoke(self, *args):
        parsed = convert.build_parser().parse_args(args)
        parsed.func(parsed)

    def test_scalar_shape_and_manifest_match_written_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scalar.npy"
            np.save(source, np.asarray(1.25, dtype=np.float64))
            for dtype in ("preserve", "float32"):
                output = root / dtype
                self.invoke("tensor", "--input", str(source), "--output", str(output),
                            "--dtype", dtype, "--manifest")
                actual = output.with_suffix(".npy")
                saved = np.load(actual, allow_pickle=False)
                manifest = json.loads(actual.with_suffix(".npy.json").read_text())
                self.assertEqual(saved.shape, ())
                self.assertEqual(saved.item(), 1.25)
                self.assertEqual(manifest["shape"], [])
                self.assertEqual(manifest["file"], actual.name)
                self.assertEqual(manifest["dtype"], str(saved.dtype))

    def test_labels_reject_lossy_casts(self):
        for values, dtype in [
            ([1.5], "int64"), ([float("nan")], "float32"),
            ([float("inf")], "int64"), ([16777217], "float32"),
            ([256], "uint8"), ([-1], "uint64"),
            ([18446744073709551615, -1], "float64"),
        ]:
            with self.subTest(values=values, dtype=dtype):
                with self.assertRaises(SystemExit):
                    convert.cast_labels(values, dtype)
        self.assertEqual(convert.cast_labels([0, 2, 16777216], "float32").tolist(),
                         [0, 2, 16777216])
        self.assertEqual(convert.cast_labels([18446744073709551615], "uint64").tolist(),
                         [18446744073709551615])

    def test_csv_labels_preserve_integer_precision_and_header_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "labels.csv"
            source.write_text("description\nname,label\none,9007199254740993\n")
            self.invoke("labels", "--input", str(source), "--output", str(root / "y"),
                        "--skip-header", "1", "--label-col", "label", "--dtype", "int64")
            self.assertEqual(np.load(root / "y.npy").tolist(), [9007199254740993])

    def test_npz_archive_closed_on_success_and_ambiguous_key(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.npz"
            np.savez(source, x=[1, 2], y=[3])
            for key in ("x", None):
                archive = np.load(source, allow_pickle=False)
                with mock.patch.object(convert.np, "load", return_value=archive):
                    if key is None:
                        with self.assertRaises(SystemExit):
                            convert.load_tensor(source, key)
                    else:
                        self.assertEqual(convert.load_tensor(source, key).tolist(), [1, 2])
                self.assertIsNone(archive.zip)

    def test_fractional_label_file_does_not_publish_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "labels.npy", [0.0, 1.75])
            with self.assertRaises(SystemExit):
                self.invoke("labels", "--input", str(root / "labels.npy"),
                            "--output", str(root / "y"), "--dtype", "int64")
            self.assertFalse((root / "y.npy").exists())


if __name__ == "__main__":
    unittest.main()
