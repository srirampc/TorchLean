"""Device selection must preserve explicit GPU indices without loading models."""

import importlib.util
from pathlib import Path
import types
import unittest
from unittest import mock


class GeometryDeviceTests(unittest.TestCase):
    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / "verification" / "geometry3d"
                  / "export_hf_depth_box3d_cert.py")
        spec = importlib.util.spec_from_file_location("geometry_device_test", source)
        self.module = importlib.util.module_from_spec(spec)
        torch = types.ModuleType("torch")
        torch.cuda = mock.Mock()
        transformers = types.ModuleType("transformers")
        transformers.pipeline = mock.Mock()
        images = types.ModuleType("safe_image_io")
        images.load_local_rgb_image = mock.Mock()
        images.load_remote_rgb_image = mock.Mock()
        pil = types.ModuleType("PIL")
        pil.Image = mock.Mock()
        with mock.patch.dict("sys.modules", {
            "torch": torch, "numpy": types.ModuleType("numpy"),
            "transformers": transformers, "safe_image_io": images, "PIL": pil,
        }):
            spec.loader.exec_module(self.module)

    def test_explicit_devices(self):
        for name, expected in [("cpu", -1), ("cuda", 0), ("cuda:0", 0),
                               ("cuda:2", 2), ("cuda:12", 12), ("mps", "mps")]:
            with self.subTest(name=name):
                self.assertEqual(self.module.pipeline_device(name), expected)

    def test_auto_device(self):
        for available, expected in [(False, -1), (True, 0)]:
            self.module.torch.cuda.is_available.return_value = available
            self.assertEqual(self.module.pipeline_device("auto"), expected)

    def test_invalid_cuda_indices(self):
        for name in ["cuda:", "cuda:-1", "cuda:abc", "cuda:²"]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "invalid CUDA device"):
                    self.module.pipeline_device(name)


if __name__ == "__main__":
    unittest.main()
