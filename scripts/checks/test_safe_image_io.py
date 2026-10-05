"""Enforce remote-image transport and byte limits without network access."""

import importlib.util
import io
from pathlib import Path
import types
import unittest
from unittest import mock
import urllib.request


class SafeImageTests(unittest.TestCase):
    def setUp(self):
        source = (Path(__file__).resolve().parents[1] / "verification" / "geometry3d"
                  / "safe_image_io.py")
        spec = importlib.util.spec_from_file_location("safe_image_test", source)
        self.module = importlib.util.module_from_spec(spec)
        pil = types.ModuleType("PIL")
        pil.Image = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"PIL": pil}):
            spec.loader.exec_module(self.module)

    def test_rejects_initial_and_redirected_non_https_urls(self):
        for url in ("http://example.test/image", "file:///tmp/image", "ftp://example.test/image"):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, "https"):
                    self.module.load_remote_rgb_image(url)
                handler = self.module._HttpsOnlyRedirectHandler()
                with self.assertRaisesRegex(ValueError, "https"):
                    handler.redirect_request(
                        urllib.request.Request("https://example.test/image"),
                        None, 302, "Found", {}, url,
                    )

    def test_rejects_oversize_body_before_decoding(self):
        opener = mock.Mock()
        response = io.BytesIO(b"12345")
        opener.open.return_value = response
        with mock.patch.object(self.module.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "exceeds 4 bytes"):
                self.module.load_remote_rgb_image("https://example.test/image", max_bytes=4)
        self.module.Image.open.assert_not_called()
        self.assertTrue(response.closed)

    def test_bounded_body_is_decoded_and_response_closed(self):
        response = io.BytesIO(b"1234")
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(self.module.urllib.request, "build_opener", return_value=opener):
            self.module.load_remote_rgb_image("https://example.test/image", max_bytes=4)
        self.assertTrue(response.closed)
        self.module.Image.open.return_value.__enter__.return_value.convert.assert_called_once_with("RGB")


if __name__ == "__main__":
    unittest.main()
