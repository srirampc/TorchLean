"""Offline tests for atomic publication of downloaded WikiText shards."""

import importlib.util
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest import mock


class WikiTextDownloadTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "datasets" / "download_wikitext.py"
        spec = importlib.util.spec_from_file_location("download_wikitext_test", source)
        self.module = importlib.util.module_from_spec(spec)
        # Downloading bytes does not require the optional parquet reader.
        with mock.patch.object(sys, "path", [str(source.parent), *sys.path]), mock.patch.dict("sys.modules", {
            "pyarrow": types.ModuleType("pyarrow"),
            "pyarrow.parquet": types.ModuleType("pyarrow.parquet"),
        }):
            spec.loader.exec_module(self.module)
            self.io = sys.modules["download_io"]

    def test_failed_transfer_is_removed_and_retry_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard.parquet"
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"partial", OSError("interrupted")]
            with mock.patch.object(self.io, "open_https",
                                   return_value=response):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    self.module.download("https://example.test/shard", shard)
            self.assertFalse(shard.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

            response.read.side_effect = [b"complete", b""]
            with mock.patch.object(self.io, "open_https",
                                   return_value=response) as open_url:
                self.module.download("https://example.test/shard", shard)
                self.module.download("https://example.test/shard", shard)
                open_url.assert_called_once()
            self.assertEqual(shard.read_bytes(), b"complete")
            self.assertEqual(list(Path(directory).iterdir()), [shard])

    def test_failed_transfer_preserves_empty_cache_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shard.parquet"
            shard.touch()
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"partial", OSError("interrupted")]
            with mock.patch.object(self.io, "open_https",
                                   return_value=response):
                with self.assertRaises(OSError):
                    self.module.download("https://example.test/shard", shard)
            self.assertEqual(shard.read_bytes(), b"")
            self.assertEqual(list(Path(directory).iterdir()), [shard])

    def test_checksum_failure_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive"
            path.write_bytes(b"old")
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.side_effect = [b"bad", b""]
            with mock.patch.object(self.io, "open_https", return_value=response):
                with self.assertRaisesRegex(ValueError, "md5 mismatch"):
                    self.io.download_atomic("https://example.test/archive", path,
                                            timeout=1, md5="0" * 32)
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_non_https_redirect_rejected(self):
        handler = self.io.HTTPSRedirectHandler()
        with self.assertRaisesRegex(SystemExit, "non-https"):
            handler.redirect_request(None, None, 302, "", {}, "http://example.test/archive")

    def test_cache_components_and_utf8_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            row = dict(config="wiki", split="train", filename="part.parquet", url="https://example.test")
            for bad in ["..", "../outside", "/absolute", "a\\b", ""]:
                with self.assertRaisesRegex(ValueError, "cache component"):
                    self.module.shard_path(cache, dict(row, filename=bad))
            output = Path(directory) / "corpus.txt"
            table = mock.MagicMock()
            table.column.return_value.to_pylist.return_value = ["abc\n", "é!"]
            with mock.patch.object(self.module, "download"), mock.patch.object(
                    self.module.pq, "read_table", return_value=table, create=True):
                self.assertEqual(self.module.export_text([row], cache, output, 6), 6)
            self.assertTrue(output.read_text().endswith("abc\né"))
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                self.module.export_text([row], cache, output, -1)


if __name__ == "__main__":
    unittest.main()
