"""HTTPS dataset transfers with atomic cache publication."""

import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request
from urllib.parse import urlparse


def require_https(url: str) -> None:
    """Reject a non-HTTPS initial URL or redirect."""
    if urlparse(url).scheme != "https":
        raise SystemExit(f"refusing non-https URL: {url}")


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_https(url: str, *, timeout: float):
    require_https(url)
    return urllib.request.build_opener(HTTPSRedirectHandler()).open(url, timeout=timeout)


def file_md5(path: Path) -> str:
    """Compute the legacy checksum published with the CIFAR archive."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "md5").hexdigest()


def download_atomic(url: str, path: Path, *, timeout: float, md5: str | None = None) -> None:
    """Publish a complete, nonempty transfer only after checking its checksum."""
    require_https(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0 and (md5 is None or file_md5(path) == md5):
        return
    temporary = None
    try:
        with open_https(url, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                             delete=False) as out:
                temporary = Path(out.name)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError(f"empty dataset download: {url}")
        if md5 is not None:
            got = file_md5(temporary)
            if got != md5:
                raise ValueError(f"md5 mismatch for {path}: expected {md5}, got {got}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
