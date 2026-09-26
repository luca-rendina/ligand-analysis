"""Content-addressed source snapshots with URL, retrieval time and SHA-256 provenance.

A snapshot directory holds ``objects/<sha256>`` files and ``source_index.tsv``. It is
both the download cache and the reproducible input of offline curation.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
import urllib.error
import urllib.request

from . import __version__
from .fileio import write_bytes_atomic
from .tables import read_table, write_table

USER_AGENT = f"ligand-analysis/{__version__} (local research pipeline)"
MAX_BYTES = 512 * 1024 * 1024
RETRY_STATUS = {429, 500, 502, 503, 504}


class SourceError(RuntimeError):
    """A source could not be downloaded, verified or found in the snapshot."""


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def https_get(url, timeout=120, attempts=3):
    """Download an HTTPS URL; return (bytes, media type)."""
    if not url.startswith("https://"):
        raise SourceError(f"Refusing non-HTTPS source URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if not response.geturl().startswith("https://"):
                    raise SourceError(f"{url} redirected to non-HTTPS {response.geturl()}")
                data = response.read(MAX_BYTES + 1)
                media_type = response.headers.get_content_type()
            if len(data) > MAX_BYTES:
                raise SourceError(f"{url} is larger than {MAX_BYTES} bytes")
            return data, media_type
        except urllib.error.HTTPError as error:
            if error.code not in RETRY_STATUS or attempt == attempts:
                raise SourceError(f"HTTP {error.code} for {url}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts:
                raise SourceError(f"Cannot download {url}: {error}") from error
        time.sleep(2 ** attempt)


class Snapshot:
    """Source snapshot directory. Offline snapshots never touch the network."""

    def __init__(self, root, offline=False, refresh=False, transport=https_get):
        self.root = Path(root)
        self.offline, self.refresh, self.transport = offline, refresh, transport
        self.index_path = self.root / "source_index.tsv"
        rows = read_table(self.index_path, "source_index") if self.index_path.exists() else []
        self.entries = {row["url"]: row for row in rows}
        self.events = []
        self._changed = False

    def get(self, source_id, url, sha256=None):
        """Return the bytes for url, from the snapshot when possible."""
        entry = self.entries.get(url)
        if entry and not self.refresh:
            if sha256 and entry["sha256"] != sha256:
                raise SourceError(f"{source_id}: snapshot holds sha256 {entry['sha256']} but the manifest "
                                  f"pins {sha256}; review the manifest or rerun fetch with --refresh")
            data = self._read(entry)
            self.events.append((source_id, url, entry["sha256"], "cached"))
            return data
        if self.offline:
            raise SourceError(f"{source_id}: {url} is not in snapshot {self.root}; "
                              "run fetch without --offline (or restore it with dvc pull)")
        data, media_type = self.transport(url)
        digest = sha256_hex(data)
        if sha256 and digest != sha256:
            raise SourceError(f"{source_id}: downloaded sha256 {digest} differs from the pinned {sha256}; "
                              "the source changed, so review it before updating the manifest")
        write_bytes_atomic(self.root / "objects" / digest, data)
        self.entries[url] = {
            "source_id": source_id, "url": url, "sha256": digest, "bytes": len(data),
            "media_type": media_type,
            "retrieved_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        self._changed = True
        self.events.append((source_id, url, digest, "downloaded"))
        return data

    def entry(self, url):
        return self.entries[url]

    def _read(self, entry):
        path = self.root / "objects" / entry["sha256"]
        if not path.is_file():
            raise SourceError(f"Missing snapshot object {path}; restore it (dvc pull) or rerun fetch --refresh")
        data = path.read_bytes()
        if sha256_hex(data) != entry["sha256"]:
            raise SourceError(f"Snapshot object {path} is corrupt (SHA-256 mismatch)")
        return data

    def save(self):
        if self._changed or not self.index_path.exists():
            rows = sorted(self.entries.values(), key=lambda row: (row["source_id"], row["url"]))
            write_table(self.index_path, rows, "source_index")
            self._changed = False
