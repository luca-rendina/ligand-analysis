"""Snapshot cache and HTTPS download tests; no network access is used."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from ligand_analysis.sources import Snapshot, SourceError, https_get, sha256_hex
from ligand_analysis.tables import read_table

URL = "https://example.invalid/data.csv"
DATA = b"a,b\n1,2\n"
DIGEST = sha256_hex(DATA)


def no_network(url):
    raise AssertionError(f"network used for {url}")


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "snapshot"
        self.downloads = []

    def tearDown(self):
        self.tmp.cleanup()

    def transport(self, url):
        self.downloads.append(url)
        return DATA, "text/csv"

    def download(self, sha256=None, refresh=False):
        snapshot = Snapshot(self.root, refresh=refresh, transport=self.transport)
        data = snapshot.get("example", URL, sha256)
        snapshot.save()
        return data, snapshot.events

    def offline(self):
        return Snapshot(self.root, offline=True, transport=no_network)

    def test_pinned_download_is_recorded_then_reused_offline(self):
        self.assertEqual(self.download(DIGEST), (DATA, [("example", URL, DIGEST, "downloaded")]))
        [row] = read_table(self.root / "source_index.tsv", "source_index")
        self.assertEqual((row["source_id"], row["url"], row["sha256"], row["bytes"], row["media_type"]),
                         ("example", URL, DIGEST, len(DATA), "text/csv"))
        self.assertEqual((self.root / "objects" / DIGEST).read_bytes(), DATA)
        snapshot = self.offline()
        self.assertEqual(snapshot.get("example", URL, DIGEST), DATA)
        self.assertEqual(snapshot.events, [("example", URL, DIGEST, "cached")])
        self.assertEqual(self.downloads, [URL])

    def test_refresh_downloads_again(self):
        self.download()
        _, events = self.download(refresh=True)
        self.assertEqual(events[0][3], "downloaded")
        self.assertEqual(self.downloads, [URL, URL])

    def test_download_that_differs_from_pin_is_not_stored(self):
        with self.assertRaisesRegex(SourceError, "differs from the pinned"):
            self.download("0" * 64)
        self.assertFalse(self.root.exists())

    def test_cached_object_that_differs_from_pin_needs_refresh(self):
        self.download()
        with self.assertRaisesRegex(SourceError, "--refresh"):
            self.offline().get("example", URL, "0" * 64)

    def test_offline_miss_fails(self):
        with self.assertRaisesRegex(SourceError, "not in snapshot"):
            self.offline().get("example", URL)

    def test_missing_or_corrupt_object_fails(self):
        self.download()
        path = self.root / "objects" / DIGEST
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(SourceError, "corrupt"):
            self.offline().get("example", URL)
        path.unlink()
        with self.assertRaisesRegex(SourceError, "Missing snapshot object"):
            self.offline().get("example", URL)


def fake_response(url, data=b"payload"):
    response = MagicMock()
    response.__enter__.return_value = response
    response.geturl.return_value = url
    response.read.side_effect = lambda limit: data[:limit]
    response.headers.get_content_type.return_value = "text/csv"
    return response


def http_error(code):
    return urllib.error.HTTPError(URL, code, "error", {}, None)


@patch("time.sleep")
class HttpsGetTests(unittest.TestCase):
    def test_transient_errors_are_retried(self, sleep):
        with patch("urllib.request.urlopen", side_effect=[http_error(503), fake_response(URL)]) as urlopen:
            self.assertEqual(https_get(URL), (b"payload", "text/csv"))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    def test_client_errors_are_not_retried(self, sleep):
        with (patch("urllib.request.urlopen", side_effect=http_error(404)) as urlopen,
              self.assertRaisesRegex(SourceError, "HTTP 404")):
            https_get(URL)
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_non_https_urls_and_redirects_are_refused(self, sleep):
        with self.assertRaisesRegex(SourceError, "non-HTTPS"):
            https_get("http://example.invalid/data.csv")
        with (patch("urllib.request.urlopen", return_value=fake_response("http://example.invalid/data.csv")),
              self.assertRaisesRegex(SourceError, "redirected to non-HTTPS")):
            https_get(URL)

    def test_oversized_downloads_are_refused(self, sleep):
        with (patch("urllib.request.urlopen", return_value=fake_response(URL)),
              patch("ligand_analysis.sources.MAX_BYTES", 3),
              self.assertRaisesRegex(SourceError, "larger than")):
            https_get(URL)


if __name__ == "__main__":
    unittest.main()
