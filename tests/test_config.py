"""Manifest loading, validation and digest tests."""

from pathlib import Path
import tempfile
import unittest

from ligand_analysis.config import ConfigError, load_manifest, manifest_sha256, validate
from ligand_analysis.sources import sha256_hex

ADRB2 = Path(__file__).resolve().parents[1] / "data" / "manifests" / "adrb2.yaml"


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_adrb2_manifest_is_valid(self):
        manifest = load_manifest(ADRB2)
        self.assertEqual(manifest["receptor"]["uniprot_accession"], "P07550")
        self.assertEqual({rule["label"] for rule in manifest["label_mapping"]}, {0, 1})

    def test_labels_other_than_0_and_1_are_rejected(self):
        manifest = load_manifest(ADRB2)
        manifest["label_mapping"][0]["label"] = 2
        with self.assertRaisesRegex(ConfigError, "label_mapping/0/label"):
            validate(manifest, "source_manifest", "manifest")

    def test_gtopdb_sources_must_pin_checksum_and_release(self):
        for field in ("sha256", "release"):
            manifest = load_manifest(ADRB2)
            del manifest["sources"]["gtopdb_ligands"][field]
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, f"'{field}' is a required"):
                validate(manifest, "source_manifest", "manifest")

    def test_malformed_yaml_is_a_config_error(self):
        path = self.folder / "broken.yaml"
        path.write_text("name: [unclosed\n", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "is not valid YAML"):
            load_manifest(path)

    def test_manifest_digest_ignores_line_endings(self):
        lf, crlf = self.folder / "lf.yaml", self.folder / "crlf.yaml"
        lf.write_bytes(b"name: example\nschema_version: 1\n")
        crlf.write_bytes(b"name: example\r\nschema_version: 1\r\n")
        self.assertEqual(manifest_sha256(crlf), manifest_sha256(lf))
        self.assertEqual(manifest_sha256(lf), sha256_hex(lf.read_bytes()))


if __name__ == "__main__":
    unittest.main()
