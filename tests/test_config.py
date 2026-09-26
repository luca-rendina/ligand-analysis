"""Manifest and pipeline configuration loading, validation and digest tests."""

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from ligand_analysis.config import (
    ConfigError,
    canonical_sha256,
    load_config,
    load_manifest,
    manifest_sha256,
    validate,
)
from ligand_analysis.sources import sha256_hex

ROOT = Path(__file__).resolve().parents[1]
ADRB2 = ROOT / "data" / "manifests" / "adrb2.yaml"
DEMO = ROOT / "configs" / "demo.yaml"


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

    def test_structures_must_be_pinned(self):
        manifest = load_manifest(ADRB2)
        self.assertEqual(manifest["structures"]["2rh1"]["pdb_id"], "2RH1")
        del manifest["structures"]["2rh1"]["sha256"]
        with self.assertRaisesRegex(ConfigError, "'sha256' is a required"):
            validate(manifest, "source_manifest", "manifest")


class PipelineConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.demo = yaml.safe_load(DEMO.read_text(encoding="utf-8"))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, config, name="config.yaml"):
        path = self.folder / name
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_demo_config_is_valid_in_every_section(self):
        sections = ["ligand_inputs", "receptor", "ligand_preparation", "docking", "pose_selection", "redocking",
                    "featurization", "split", "models", "prediction"]
        loaded = load_config(DEMO, sections)
        self.assertEqual(set(loaded), set(sections))
        self.assertEqual(loaded["featurization"]["plec"],
                         {"depth_ligand": 2, "depth_protein": 4, "size": 65536, "distance_cutoff": 4.5,
                          "count_bits": True, "ignore_hoh": True, "backend": "ob"})

    def test_only_requested_sections_are_read_and_validated(self):
        config = {"docking": self.demo["docking"], "pose_selection": {**self.demo["pose_selection"], "rule": "random"}}
        path = self.write(config)
        self.assertEqual(load_config(path, ["docking"]), {"docking": self.demo["docking"]})
        with self.assertRaisesRegex(ConfigError, "pose_selection/rule"):
            load_config(path, ["pose_selection"])
        with self.assertRaisesRegex(ConfigError, "lacks the section"):
            load_config(path, ["redocking"])
        with self.assertRaisesRegex(ConfigError, "unknown configuration section"):
            load_config(path, ["dockin"])

    def test_stage_json_written_by_the_workflow_is_accepted(self):
        path = self.folder / "stage_config.json"
        path.write_text(json.dumps({"docking": self.demo["docking"]}), encoding="utf-8")
        self.assertEqual(load_config(path, ["docking"])["docking"]["exhaustiveness"], 8)

    def test_seeds_that_molscrub_would_randomise_are_rejected(self):
        self.demo["ligand_preparation"]["conformer_seed"] = 0
        with self.assertRaisesRegex(ConfigError, "conformer_seed"):
            load_config(self.write(self.demo), ["ligand_preparation"])

    def test_config_digest_ignores_key_order(self):
        self.assertEqual(canonical_sha256({"a": 1, "b": [1, 2]}), canonical_sha256({"b": [1, 2], "a": 1}))
        self.assertNotEqual(canonical_sha256({"a": 1}), canonical_sha256({"a": 2}))


if __name__ == "__main__":
    unittest.main()
