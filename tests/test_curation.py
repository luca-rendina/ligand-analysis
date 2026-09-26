"""M1 fetch and curation tests on a tiny synthetic GtoPdb, UniProt and PubChem fixture.

Fixture IDs and names are fake. InChIKeys are format-valid and most are real keys of simple
molecules. The fixture exercises every issue type and exclusion reason; it carries no
pharmacology and is for software regression only.
"""

import collections
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from ligand_analysis import cli
from ligand_analysis.config import load_manifest, load_schema, manifest_sha256
from ligand_analysis.curation import OUTPUT_FILES, CurationError, curate, fetch
from ligand_analysis.labels import join_labels
from ligand_analysis.sources import Snapshot, sha256_hex
from ligand_analysis.tables import read_table

ROOT = Path(__file__).resolve().parents[1]
ETHANOL = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
ETHANOL_D = "LFQSCWFLJHTTHZ-MICDWDOJSA-N"  # same connectivity block as ethanol
METHANOL = "OKKJLVBELUTLKV-UHFFFAOYSA-N"
BENZENE = "UHOVQNZJYSORNB-UHFFFAOYSA-N"
TOLUENE = "YXFVVABEGXRONW-UHFFFAOYSA-N"
PHENOL = "ISWSIDIOOBJBQZ-UHFFFAOYSA-N"
ACETIC_ACID = "QTBSBXVTEAMEQO-UHFFFAOYSA-N"
ACETONE = "CSCPPACGZOOCGX-UHFFFAOYSA-N"
PROPANOL = "BDERNNFJNOPAEC-UHFFFAOYSA-N"
PROPANOL_D = "BDERNNFJNOPAEC-QYKNYGDISA-N"  # illustrative key in the propanol block
ACC = "P0TST1"
HEADER = '"# GtoPdb Version: 9999.1 - published: 2000-01-01"\n'
INTERACTION_COLUMNS = ["Target", "Target ID", "Target Gene Symbol", "Target UniProt ID", "Target Species",
                       "Ligand ID", "Ligand", "Type", "Action", "Action comment", "Selectivity", "Endogenous",
                       "Primary Target", "Affinity Units", "Affinity High", "Affinity Median", "Affinity Low",
                       "Assay Description", "Receptor Site", "Ligand Context", "PubMed ID"]
LIGAND_COLUMNS = ["Ligand ID", "Name", "Type", "Approved", "Withdrawn", "Labelled", "Radioactive",
                  "PubChem CID", "ChEMBL ID", "SMILES", "InChIKey"]

# Ligand: what it exercises
#   1, 2   eligible agonists in one connectivity block; only 1 is selected
#   3      eligible antagonist with an exact duplicate interaction row
#   4      conflicting labels             5   mapped and unmapped terms (ambiguous)
#   6      unmapped term only             7   no structure
#   8      PubChem InChIKey mismatch      9   excluded ligand type
#   10     labelled and radioactive       11  approved eligible antagonist, selected first
#   12     two ligands.csv rows           13  no ligands.csv row
#   14     malformed InChIKey             15  no PubChem InChIKey (unverified)
#   16, 17 both labels within one connectivity block
# (ligand id, Type, Action)
INTERACTIONS = [
    (1, "Agonist", "Agonist"), (2, "Agonist", "Full agonist"), (3, "Antagonist", "Antagonist"),
    (3, "Antagonist", "Antagonist"),
    (4, "Agonist", "Agonist"), (4, "Antagonist", "Antagonist"),
    (5, "Agonist", "Agonist"), (5, "Agonist", "Partial agonist"),
    (6, "Agonist", "Partial agonist"),
    (7, "Antagonist", "Antagonist"), (8, "Antagonist", "Antagonist"), (9, "Antagonist", "Antagonist"),
    (10, "Antagonist", "Antagonist"), (11, "Antagonist", "Antagonist"), (12, "Antagonist", "Antagonist"),
    (13, "Antagonist", "Antagonist"), (14, "Antagonist", "Antagonist"), (15, "Antagonist", "Antagonist"),
    (16, "Agonist", "Agonist"), (17, "Antagonist", "Antagonist"),
]
# id: (name, type, labelled/radioactive, PubChem CID, SMILES, InChIKey)
LIGANDS = {
    1: ("fake-agonist-a", "Synthetic organic", "", 101, "CCO", ETHANOL),
    2: ("fake-agonist-b", "Synthetic organic", "", 102, "[2H]OCC", ETHANOL_D),
    3: ("fake-antagonist-a", "Synthetic organic", "", 103, "c1ccccc1", BENZENE),
    4: ("fake-conflict", "Synthetic organic", "", 104, "CO", METHANOL),
    5: ("fake-ambiguous", "Synthetic organic", "", 105, "Cc1ccccc1", TOLUENE),
    6: ("fake-partial", "Synthetic organic", "", 106, "Oc1ccccc1", PHENOL),
    7: ("fake-no-structure", "Synthetic organic", "", 107, "", ""),
    8: ("fake-mismatch", "Synthetic organic", "", 108, "Oc1ccccc1", PHENOL),
    9: ("fake-peptide", "Peptide", "", 109, "CCO", ETHANOL),
    10: ("fake-labelled", "Synthetic organic", "yes", 110, "Oc1ccccc1", PHENOL),
    11: ("fake-antagonist-b", "Synthetic organic", "", 111, "Cc1ccccc1", TOLUENE),
    12: ("fake-duplicate-record", "Synthetic organic", "", 112, "CC(=O)O", ACETIC_ACID),
    14: ("fake-invalid-key", "Synthetic organic", "", 114, "C", "NOT-AN-INCHIKEY"),
    15: ("fake-unverified", "Synthetic organic", "", 115, "CC(C)=O", ACETONE),
    16: ("fake-block-agonist", "Synthetic organic", "", 116, "CCCO", PROPANOL),
    17: ("fake-block-antagonist", "Synthetic organic", "", 117, "[2H]OCCC", PROPANOL_D),
}
APPROVED = {11}
# InChIKey that PubChem returns per CID; CIDs 107, 114 and 115 come back without one.
PUBCHEM = {101: ETHANOL, 102: ETHANOL_D, 103: BENZENE, 104: METHANOL, 105: TOLUENE, 106: PHENOL,
           108: BENZENE, 109: ETHANOL, 110: PHENOL, 111: TOLUENE, 112: ACETIC_ACID, 116: PROPANOL,
           117: PROPANOL_D}


def lig(number):
    return f"iuphar.ligand:{number}"


def _csv(columns, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return (HEADER + stream.getvalue()).encode()


def fixture_files(gene="TST1"):
    interactions = [{"Target": "fake &beta;<sub>1</sub> receptor", "Target ID": "900", "Target Gene Symbol": gene,
                     "Target UniProt ID": ACC, "Target Species": "Human", "Ligand ID": str(lid),
                     "Ligand": f"lig{lid}", "Type": kind, "Action": action, "Endogenous": "false",
                     "Primary Target": "false", "PubMed ID": str(1000 + lid)}
                    for lid, kind, action in INTERACTIONS]
    interactions.append({"Target UniProt ID": "P0TST2", "Target ID": "901", "Ligand ID": "99",
                         "Type": "Agonist", "Action": "Agonist"})  # other receptor: ignored
    ligands = [{"Ligand ID": str(lid), "Name": name, "Type": kind, "Approved": "yes" if lid in APPROVED else "",
                "Labelled": flag, "Radioactive": flag, "PubChem CID": str(cid), "SMILES": smiles, "InChIKey": key}
               for lid, (name, kind, flag, cid, smiles, key) in LIGANDS.items()]
    duplicate = next(row for row in ligands if row["Ligand ID"] == "12")
    ligands.append({**duplicate, "Name": "fake-duplicate-record-copy"})
    uniprot = {"primaryAccession": ACC, "uniProtkbId": "TST1_HUMAN", "organism": {"taxonId": 9606},
               "genes": [{"geneName": {"value": "TST1"}}],
               "sequence": {"length": 10, "md5": "0" * 32, "value": "MAAAAAAAAA"}}
    return {"interactions": _csv(INTERACTION_COLUMNS, interactions), "ligands": _csv(LIGAND_COLUMNS, ligands),
            "uniprot": json.dumps(uniprot).encode()}


class FakeTransport:
    def __init__(self, files):
        self.files, self.calls = files, []

    def __call__(self, url):
        self.calls.append(url)
        if url.endswith("interactions.csv"):
            return self.files["interactions"], "text/csv"
        if url.endswith("ligands.csv"):
            return self.files["ligands"], "text/csv"
        if "uniprot" in url:
            return self.files["uniprot"], "application/json"
        cids = [int(cid) for cid in url.split("/cid/")[1].split("/")[0].split(",")]
        rows = [{"CID": cid, **({"InChIKey": PUBCHEM[cid]} if PUBCHEM.get(cid) else {})} for cid in cids]
        return json.dumps({"PropertyTable": {"Properties": rows}}).encode(), "application/json"


def no_network(url):
    raise AssertionError(f"network used for {url}")


def make_manifest(files, max_per_class=5, require_pubchem_match=True):
    return {
        "schema_version": 1, "name": "fixture",
        "receptor": {"uniprot_accession": ACC, "gene_symbol": "TST1", "species": "Human", "taxon_id": 9606,
                     "gtopdb_target_id": 900},
        "sources": {
            "gtopdb_interactions": {"url": "https://example.invalid/interactions.csv",
                                    "sha256": sha256_hex(files["interactions"]), "release": "9999.1", "license": "test"},
            "gtopdb_ligands": {"url": "https://example.invalid/ligands.csv",
                               "sha256": sha256_hex(files["ligands"]), "release": "9999.1", "license": "test"},
            "uniprot_entry": {"url": "https://example.invalid/uniprot/P0TST1.json", "license": "test"},
            "pubchem_identity": {"url": "https://example.invalid/pubchem/cid/{cids}/property/InChIKey/JSON",
                                 "license": "test"},
        },
        "label_mapping": [{"type": "Agonist", "action": "Agonist", "label": 0},
                          {"type": "Agonist", "action": "Full agonist", "label": 0},
                          {"type": "Antagonist", "action": "Antagonist", "label": 1}],
        "inclusion": {"ligand_types": ["Synthetic organic"], "exclude_labelled": True,
                      "exclude_radioactive": True, "require_pubchem_match": require_pubchem_match},
        "selection": {"max_per_class": max_per_class},
    }


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = fixture_files()
        self.manifest = make_manifest(self.files)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sources_are_downloaded_once_then_reused_offline(self):
        transport = FakeTransport(self.files)
        events = fetch(self.manifest, Snapshot(self.root, transport=transport))
        self.assertEqual([event[3] for event in events], ["downloaded"] * 4)
        again = fetch(self.manifest, Snapshot(self.root, offline=True, transport=no_network))
        self.assertEqual([event[3] for event in again], ["cached"] * 4)
        self.assertEqual(len(transport.calls), 4)
        index = read_table(self.root / "source_index.tsv", "source_index")
        self.assertEqual({row["source_id"] for row in index}, set(self.manifest["sources"]))
        for row in index:
            self.assertEqual(sha256_hex((self.root / "objects" / row["sha256"]).read_bytes()), row["sha256"])

    def test_file_release_must_match_the_manifest(self):
        self.manifest["sources"]["gtopdb_ligands"]["release"] = "9999.2"
        with self.assertRaisesRegex(CurationError, "manifest pins 9999.2"):
            fetch(self.manifest, Snapshot(self.root, transport=FakeTransport(self.files)))

    def test_invalid_json_response_is_reported(self):
        transport = FakeTransport(self.files)
        for broken in ("uniprot", "pubchem"):
            def serve(url, broken=broken):
                return (b"<html>busy</html>", "text/html") if f"/{broken}/" in url else transport(url)

            with (self.subTest(broken=broken), tempfile.TemporaryDirectory() as tmp,
                  self.assertRaisesRegex(CurationError, f"/{broken}/.* is not valid JSON")):
                fetch(self.manifest, Snapshot(Path(tmp), transport=serve))


class CurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.files = fixture_files()
        cls.snapshot_dir = root / "snapshot"
        fetch(make_manifest(cls.files), Snapshot(cls.snapshot_dir, transport=FakeTransport(cls.files)))
        cls.out = root / "curated"
        cls.report = cls.run_curate(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_curate(cls, output_dir, **options):
        snapshot = Snapshot(cls.snapshot_dir, offline=True, transport=no_network)
        return curate(make_manifest(cls.files, **options), snapshot, output_dir)

    def table(self, name, schema, folder=None):
        return read_table((folder or self.out) / f"{name}.tsv", schema)

    def candidates(self, folder=None):
        return {row["ligand_id"]: row for row in self.table("candidates", "candidate", folder)}

    def test_selection_prefers_approved_then_id_with_one_ligand_per_block(self):
        rows = self.table("candidates", "candidate")
        self.assertEqual([(row["ligand_id"], row["label"], row["selection_rank"]) for row in rows],
                         [(lig(1), 0, 1), (lig(2), 0, None), (lig(11), 1, 1), (lig(3), 1, 2)])
        candidates = self.candidates()
        self.assertEqual(candidates[lig(2)]["selection_note"], f"same connectivity block as selected {lig(1)}")
        self.assertTrue(candidates[lig(11)]["approved"])
        self.assertEqual(candidates[lig(3)]["n_annotations"], 1)  # duplicate row counted once
        self.assertEqual(candidates[lig(3)]["pubmed_ids"], "1003")
        self.assertEqual(self.report["counts"]["eligible_by_class"], {"agonist": 2, "antagonist": 2})
        self.assertEqual(self.report["counts"]["selected_by_class"], {"agonist": 1, "antagonist": 2})

    def test_class_limit_and_optional_pubchem_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_curate(Path(tmp), max_per_class=1)
            limited = self.candidates(Path(tmp))
        self.assertEqual([ligand for ligand, row in limited.items() if row["selected"]], [lig(1), lig(11)])
        self.assertEqual(limited[lig(3)]["selection_note"], "class limit of 1 reached")
        with tempfile.TemporaryDirectory() as tmp:
            self.run_curate(Path(tmp), require_pubchem_match=False)
            relaxed = self.candidates(Path(tmp))
        self.assertEqual(set(relaxed) - set(self.candidates()), {lig(15)})
        self.assertEqual(relaxed[lig(15)]["selection_rank"], 3)

    def test_every_issue_is_reported(self):
        issues = self.table("issues", "issue")
        found = collections.Counter((row["issue_type"], row["subject_id"]) for row in issues
                                    if row["issue_type"] != "duplicate_annotation")
        self.assertEqual(found, collections.Counter([
            ("conflicting_labels", lig(4)), ("ambiguous_labels", lig(5)), ("missing_ligand_record", lig(13)),
            ("duplicate_ligand_record", lig(12)), ("missing_structure", lig(7)), ("invalid_inchikey", lig(14)),
            ("pubchem_mismatch", lig(8)), ("pubchem_unverified", lig(15)), ("duplicate_structure", lig(1)),
            ("duplicate_structure", lig(5)), ("duplicate_structure", lig(6)), ("shared_connectivity", lig(1)),
            ("shared_connectivity", lig(16)), ("connectivity_label_conflict", lig(16))]))
        related = {(row["issue_type"], row["subject_id"]): row["related_ids"] for row in issues}
        self.assertEqual(related["duplicate_structure", lig(6)], f"{lig(8)}|{lig(10)}")
        self.assertEqual(related["shared_connectivity", lig(1)], f"{lig(2)}|{lig(9)}")
        self.assertEqual(related["connectivity_label_conflict", lig(16)], lig(17))
        duplicates = [row for row in issues if row["issue_type"] == "duplicate_annotation"]
        self.assertEqual([row["related_ids"] for row in duplicates], [lig(3)])
        self.assertEqual({row["issue_type"] for row in issues},
                         set(load_schema("issue")["properties"]["issue_type"]["enum"]))

    def test_every_exclusion_has_a_reason(self):
        reasons = {(row["ligand_id"], row["reason"]) for row in self.table("exclusions", "exclusion")}
        self.assertEqual(reasons, {
            (lig(4), "conflicting_label"), (lig(5), "ambiguous_label"), (lig(6), "unmapped_action"),
            (lig(7), "missing_structure"), (lig(8), "identity_mismatch"), (lig(9), "ligand_type"),
            (lig(10), "labelled"), (lig(10), "radioactive"), (lig(12), "duplicate_ligand_record"),
            (lig(13), "missing_ligand_record"), (lig(14), "invalid_inchikey"), (lig(15), "identity_unverified"),
            (lig(16), "connectivity_label_conflict"), (lig(17), "connectivity_label_conflict")})
        self.assertEqual({reason for _, reason in reasons},
                         set(load_schema("exclusion")["properties"]["reason"]["enum"]))
        self.assertEqual(self.report["counts"]["excluded_ligands"], 13)

    def test_annotations_keep_original_terms_for_this_receptor_only(self):
        rows = self.table("annotations", "annotation")
        self.assertEqual(len(rows), len(INTERACTIONS) - 1)
        self.assertEqual({row["receptor_id"] for row in rows}, {ACC})
        self.assertNotIn(lig(99), {row["ligand_id"] for row in rows})
        partial = [row for row in rows if row["original_action"] == "Partial agonist"]
        self.assertEqual(len(partial), 2)
        self.assertTrue(all(row["label"] is None and row["label_status"] == "unmapped" for row in partial))
        status = {row["ligand_id"]: row["pair_status"] for row in rows}
        self.assertEqual([status[lig(number)] for number in (3, 4, 5, 6)],
                         ["consistent", "conflicting", "ambiguous", "unmapped"])
        self.assertEqual(self.table("receptor", "receptor")[0]["name"], "fake β1 receptor")

    def test_release_header_is_parsed_without_quotes(self):
        self.assertEqual(self.report["gtopdb_release"], {"version": "9999.1", "published": "2000-01-01"})

    def test_curation_is_deterministic_and_offline(self):
        with tempfile.TemporaryDirectory() as other:
            self.run_curate(Path(other))
            for name in OUTPUT_FILES:
                self.assertEqual((Path(other) / name).read_bytes(), (self.out / name).read_bytes(), name)

    def test_receptor_identity_mismatch_fails(self):
        files = fixture_files(gene="WRONG")
        with (tempfile.TemporaryDirectory() as tmp,
              self.assertRaisesRegex(CurationError, "Receptor identity check failed")):
            curate(make_manifest(files), Snapshot(Path(tmp) / "snapshot", transport=FakeTransport(files)),
                   Path(tmp) / "out")

    def test_candidate_labels_join_by_identity_not_filename(self):
        labels = self.table("candidates", "candidate")
        # The file name suggests "agonist" but the curated annotation says antagonist.
        samples = [{"ligand_id": lig(3), "receptor_id": ACC, "pose_file": "3_agonist_pose.sdf"}]
        self.assertEqual(join_labels(samples, labels), [1])


class CurationCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        files = fixture_files()
        self.manifest_path = self.root / "fixture.yaml"
        self.manifest_path.write_text(yaml.safe_dump(make_manifest(files)), encoding="utf-8")
        fetch(make_manifest(files), Snapshot(self.root / "sources" / "fixture", transport=FakeTransport(files)))

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main([str(arg) for arg in args])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_fetch_and_curate_default_to_data_dir(self):
        code, out, _ = self.run_cli("fetch", self.manifest_path, "--data-dir", self.root, "--offline")
        self.assertEqual(code, 0)
        self.assertIn("0 downloaded, 4 reused", out)
        code, out, _ = self.run_cli("curate", self.manifest_path, "--data-dir", self.root)
        self.assertEqual(code, 0)
        self.assertIn("Curated TST1 (P0TST1), GtoPdb 9999.1", out)
        report = json.loads((self.root / "curated" / "fixture" / "curation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["manifest"], {"name": "fixture", "sha256": manifest_sha256(self.manifest_path)})

    def test_errors_are_reported_with_exit_status_1(self):
        not_yaml = self.root / "not_yaml.yaml"
        not_yaml.write_text("name: [unclosed\n", encoding="utf-8")
        incomplete = self.root / "incomplete.yaml"
        incomplete.write_text("schema_version: 1\nname: incomplete\n", encoding="utf-8")
        empty = self.root / "empty"
        cases = [(("fetch", self.manifest_path, "--snapshot-dir", empty, "--offline"), "not in snapshot"),
                 (("curate", self.manifest_path, "--snapshot-dir", empty, "--data-dir", self.root), "not in snapshot"),
                 (("curate", not_yaml, "--data-dir", self.root), "is not valid YAML"),
                 (("fetch", incomplete, "--data-dir", self.root), "is invalid")]
        for args, message in cases:
            with self.subTest(args=args[:2]):
                code, _, err = self.run_cli(*args)
                self.assertEqual(code, 1)
                self.assertIn(message, err)


class Adrb2OutputTests(unittest.TestCase):
    """The committed ADRB2 outputs; reproducing them needs the DVC snapshot (dvc pull)."""

    manifest_path = ROOT / "data" / "manifests" / "adrb2.yaml"
    snapshot_dir = ROOT / "data" / "sources" / "adrb2"
    curated_dir = ROOT / "data" / "curated" / "adrb2"

    def test_outputs_reproduce_from_the_snapshot(self):
        if not (self.snapshot_dir / "source_index.tsv").is_file():
            self.skipTest("ADRB2 source snapshot not restored; run dvc pull")
        with tempfile.TemporaryDirectory() as tmp:
            curate(load_manifest(self.manifest_path), Snapshot(self.snapshot_dir, offline=True, transport=no_network),
                   tmp, manifest_sha256=manifest_sha256(self.manifest_path))
            for name in OUTPUT_FILES:
                self.assertEqual((Path(tmp) / name).read_bytes(), (self.curated_dir / name).read_bytes(), name)

    def test_selected_candidates_meet_the_class_minimum(self):
        rows = read_table(self.curated_dir / "candidates.tsv", "candidate")
        selected = collections.Counter(row["class_name"] for row in rows if row["selected"])
        self.assertGreaterEqual(min(selected["agonist"], selected["antagonist"]), 10)
        self.assertEqual({row["receptor_id"] for row in rows}, {"P07550"})


if __name__ == "__main__":
    unittest.main()
