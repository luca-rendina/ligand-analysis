"""Schema-validated TSV table tests."""

from pathlib import Path
import tempfile
import unittest

from ligand_analysis.config import ConfigError
from ligand_analysis.tables import read_table, write_table

INDEX_ROW = {"source_id": "example", "url": "https://example.invalid/data.csv", "sha256": "a" * 64, "bytes": 3,
             "media_type": "text/csv", "retrieved_utc": "2000-01-01T00:00:00+00:00"}
LIGAND_ROW = {"ligand_id": "iuphar.ligand:1", "gtopdb_ligand_id": 1, "name": "123", "ligand_type": "Metabolite",
              "approved": True, "withdrawn": False, "labelled": False, "radioactive": False, "smiles": None,
              "inchikey": None, "connectivity_key": None, "pubchem_cid": None, "pubchem_inchikey": None,
              "chembl_id": None, "identity_status": "missing_structure"}


class TableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "table.tsv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_uses_schema_column_order(self):
        write_table(self.path, [dict(reversed(INDEX_ROW.items()))], "source_index")
        self.assertEqual(self.path.read_bytes().split(b"\n")[0].decode().split("\t"), list(INDEX_ROW))
        self.assertEqual(read_table(self.path, "source_index"), [INDEX_ROW])

    def test_nulls_booleans_and_numeric_strings_round_trip(self):
        write_table(self.path, [LIGAND_ROW], "ligand")
        self.assertEqual(read_table(self.path, "ligand"), [LIGAND_ROW])

    def test_invalid_rows_are_not_written(self):
        with self.assertRaisesRegex(ConfigError, "row 2: url"):
            write_table(self.path, [{**INDEX_ROW, "url": "http://example.invalid/data.csv"}], "source_index")
        self.assertFalse(self.path.exists())

    def test_unexpected_or_missing_columns_are_rejected(self):
        self.path.write_text("source_id\tunexpected\nexample\tvalue\n", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, r"missing \['bytes'.*unexpected \['unexpected'\]"):
            read_table(self.path, "source_index")

    def test_rows_with_wrong_field_count_are_rejected(self):
        write_table(self.path, [INDEX_ROW], "source_index")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("example\textra\n")
        with self.assertRaisesRegex(ConfigError, "line 3: wrong number of fields"):
            read_table(self.path, "source_index")

    def test_invalid_values_are_rejected_on_read(self):
        write_table(self.path, [INDEX_ROW], "source_index")
        self.path.write_text(self.path.read_text(encoding="utf-8").replace("\t3\t", "\tthree\t"), encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "row 2: bytes"):
            read_table(self.path, "source_index")


if __name__ == "__main__":
    unittest.main()
