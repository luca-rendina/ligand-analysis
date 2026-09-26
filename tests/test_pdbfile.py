"""Fixed-column PDB parsing tests (pure Python)."""

from pathlib import Path
import unittest

from ligand_analysis.pdbfile import PdbError, parse_atoms, parse_dbref, parse_header, parse_seqadv

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "2rh1_pocket.pdb"


class PdbFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = FIXTURE.read_text(encoding="utf-8")

    def test_dbref_maps_author_numbers_to_database_positions(self):
        refs = parse_dbref(self.text)
        self.assertEqual([(ref.chain, ref.seq_begin, ref.seq_end, ref.accession, ref.db_begin) for ref in refs],
                         [("A", 1, 230, "P07550", 1), ("A", 1002, 1161, "P00720", 2), ("A", 263, 365, "P07550", 263)])
        self.assertEqual(refs[1].db_position(1004), 4)
        self.assertIsNone(refs[0].db_position(231))
        self.assertIsNone(refs[0].db_position(100, "A"))

    def test_seqadv_keeps_engineered_mutations_and_tags(self):
        records = {record.resnum: record for record in parse_seqadv(self.text)}
        self.assertEqual((records[187].resname, records[187].db_resname, records[187].db_resnum,
                          records[187].comment), ("GLU", "ASN", 187, "ENGINEERED MUTATION"))
        self.assertIsNone(records[0].db_resnum)

    def test_atoms_keep_identity_coordinates_and_elements(self):
        atoms = parse_atoms(self.text)
        first = atoms[0]
        self.assertEqual((first.record, first.name, first.resname, first.chain, first.resnum, first.element),
                         ("ATOM", "N", "CYS", "A", 106, "N"))
        self.assertAlmostEqual(first.x, -38.952)
        ligand = [atom for atom in atoms if atom.resname == "CAU"]
        self.assertEqual(len(ligand), 22)
        self.assertEqual({atom.record for atom in ligand}, {"HETATM"})

    def test_header_gives_method_and_resolution(self):
        self.assertEqual(parse_header(self.text), {"method": "X-RAY DIFFRACTION", "resolution_angstrom": 2.4})

    def test_text_without_atoms_is_rejected(self):
        with self.assertRaises(PdbError):
            parse_atoms("HEADER    EMPTY\nEND\n")


if __name__ == "__main__":
    unittest.main()
