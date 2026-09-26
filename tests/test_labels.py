"""Class mapping and identity-based label join tests."""

import unittest

from ligand_analysis.labels import AGONIST, ANTAGONIST, CLASS_NAMES, LabelJoinError, join_labels

LABELS = [
    {"ligand_id": "L1", "receptor_id": "R1", "label": AGONIST},
    {"ligand_id": "L2", "receptor_id": "R1", "label": ANTAGONIST},
    {"ligand_id": "L2", "receptor_id": "R1", "label": ANTAGONIST},  # repeated, consistent
    {"ligand_id": "conflict", "receptor_id": "R1", "label": AGONIST},
    {"ligand_id": "conflict", "receptor_id": "R1", "label": ANTAGONIST},
    {"ligand_id": "out_of_range", "receptor_id": "R1", "label": 2},
    {"ligand_id": "boolean", "receptor_id": "R1", "label": True},
    {"ligand_id": "missing_label", "receptor_id": "R1", "label": None},
]


def sample(ligand_id, receptor_id="R1", **extra):
    return {"ligand_id": ligand_id, "receptor_id": receptor_id, **extra}


class LabelJoinTests(unittest.TestCase):
    def test_class_mapping(self):
        self.assertEqual((AGONIST, ANTAGONIST), (0, 1))
        self.assertEqual(CLASS_NAMES, {0: "agonist", 1: "antagonist"})

    def test_labels_come_from_the_table_not_from_file_names(self):
        samples = [sample("L2", pose_file="L2_agonist_best_pose.sdf"),
                   sample("L1", pose_file="L1_antagonist_best_pose.sdf"),
                   sample("L2", pose_file="L2_conformer_2.sdf")]
        self.assertEqual(join_labels(samples, LABELS), [ANTAGONIST, AGONIST, ANTAGONIST])

    def test_problems_are_reported_only_for_keys_in_use(self):
        self.assertEqual(join_labels([sample("L1")], LABELS), [AGONIST])

    def test_missing_conflicting_and_invalid_labels_raise(self):
        cases = {"conflict": "conflicting label for conflict@R1", "out_of_range": "invalid label",
                 "boolean": "invalid label", "missing_label": "invalid label", "unknown": "missing label"}
        for ligand_id, message in cases.items():
            with self.subTest(ligand_id=ligand_id), self.assertRaisesRegex(LabelJoinError, message):
                join_labels([sample(ligand_id)], LABELS)
        with self.assertRaisesRegex(LabelJoinError, "missing label for L1@R2"):
            join_labels([sample("L1", receptor_id="R2")], LABELS)

    def test_every_affected_key_is_named(self):
        with self.assertRaises(LabelJoinError) as caught:
            join_labels([sample("unknown"), sample("conflict"), sample("L1")], LABELS)
        self.assertIn("missing label for unknown@R1", str(caught.exception))
        self.assertIn("conflicting label for conflict@R1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
