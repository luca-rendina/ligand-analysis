"""Pharmacology class labels and identity-based label joins."""

from numbers import Integral

AGONIST = 0
ANTAGONIST = 1
CLASS_NAMES = {AGONIST: "agonist", ANTAGONIST: "antagonist"}


class LabelJoinError(ValueError):
    """Samples cannot be labelled unambiguously from the label table."""


def join_labels(samples, labels):
    """Return one class label per sample, joined on (ligand_id, receptor_id).

    ``labels`` rows need ``ligand_id``, ``receptor_id`` and ``label`` (0 or 1).
    Labels come only from this table; filenames, docking scores, conformers and
    poses are never consulted. Missing, conflicting or invalid labels for any
    sample raise LabelJoinError naming every affected key.
    """
    table, conflicting, invalid = {}, set(), set()
    for row in labels:
        key = (row["ligand_id"], row["receptor_id"])
        label = row["label"]
        if isinstance(label, bool) or not isinstance(label, Integral) or label not in CLASS_NAMES:
            invalid.add(key)
        elif table.setdefault(key, int(label)) != label:
            conflicting.add(key)

    keys = [(sample["ligand_id"], sample["receptor_id"]) for sample in samples]
    used = set(keys)
    problems = {
        "missing": used - table.keys() - invalid,
        "conflicting": used & conflicting,
        "invalid": used & invalid,
    }
    messages = [f"{kind} label for " + ", ".join(f"{ligand}@{receptor}" for ligand, receptor in sorted(found))
                for kind, found in problems.items() if found]
    if messages:
        raise LabelJoinError("Cannot label samples: " + "; ".join(messages))
    return [table[key] for key in keys]
