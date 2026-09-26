"""Ligand preparation: RDKit validation and standardization, stereoisomers, molscrub states, Meeko PDBQT.

Policy (explicit and bounded):
- keep the largest fragment, normalize functional groups and neutralize the parent (RDKit MolStandardize);
- keep specified stereochemistry; enumerate centres the input leaves unassigned, and exclude a
  ligand with more than ``max_stereoisomers`` stereoisomers rather than truncating silently;
- molscrub protomers and tautomers at ``ph`` with a seeded ETKDG/MMFF conformer per state (ring
  fixing optional), and exclude a ligand with more than ``max_states`` states for a stereoisomer;
- every 3D state must reproduce its stereochemistry, then Meeko writes the PDBQT.
Labels are never read: inputs are identities only.
"""

import io
from pathlib import Path

from .. import __version__
from ..fileio import write_json, write_text_atomic
from ..tables import write_table
from . import ChemistryError, distribution_version, safe_key

OUTPUT_FILES = ("ligand_states.tsv", "ligand_outcomes.tsv", "ligand_preparation.json")


class LigandPreparationError(ChemistryError):
    def __init__(self, reason, details):
        super().__init__(f"{reason}: {details}")
        self.reason, self.details = reason, details


def standardize(mol):
    """Return the standardized neutral parent and the operations that changed it."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    operations = []
    fragments = Chem.GetMolFrags(mol, asMols=True)
    parent = mol
    if len(fragments) > 1:
        parent = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
        operations.append(f"largest_fragment_of_{len(fragments)}")
    for name, step in (("normalized", rdMolStandardize.Normalizer().normalize),
                       ("neutralized", rdMolStandardize.Uncharger().uncharge)):
        before = Chem.MolToSmiles(parent)
        parent = step(parent)
        if Chem.MolToSmiles(parent) != before:
            operations.append(name)
    return parent, operations


def stereo_summary(mol):
    """(number of stereo elements, number unassigned)."""
    from rdkit import Chem

    elements = Chem.FindPotentialStereo(mol)
    unassigned = sum(1 for element in elements if element.specified == Chem.StereoSpecified.Unspecified)
    return len(elements), unassigned


def stereoisomers(parent):
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

    options = StereoEnumerationOptions(onlyUnassigned=True, unique=True, tryEmbedding=False)
    return sorted({Chem.MolToSmiles(isomer) for isomer in EnumerateStereoisomers(parent, options=options)})


def _state_smiles(mol):
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def _stereo_from_3d(mol):
    from rdkit import Chem

    copy = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(copy)
    return _state_smiles(copy)


def make_scrubber(cfg):
    from molscrub import Scrub

    return Scrub(ph_low=cfg["ph"], ph_high=cfg["ph"], etkdg_rng_seed=cfg["conformer_seed"], ff=cfg["forcefield"],
                 skip_ringfix=not cfg["ring_fix"])


def prepare_ligand(row, cfg, output_dir, scrubber=None, preparator=None):
    """Prepare one input row; return (outcome row, state rows). Failures become outcome rows."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem

    scrubber = scrubber or make_scrubber(cfg)
    preparator = preparator or MoleculePreparation()
    ligand_id = row["ligand_id"]
    outcome = {"ligand_id": ligand_id, "name": row.get("name"), "status": "failed", "reason": None, "details": None,
               "input_smiles": row["smiles"], "parent_smiles": None, "parent_inchikey": None,
               "input_inchikey": row.get("inchikey"), "inchikey_match": None, "operations": None,
               "stereocenters": None, "unassigned_stereo": None, "n_stereoisomers": 0, "n_states": 0}
    states = []
    try:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise LigandPreparationError("invalid_smiles", "RDKit cannot parse or sanitize the SMILES")
        parent, operations = standardize(mol)
        outcome["parent_smiles"] = Chem.MolToSmiles(parent)
        outcome["parent_inchikey"] = Chem.MolToInchiKey(parent)
        if outcome["input_inchikey"]:
            outcome["inchikey_match"] = outcome["parent_inchikey"] == outcome["input_inchikey"]
        outcome["operations"] = "|".join(operations) or None
        outcome["stereocenters"], outcome["unassigned_stereo"] = stereo_summary(parent)
        isomers = stereoisomers(parent)
        outcome["n_stereoisomers"] = len(isomers)
        if len(isomers) > cfg["max_stereoisomers"]:
            raise LigandPreparationError("stereoisomer_limit", f"{len(isomers)} stereoisomers exceed max_stereoisomers "
                                                      f"{cfg['max_stereoisomers']}")
        if outcome["unassigned_stereo"]:
            stereo_status = "enumerated"
        else:
            stereo_status = "specified" if outcome["stereocenters"] else "no_stereo"
        for stereo_index, isomer in enumerate(isomers, start=1):
            try:
                scrubbed = scrubber(Chem.MolFromSmiles(isomer))
            except Exception as error:
                # molscrub raises plain exceptions (e.g. RuntimeError) for embedding failures.
                raise LigandPreparationError("state_generation_failed", f"stereoisomer {stereo_index}: {error}") from error
            if not scrubbed:
                raise LigandPreparationError("no_states", f"molscrub returned no states for stereoisomer {stereo_index}")
            if len(scrubbed) > cfg["max_states"]:
                raise LigandPreparationError("state_limit", f"stereoisomer {stereo_index} has {len(scrubbed)} states, "
                                                   f"max_states is {cfg['max_states']}")
            for protomer_index, state in enumerate(sorted(scrubbed, key=_state_smiles), start=1):
                if _stereo_from_3d(state) != _state_smiles(state):
                    raise LigandPreparationError("stereo_not_preserved",
                                        f"3D coordinates of {_state_smiles(state)} imply {_stereo_from_3d(state)}")
                for conformer_index, conformer in enumerate(state.GetConformers(), start=1):
                    state_id = f"{ligand_id}#s{stereo_index}p{protomer_index}c{conformer_index}"
                    single = Chem.Mol(state, confId=conformer.GetId())
                    setups = preparator.prepare(single)
                    pdbqt, ok, error = PDBQTWriterLegacy.write_string(setups[0])
                    if not ok or len(setups) != 1:
                        raise LigandPreparationError("meeko_failed", f"{state_id}: {error or f'{len(setups)} setups'}")
                    key = safe_key(state_id)
                    states.append(_write_state(single, pdbqt, output_dir, {
                        "state_id": state_id, "ligand_id": ligand_id, "stereo_index": stereo_index,
                        "protomer_index": protomer_index, "conformer_index": conformer_index,
                        "stereo_status": stereo_status, "smiles": _state_smiles(single),
                        "inchikey": Chem.MolToInchiKey(single), "formal_charge": Chem.GetFormalCharge(single),
                        "heavy_atoms": single.GetNumHeavyAtoms(), "sdf_file": f"states/{key}.sdf",
                        "pdbqt_file": f"states/{key}.pdbqt"}))
        outcome.update(status="prepared", n_states=len(states))
    except LigandPreparationError as failure:
        outcome.update(reason=failure.reason, details=failure.details)
        for state in states:
            for name in ("sdf_file", "pdbqt_file"):
                (Path(output_dir) / state[name]).unlink(missing_ok=True)
        states = []
    return outcome, states


def _write_state(mol, pdbqt, output_dir, row):
    from rdkit import Chem

    output_dir = Path(output_dir)
    mol.SetProp("_Name", row["state_id"])
    for name in ("state_id", "ligand_id", "smiles"):
        mol.SetProp(name, row[name])
    for name in ("stereo_index", "protomer_index", "conformer_index"):
        mol.SetIntProp(name, row[name])
    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    writer.write(mol)
    writer.close()
    write_text_atomic(output_dir / row["sdf_file"], stream.getvalue())
    write_text_atomic(output_dir / row["pdbqt_file"], pdbqt)
    return row


def versions():
    import meeko
    import rdkit

    return {"rdkit": rdkit.__version__, "meeko": meeko.__version__, "molscrub": distribution_version("molscrub")}


def prepare_ligands(rows, cfg, output_dir):
    """Prepare every input ligand; write states, outcomes and a summary. Returns the summary."""
    from meeko import MoleculePreparation

    output_dir = Path(output_dir)
    (output_dir / "states").mkdir(parents=True, exist_ok=True)
    ids = [row["ligand_id"] for row in rows]
    duplicates = sorted({ligand_id for ligand_id in ids if ids.count(ligand_id) > 1})
    if duplicates:
        raise ChemistryError(f"duplicate ligand IDs in the input: {duplicates}")
    if not rows:
        raise ChemistryError("no input ligands")
    scrubber, preparator = make_scrubber(cfg), MoleculePreparation()
    outcomes, states = [], []
    for row in rows:
        outcome, ligand_states = prepare_ligand(row, cfg, output_dir, scrubber, preparator)
        outcomes.append(outcome)
        states += ligand_states
    write_table(output_dir / "ligand_states.tsv", states, "ligand_state")
    write_table(output_dir / "ligand_outcomes.tsv", outcomes, "ligand_outcome")
    summary = {
        "report_type": "ligand_preparation",
        "package_version": __version__,
        "config": cfg,
        "policy": __doc__.split("Policy (explicit and bounded):", 1)[1].strip(),
        "meeko_preparation": "MoleculePreparation() defaults: non-polar hydrogens merged, Gasteiger charges, "
                             "rotatable bonds from Meeko's flexibility model",
        "tools": versions(),
        "counts": {
            "inputs": len(rows),
            "prepared": sum(1 for row in outcomes if row["status"] == "prepared"),
            "failed": sum(1 for row in outcomes if row["status"] == "failed"),
            "states": len(states),
            "failures_by_reason": _count(row["reason"] for row in outcomes if row["reason"]),
        },
        "outputs": list(OUTPUT_FILES) + ["states/"],
    }
    write_json(output_dir / "ligand_preparation.json", summary)
    return summary


def _count(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
