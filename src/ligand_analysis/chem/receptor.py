"""Receptor preparation: chain selection, explicit repairs, protonation, Meeko PDBQT, docking box and QC.

Only residues that the entry's DBREF records map to the receptor's UniProt accession are kept,
so fusion partners (e.g. T4 lysozyme), tags, waters, ions, lipids and additives are removed and
listed. Chain breaks become separate segments with charged termini instead of rebuilt loops.
Missing heavy atoms are added by PDBFixer only outside the pocket; a pocket that needs
reconstruction stops the stage. Hydrogens follow OpenMM's pH rules unless a residue variant is
configured, and Meeko must match every residue to a template (nothing is silently deleted).
"""

import collections
import io
import itertools
from pathlib import Path
import random

import numpy as np

from .. import __version__
from ..fileio import write_json, write_text_atomic
from ..pdbfile import THREE_TO_ONE, WATER_NAMES, parse_atoms, parse_dbref, parse_header, parse_seqadv
from ..tables import write_table
from . import ChemistryError, distribution_version

PEPTIDE_BOND_MAX = 2.0
MIN_BOX_MARGIN = 2.0
DEFAULT_STATES = {"ASP": "ASP", "GLU": "GLU", "LYS": "LYS", "ARG": "ARG", "CYS": "CYS", "TYR": "TYR"}
OUTPUT_FILES = ("receptor.pdb", "receptor.pdbqt", "reference_ligand.sdf", "box.json", "residues.tsv",
                "receptor_structure.json")


def residue_label(key):
    chain, resnum, icode = key
    return f"{chain}:{resnum}{icode}"


def _first_altloc(atoms):
    """Keep blank-altloc atoms plus the first alternate location of the residue; drop repeated names."""
    altlocs = sorted({atom.altloc for atom in atoms if atom.altloc})
    wanted = {"", altlocs[0]} if altlocs else {""}
    kept, names = [], set()
    for atom in atoms:
        if atom.altloc in wanted and atom.name not in names:
            kept.append(atom)
            names.add(atom.name)
    return kept, bool(altlocs)


def _coords(atoms):
    return np.array([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)


def _pdb_line(atom):
    return atom.line[:16] + " " + atom.line[17:]


def _ranges(keys):
    """Compress residue keys of one chain into 'A:1002-1161' style ranges."""
    spans, start, previous = [], None, None
    for key in keys:
        if start is not None and key[1] == previous[1] + 1 and not key[2] and not previous[2]:
            previous = key
            continue
        if start is not None:
            spans.append((start, previous))
        start = previous = key
    if start is not None:
        spans.append((start, previous))
    return [residue_label(first) if first == last else f"{residue_label(first)}-{last[1]}{last[2]}"
            for first, last in spans]


def _describe_removed(grouped, receptor_keys, reference_key, dbrefs_all):
    polymers, heterogens = collections.defaultdict(list), collections.defaultdict(list)
    for key, residue_atoms in grouped.items():
        if key in receptor_keys or key == reference_key:
            continue
        first = residue_atoms[0]
        if first.record == "ATOM":
            ref = next((ref for ref in dbrefs_all
                        if ref.chain == key[0] and ref.db_position(key[1], key[2]) is not None), None)
            polymers[(key[0], f"{ref.database} {ref.accession}" if ref else "no DBREF")].append(key)
        else:
            heterogens[(key[0], first.resname)].append(key)
    removed = [{"kind": "polymer", "chain": chain, "source": source, "count": len(keys), "residues": _ranges(keys),
                "reason": "not mapped to the receptor accession"}
               for (chain, source), keys in sorted(polymers.items())]
    for (chain, resname), keys in sorted(heterogens.items()):
        water = resname in WATER_NAMES
        removed.append({"kind": "water" if water else "heterogen", "chain": chain, "resname": resname,
                        "count": len(keys), "residues": None if water else [residue_label(key) for key in keys],
                        "reason": "not part of the receptor model"})
    return removed


def _segments(residues):
    """Split residues into segments at numbering gaps or broken peptide bonds."""
    segments, previous = [], None
    for key, atoms in residues.items():
        if previous is not None:
            prev_key, prev_atoms = previous
            carbon = next((atom for atom in prev_atoms if atom.name == "C"), None)
            nitrogen = next((atom for atom in atoms if atom.name == "N"), None)
            consecutive = key[1] == prev_key[1] + 1 or (key[1] == prev_key[1] and key[2] != prev_key[2])
            bonded = (carbon is not None and nitrogen is not None and
                      np.linalg.norm(_coords([carbon])[0] - _coords([nitrogen])[0]) <= PEPTIDE_BOND_MAX)
            if not (consecutive and bonded):
                segments.append([])
        if not segments:
            segments.append([])
        segments[-1].append(key)
        previous = (key, atoms)
    return segments


def _reference_ligand(lines, spec):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    crystal = Chem.MolFromPDBBlock("".join(line + "\n" for line in lines) + "END\n", removeHs=False)
    template = Chem.MolFromSmiles(spec["smiles"])
    if crystal is None or template is None:
        raise ChemistryError(f"cannot read reference ligand {spec['resname']} or its SMILES")
    crystal = Chem.RemoveHs(crystal)
    if crystal.GetNumAtoms() != template.GetNumAtoms():
        raise ChemistryError(f"reference ligand {spec['resname']} has {crystal.GetNumAtoms()} heavy atoms, "
                             f"the SMILES {template.GetNumAtoms()}")
    try:
        crystal = AllChem.AssignBondOrdersFromTemplate(template, crystal)
    except ValueError as error:
        raise ChemistryError(f"reference ligand does not match its SMILES: {error}") from error
    Chem.AssignStereochemistryFrom3D(crystal)
    inchikey = Chem.MolToInchiKey(crystal)
    if inchikey != spec["inchikey"]:
        raise ChemistryError(f"reference ligand coordinates give InChIKey {inchikey}, configured {spec['inchikey']}")
    return Chem.AddHs(crystal, addCoords=True), crystal


def _box(reference_heavy, size):
    coords = reference_heavy.GetConformer().GetPositions()
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2
    half = np.asarray(size, dtype=float) / 2
    margin = float(np.min(half - np.abs(coords - center)))
    return [round(float(value), 3) for value in center], margin


def _versions():
    import meeko
    import openmm
    import rdkit

    return {"rdkit": rdkit.__version__, "meeko": meeko.__version__, "openmm": openmm.__version__,
            "pdbfixer": distribution_version("pdbfixer")}


def prepare_receptor(pdb_bytes, uniprot, manifest, cfg, output_dir):
    """Prepare the configured structure and write OUTPUT_FILES; return the structure metadata."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    from meeko.polymer import PolymerCreationError
    from openmm import app
    from pdbfixer import PDBFixer

    structure_key = cfg["structure"]
    if structure_key not in manifest.get("structures", {}):
        raise ChemistryError(f"structure {structure_key!r} is not in the manifest 'structures'")
    source = manifest["structures"][structure_key]
    accession = manifest["receptor"]["uniprot_accession"]
    chain, spec = cfg["chain"], cfg["reference_ligand"]
    text = pdb_bytes.decode("utf-8")
    atoms = parse_atoms(text)
    dbrefs_all = parse_dbref(text)
    dbrefs = [ref for ref in dbrefs_all if ref.chain == chain and ref.database == "UNP" and ref.accession == accession]
    if not dbrefs:
        raise ChemistryError(f"{source['pdb_id']} has no DBREF mapping chain {chain} to UniProt {accession}")
    sequence = uniprot["sequence"]["value"]

    def uniprot_position(key):
        return next((pos for ref in dbrefs if (pos := ref.db_position(key[1], key[2])) is not None), None)

    grouped = collections.OrderedDict()
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    receptor, altloc_residues = collections.OrderedDict(), set()
    for key, residue_atoms in grouped.items():
        if key[0] != chain or residue_atoms[0].record != "ATOM" or uniprot_position(key) is None:
            continue
        kept, has_altlocs = _first_altloc(residue_atoms)
        receptor[key] = kept
        if has_altlocs:
            altloc_residues.add(key)
    if not receptor:
        raise ChemistryError(f"no chain {chain} residues map to {accession}")
    reference_keys = [key for key, residue_atoms in grouped.items()
                      if key[0] == chain and residue_atoms[0].record == "HETATM" and residue_atoms[0].resname == spec["resname"]]
    if len(reference_keys) != 1:
        raise ChemistryError(f"expected one {spec['resname']} in chain {chain}, found {len(reference_keys)}")
    reference_key = reference_keys[0]
    reference_atoms, _ = _first_altloc(grouped[reference_key])

    residue_rows, mutations = [], []
    seqadv = {(record.resnum, record.icode): record for record in parse_seqadv(text)
              if record.chain == chain and record.accession == accession}
    for key, residue_atoms in receptor.items():
        resname = residue_atoms[0].resname
        if resname not in THREE_TO_ONE:
            raise ChemistryError(f"nonstandard residue {resname} at {residue_label(key)}; configure a replacement first")
        position = uniprot_position(key)
        if not 1 <= position <= len(sequence):
            raise ChemistryError(f"{residue_label(key)} maps to UniProt position {position} outside the sequence")
        expected = sequence[position - 1]
        status = "match" if THREE_TO_ONE[resname] == expected else "mutation"
        if status == "mutation":
            record = seqadv.get((key[1], key[2]))
            mutations.append({"residue": residue_label(key), "uniprot": f"{expected}{position}",
                              "structure": THREE_TO_ONE[resname], "seqadv": record.comment if record else None})
        residue_rows.append({"chain": key[0], "resnum": key[1], "icode": key[2] or None, "resname": resname,
                             "uniprot_position": position, "uniprot_residue": expected, "sequence_status": status,
                             "deposited_heavy_atoms": sum(1 for atom in residue_atoms if not atom.is_hydrogen),
                             "altlocs": key in altloc_residues})

    reference_heavy_coords = _coords([atom for atom in reference_atoms if not atom.is_hydrogen])
    pocket = set()
    for key, residue_atoms in receptor.items():
        heavy = _coords([atom for atom in residue_atoms if not atom.is_hydrogen])
        distances = np.linalg.norm(heavy[:, None, :] - reference_heavy_coords[None, :, :], axis=2)
        if distances.min() <= cfg["pocket_radius"]:
            pocket.add(key)
    for mutation in mutations:
        mutation["in_pocket"] = any(residue_label(key) == mutation["residue"] for key in pocket)

    segments = _segments(receptor)
    segment_of = {key: index for index, keys in enumerate(segments, start=1) for key in keys}
    breaks = []
    for before, after in itertools.pairwise(segments):
        gap = uniprot_position(after[0]) - uniprot_position(before[-1]) - 1
        breaks.append({"after": residue_label(before[-1]), "before": residue_label(after[0]), "missing_residues": gap,
                       "near_pocket": bool({before[-1], after[0]} & pocket),
                       "treatment": "separate segments with charged termini; the gap is not rebuilt"})
    lines = []
    for keys in segments:
        lines += [_pdb_line(atom) for key in keys for atom in receptor[key]]
        lines.append("TER")
    selected_pdb = "\n".join(lines + ["END"]) + "\n"

    fixer = PDBFixer(pdbfile=io.StringIO(selected_pdb))
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        raise ChemistryError(f"PDBFixer found nonstandard residues {fixer.nonstandardResidues}")
    fixer.findMissingAtoms()

    def fixer_key(residue):
        return residue.chain.id, int(residue.id), (residue.insertionCode or "").strip()

    added_atoms = {fixer_key(residue): [atom.name for atom in missing] for residue, missing in fixer.missingAtoms.items()}
    added_terminals = {fixer_key(residue): list(names) for residue, names in fixer.missingTerminals.items()}
    pocket_repairs = sorted(residue_label(key) for key in added_atoms if key in pocket)
    if pocket_repairs:
        raise ChemistryError(f"pocket residues lack heavy atoms and would need reconstruction: {pocket_repairs}")
    fixer.addMissingAtoms(seed=cfg["seed"])
    overrides = cfg["residue_variants"]
    residues = list(fixer.topology.residues())
    unknown = set(overrides) - {residue_label(fixer_key(residue)) for residue in residues}
    if unknown:
        raise ChemistryError(f"residue_variants name residues that are not in the prepared chain: {sorted(unknown)}")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    # OpenMM starts added hydrogens at positions from Python's global RNG before minimizing them.
    rng_state = random.getstate()
    random.seed(cfg["seed"])
    try:
        applied = modeller.addHydrogens(pH=cfg["ph"], variants=[overrides.get(residue_label(fixer_key(residue)))
                                                                 for residue in residues])
    finally:
        random.setstate(rng_state)
    variants = {}
    for residue, variant in zip(modeller.topology.residues(), applied, strict=True):
        variants[fixer_key(residue)] = variant or DEFAULT_STATES.get(residue.name)
    stream = io.StringIO()
    app.PDBFile.writeFile(modeller.topology, modeller.positions, stream, keepIds=True)
    receptor_pdb = stream.getvalue()

    try:
        polymer = Polymer.from_pdb_string(receptor_pdb, ResidueChemTemplates.create_from_defaults(),
                                          MoleculePreparation())
    except PolymerCreationError as error:
        raise ChemistryError(f"Meeko could not match residue templates: {error}") from error
    ignored = polymer.get_ignored_monomers()
    if ignored:
        raise ChemistryError(f"Meeko ignored residues {sorted(ignored)}")
    templates = {residue_id: monomer.residue_template_key for residue_id, monomer in polymer.monomers.items()}
    rigid_pdbqt, _flexible = PDBQTWriterLegacy.write_from_polymer(polymer)

    reference_h, reference_heavy = _reference_ligand([atom.line for atom in reference_atoms], spec)
    center, margin = _box(reference_heavy, cfg["box"]["size"])
    if margin < MIN_BOX_MARGIN:
        raise ChemistryError(f"the box leaves only {margin:.2f} A around the reference ligand; enlarge box.size")

    for row in residue_rows:
        key = (row["chain"], row["resnum"], row["icode"] or "")
        added = added_atoms.get(key, []) + added_terminals.get(key, [])
        row.update({"segment": segment_of[key], "in_pocket": key in pocket, "added_heavy_atoms": "|".join(added) or None,
                    "variant": variants.get(key), "template": templates.get(residue_label(key))})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output_dir / "receptor.pdb", receptor_pdb)
    write_text_atomic(output_dir / "receptor.pdbqt", rigid_pdbqt)
    reference_h.SetProp("_Name", f"{source['pdb_id']}:{residue_label(reference_key)}:{spec['resname']}")
    for name, value in (("name", spec["name"]), ("resname", spec["resname"]), ("pdb_id", source["pdb_id"]),
                        ("residue", residue_label(reference_key)), ("inchikey", spec["inchikey"])):
        reference_h.SetProp(name, value)
    write_text_atomic(output_dir / "reference_ligand.sdf", _sdf([reference_h]))
    structure_id = f"pdb:{source['pdb_id']}"
    box = {"structure_id": structure_id, "center": center, "size": [float(value) for value in cfg["box"]["size"]],
           "spacing": 0.375, "reference_ligand": spec["name"], "min_margin_to_reference": round(margin, 3),
           "rationale": f"centre of the {spec['name']} ({spec['resname']}) bounding box in {source['pdb_id']}; "
                        "the same box for every ligand"}
    write_json(output_dir / "box.json", box)
    write_table(output_dir / "residues.tsv", residue_rows, "receptor_residue")

    qc_failures = []
    if any(item["near_pocket"] for item in breaks):
        qc_failures.append("chain break at a pocket residue")
    undocumented = [item["residue"] for item in mutations if not item["seqadv"]]
    qc = {
        "passed": not qc_failures,
        "failures": qc_failures,
        "pocket_radius": cfg["pocket_radius"],
        "pocket_residues": [residue_label(key) for key in sorted(pocket)],
        "pocket_heavy_atoms_added": [],
        "pocket_altlocs": sorted(residue_label(key) for key in altloc_residues & pocket),
        "pocket_mutations": [item["residue"] for item in mutations if item["in_pocket"]],
        "undocumented_sequence_differences": undocumented,
        "unmatched_templates": [],
        "reference_inside_box": True,
        "reference_min_box_margin": round(margin, 3),
    }
    metadata = {
        "record_type": "receptor_structure",
        "package_version": __version__,
        "structure_id": structure_id,
        "receptor_id": accession,
        "uniprot_accession": accession,
        "uniprot_entry_name": uniprot.get("uniProtkbId"),
        "gene_symbol": manifest["receptor"]["gene_symbol"],
        "species": manifest["receptor"]["species"],
        "taxon_id": manifest["receptor"]["taxon_id"],
        "pdb_id": source["pdb_id"],
        "source": {key: source.get(key) for key in ("url", "sha256", "release", "license", "citation")},
        **parse_header(text),
        "chain": chain,
        "activation_state": cfg["activation_state"],
        "residue_mapping": {
            "method": "PDB DBREF records to UniProt",
            "segments": [{"author": [ref.seq_begin, ref.seq_end], "uniprot": [ref.db_begin, ref.db_end]} for ref in dbrefs],
            "identity": all(ref.seq_begin == ref.db_begin for ref in dbrefs),
        },
        "retained": {"residues": len(receptor), "segments": [[residue_label(keys[0]), residue_label(keys[-1])]
                                                               for keys in segments]},
        "chain_breaks": breaks,
        "mutations": mutations,
        "removed_components": _describe_removed(grouped, set(receptor), reference_key, dbrefs_all),
        "reference_ligand": {"resname": spec["resname"], "name": spec["name"], "residue": residue_label(reference_key),
                             "smiles": spec["smiles"], "inchikey": spec["inchikey"],
                             "heavy_atoms": reference_heavy.GetNumAtoms()},
        "repairs": {
            "missing_residues_rebuilt": 0,
            "added_heavy_atoms": {residue_label(key): names for key, names in sorted(added_atoms.items())},
            "added_terminal_atoms": {residue_label(key): names for key, names in sorted(added_terminals.items())},
            "pdbfixer_seed": cfg["seed"],
        },
        "protonation": {
            "ph": cfg["ph"],
            "method": "OpenMM Modeller.addHydrogens: pKa rules, histidine tautomer from hydrogen bonding",
            "overrides": overrides,
            "histidines": {residue_label(key): variant for key, variant in variants.items()
                           if variant in ("HID", "HIE", "HIP")},
            "pocket": {residue_label(key): variants[key] for key in sorted(pocket) if variants.get(key)},
            "note": "Meeko matches these hydrogens to residue templates; this is not a full pKa analysis",
        },
        "box": box,
        "qc": qc,
        "tools": _versions(),
        "outputs": list(OUTPUT_FILES),
    }
    write_json(output_dir / "receptor_structure.json", metadata)
    return metadata


def _sdf(mols):
    from rdkit import Chem

    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    for mol in mols:
        writer.write(mol)
    writer.close()
    return stream.getvalue()
