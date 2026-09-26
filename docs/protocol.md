# Molecular protocol (M2/M3 public demo)

The scientific settings live in [configs/demo.yaml](../configs/demo.yaml) and are validated by
[pipeline_config.schema.json](../src/ligand_analysis/schemas/pipeline_config.schema.json). This page records
what each stage does and why. The example is an execution demonstration, not a scientific benchmark
(see [the plan](LOCAL_PIPELINE_PLAN.md), section 9).

## Inputs

- **Ligands.** `ligand-inputs` takes the curated selected candidates of ADRB2 (P07550) with
  `selection_rank <= 10`, which gives 10 agonists and 10 antagonists. The input table holds identities only
  (`ligand_id`, `name`, `smiles`, `inchikey`). Its schema forbids other columns, so no preparation, docking or
  featurization stage can read a label.
- **Receptor.** PDB 2RH1 (inactive β2AR–T4 lysozyme fusion with the inverse agonist carazolol, 2.4 Å) is pinned
  by SHA-256 in [adrb2.yaml](../data/manifests/adrb2.yaml) and fetched into the checksummed snapshot like the
  other sources. Its binding pocket is complete in the deposited model, so no pocket reconstruction is needed.
- **Unlabeled prediction ligands.** Salbutamol, terbutaline and alprenolol are listed by GtoPdb only as partial
  agonists at β2. They have no binary label, are never used for fitting and are only predicted. The
  `prediction-inputs` command refuses any ligand that is a labelled candidate.

## Receptor preparation (`prepare-receptor`)

1. Keep chain A residues that the entry's DBREF records map to P07550: 29–230 and 263–342. DBREF gives author
   numbering equal to UniProt numbering. Each residue is compared with the UniProt sequence. The only
   difference, N187E, is the engineered mutation listed in SEQADV and lies outside the pocket.
2. Remove and list everything else: T4 lysozyme (1002–1161, UniProt P00720), waters, sulfate, butanediol,
   acetamide, cholesterol, palmitate, PEG and the glucose of chain B. The carazolol copy (CAU A:408) becomes the
   reference ligand. For alternate locations the first one is kept, and affected residues are flagged.
3. Split the chain at numbering gaps or broken peptide bonds (230/263, where T4L was inserted). Each segment
   gets charged termini. Gaps are not rebuilt; the report shows whether a break touches the pocket (none does).
4. PDBFixer adds missing heavy atoms only: the Asp29 side chain and the terminal OXT atoms. The stage stops if
   any pocket residue (heavy atom within 5 Å of carazolol) would need atoms added.
5. OpenMM `Modeller.addHydrogens` protonates at pH 7.4 (pKa rules; histidine tautomers from hydrogen bonding;
   His93/172/178/296 HID, His269 HIE). OpenMM starts the added hydrogens at random positions before minimizing
   them, so its random generator is seeded (`receptor.seed`); prepared receptors are then byte-identical between
   runs. `receptor.residue_variants` can set explicit states. Meeko must match every residue to a template.
   Nothing is deleted silently, and unknown residues are an error rather than a download of templates.
6. Outputs: `receptor.pdb`, the Meeko `receptor.pdbqt`, `reference_ligand.sdf`, `residues.tsv` (UniProt mapping,
   segment, pocket flag, added atoms, variant, template), `box.json` and `receptor_structure.json` (provenance,
   removed components, repairs, protonation, QC, tool versions).
7. Crystal ligand: carazolol bond orders come from its SMILES (PDB CCD CAU, 2S). The InChIKey computed from the
   coordinates must equal the configured key, and hydrogens are added with coordinates.
8. Docking box: centred on the carazolol bounding box, with 26 Å edges. The edges cover the orthosteric site and
   the extracellular vestibule reached by long agonists (vilanterol, indacaterol). Every ligand uses the same box,
   which leaves 8.2 Å around carazolol.

## Ligand preparation (`prepare-ligands`)

- RDKit parses and sanitizes the SMILES. RDKit MolStandardize keeps the largest fragment, normalizes and
  neutralizes, and the changes are listed per ligand. The parent InChIKey is compared with the curated one
  (20 of 20 match).
- Specified stereochemistry is kept. Unassigned stereocentres are enumerated (for example racemic isoprenaline
  gives 2 stereoisomers, fenoterol 4). A ligand with more than 4 stereoisomers is excluded with the reason
  `stereoisomer_limit` rather than truncated.
- molscrub 0.3 at pH 7.4 generates protomer and tautomer states (at most 4 per stereoisomer) and one seeded
  ETKDGv3/MMFF94s conformer per state (seed 42). The seed must be at least 1 because molscrub treats 0 as random.
  Ring fixing is off: it inverted a ring stereocentre of nadolol (tetralin cis-diol) for some seeds.
- Every 3D state must reproduce its stereochemistry (`stereo_not_preserved` otherwise). Meeko
  (`MoleculePreparation` defaults) writes the PDBQT with the SMILES remarks that pose export needs.
- State identifiers: `<ligand_id>#s<stereoisomer>p<protomer>c<conformer>`. Every input gets a row in
  `ligand_outcomes.tsv`; every state gets a row in `ligand_states.tsv`.

## Docking (`dock`)

AutoDock Vina 1.2.7 with a rigid receptor, the `vina` scoring function, exhaustiveness 8, 9 modes, a 3 kcal/mol
energy range, 1 Å minimum RMSD between modes, seed 42, and 2 threads per task in the workflow. Each state is
docked with a fresh Vina object, so results do not depend on task order. Identical scores came out with 16
threads in one process and with 2 threads per ligand task. All poses are kept as PDBQT and as SDF exported by
Meeko, which rebuilds bond orders and hydrogens from the preparation remarks. Every state gets a task row
(docked or failed) with the settings and a configuration digest.

## Pose selection and QC (`select-poses`, `redock-reference`)

- A pose is valid when (1) Meeko's export reproduces the docked state (connectivity, charges and 3D
  stereochemistry), (2) all heavy atoms lie inside the box plus 0.5 Å, and (3) no ligand–receptor heavy-atom
  pair is closer than 2.2 Å.
- Rule `best_valid_score`: among the valid poses of all states of a parent ligand, take the lowest Vina score;
  ties go to the lower state ID, then pose rank. The rule never reads labels. Alternative states and poses stay
  in the docking outputs. `pose_checks.tsv` lists every check, and `pose_selection.tsv` has one row per input
  ligand (selected, or rejected with a reason). Selected poses carry an atom map onto the prepared state.
- Redocking diagnostic: carazolol is prepared from its SMILES with the ligand pipeline (no crystal coordinates)
  and docked with the same settings. The symmetry-aware heavy-atom RMSD to the crystal pose is computed in the
  receptor frame without superposition (RDKit `CalcRMS`). The QC threshold (top-ranked pose ≤ 2.0 Å) was chosen
  before docking the labelled ligands. A failed diagnostic makes the run report fail.

## Features (`featurize`)

- **PLEC** (ODDT 0.7, Open Babel 3.2 backend, as in the legacy pipeline): ligand depth 2, protein depth 4,
  65,536 bits, 4.5 Å contact cutoff, count bits, waters ignored. The ligand is the selected pose SDF (all
  hydrogens); the protein is the Meeko receptor PDBQT (polar hydrogens) read with `protein=True`. Hydrogens are
  excluded from contacts but counted in the atom invariants. ODDT hashes with Python's `hash()` of integer
  tuples, which is deterministic but changed in Python 3.8, so exact legacy bit positions from older Python
  versions are not expected.
- ODDT's sparse output repeats an index once per count. It is converted to a CSR row with exact int32 counts,
  and every row is compared with ODDT's dense output. The dense vector is uint8 and would wrap above 255; such
  rows would be reported. The largest count in the demo is 18.
- **Morgan** (ligand-only baseline): RDKit radius 2, 2,048 bits, from the standardized parent. It does not use
  poses or the receptor.
- Each representation is written as `features.npz` (SciPy CSR), `rows.tsv` (row order, sample ID
  `<ligand>@<receptor>`, state, pose, connectivity group) and `feature_schema.json`.

## Split, models and evaluation (`split`, `train`, `evaluate`, `predict`)

- The split is persisted before fitting and shared by every representation: stratified by label, and ligands
  that share an InChIKey connectivity block stay together (test fraction 0.3, seed 0 → 7+7 train, 3+3 test). It
  is a smoke-test split; grouped scaffold validation is M5 work.
- All representations are restricted to the common cohort of samples that every representation could featurize.
  Comparisons therefore stay paired, and differential exclusions appear in the report.
- Models: `dummy` (class prior), `logistic_regression` (liblinear, C=1), `random_forest` (500 trees) and the
  corrected legacy ensemble (leave-one-out selection on the training partition, accuracy ≥ 0.6, voting with
  P(true class | member prediction) from raw counts). The ensemble's continuous class-1 score is its vote share.
  It is not a calibrated probability.
- Metrics (test partition): raw confusion counts (true classes in rows), balanced accuracy, MCC, per-class
  precision/recall/F1, ROC AUC and PR AUC for class 1 (PR AUC also for class 0), and the legacy scalar metrics
  with class 0 as positive. Undefined metrics stay empty.
- A model bundle is `model.joblib` (a pickle: load only trusted runs) plus `model.json`, which records the
  SHA-256, feature schema, training samples, split digest and settings. `predict` verifies the checksum and that
  the features were made with the same representation and parameters before predicting unlabeled rows.

## Known limitations

- Twenty ligands, one inactive structure, one conformer per state and rigid docking: the results show that the
  workflow runs end to end, not how well it generalizes.
- The two ligand classes in this demo are largely separable by chemotype (arylethanolamines versus
  aryloxypropanolamines). Ligand-only and PLEC models can therefore score perfectly on 6 test ligands without
  the structure contributing anything.
- Charged termini at the artificial T4L break and template-based protonation are approximations, but they lie
  far from the pocket.
