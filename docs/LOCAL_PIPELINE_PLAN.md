# Ligand analysis: local pipeline and Copilot implementation plan

Prepared for Luca Rendina, 26 September 2026.

Repository: [luca-rendina/ligand-analysis](https://github.com/luca-rendina/ligand-analysis)

This is an implementation plan. The images, commands, directories and interfaces proposed below still need to be built. Start with milestone M0 and work through the acceptance criteria in order.

## 1. Objective and first deliverable

Build a reproducible, CPU-based, open source pipeline for classifying GPCR ligands as agonists or antagonists. Run every analysis stage locally, first with Podman and then with the same workflow and images on local Kubernetes. Introduce online data storage after the local workflow works.

The first complete example should use one human GPCR, one suitable experimental receptor structure and approximately 10 to 20 public ligands with documented labels covering both classes. It must produce prepared structures, docked poses, interaction fingerprints, trained models, predictions and an HTML report without executing notebook cells manually.

Use artificial feature matrices for software regression tests. Use real public molecules and experimentally annotated pharmacology for the molecular example. Computationally generated conformers and docking poses are useful model inputs, but do not generate biological ground truth. The small example is an execution demonstration; larger data and suitable splits are needed for scientific performance claims.

Open source preparation and docking are the baseline. Evaluate their suitability through structural checks and held-out classification experiments. A future commercial comparison should use the same data contracts and evaluation procedure.

## 2. Verified starting point

- [PR #1](https://github.com/luca-rendina/ligand-analysis/pull/1) is merged, with merge commit `5e2ebb23be2e0078f7ca91dd74962bac3f4084d7`. Its three implementation commits end at `39cb82818f24b21024e9a69d014f3db212913fa0`.
- The existing 10 regression tests were rerun successfully on the available checkout containing those changes. The full molecular workflow has not been validated.
- Preserve the corrected metric calculations, raw confusion counts for ensemble selection and voting, and continuous scores for ROC/AUC. Serialized ensembles from before the fix require retraining.
- Current labels are `0 = agonist` and `1 = antagonist`. Legacy scalar precision/recall/F1 use class 0 as positive; legacy ROC uses class 1. Preserve this explicitly during refactoring. New reports should include per-class values and identify the class associated with every score.
- The molecular preprocessing is in `code/ml_protocol/pipeline_functions.py`. It finds labels through filenames, reads `*_proc_cleaned.pdbqt` receptors and `*best.sdf` poses, and calculates ODDT PLEC fingerprints with protein depth 4, ligand depth 2 and 65,536 features.
- Acquisition logic exists in `Get receptors and ligands from iuphar.ipynb`, using IUPHAR/Guide to Pharmacology, PubChem and PDB. It needs extraction into repeatable commands with explicit inputs and state.
- The Dockerfile installs `openeye-toolkits`, along with many unpinned packages. The new baseline must remove this proprietary dependency and retain only packages that the implemented workflow uses.
- The repository does not establish the lab's full preparation, docking or pose-selection protocol. A documented replacement can be built without waiting for the full lab archive.

Existing regression command, run inside a suitable Python environment:

```sh
PYTHONPATH=code/ml_protocol MPLBACKEND=Agg python -m unittest discover -s tests -v
```

## 3. Proposed architecture

| Component | Choice | Responsibility |
|---|---|---|
| Scientific implementation | Python package with a CLI | Validate inputs and execute individual analysis stages |
| Workflow | Nextflow DSL2 | Dependencies, parallel tasks, resource allocation, logs and resume |
| Local containers | Podman | Build OCI images and run tasks locally |
| Local Kubernetes | Single-node kind with Podman as node provider | Execute the same workflow using Nextflow's Kubernetes executor |
| Dependency environments | Micromamba and committed lockfiles | Resolve native chemistry dependencies and pin tested versions |
| Ligand processing | RDKit, Scrubber/molscrub, Meeko | Standardization, molecular states, 3D preparation and docking inputs |
| Receptor processing | PDBFixer when repair is needed, then Meeko | Explicit structural cleanup, supported repairs and receptor parameterization |
| Docking | AutoDock Vina | CPU docking with recorded box, scoring, sampling and seed settings |
| Structural features | ODDT PLEC with a fixed backend | Preserve the project's existing feature family |
| Ligand-only features | RDKit Morgan fingerprints | Baseline for measuring the value of receptor information |
| Machine learning | scikit-learn | Baselines, existing ensemble, splits, evaluation and model serialization |
| Data versioning | DVC | Track selected datasets and model artifacts locally, then synchronize externally |
| Reporting | JSON, TSV/Parquet and static HTML | Inspect results without requiring a running service |

Nextflow supports Podman execution and a Kubernetes executor [S1, S2]. Keep it responsible for workflow execution, and DVC responsible for dataset/model snapshots. Do not maintain a second copy of the workflow in `dvc.yaml`.

Use the open source Nextflow engine with ordinary filesystem storage. The baseline has no dependency on hosted workflow services or Fusion. Run the controller with `nextflow run` inside a Kubernetes runner pod; `nextflow kuberun` is obsolete [S2, S13].

Podman builds the images and can host kind's node container. Inside kind, Kubernetes normally runs workloads through containerd. Load the images into that runtime explicitly. Installing Podman or using `podman kube play` alone does not create the Kubernetes environment required by this plan.

Use Linux as the execution environment. On Windows, prefer VS Code and Copilot attached to a WSL2 Linux environment, subject to the actual Podman setup. Detect the OS, architecture, CPU, RAM, disk space, virtualization and Podman connection during M0. Do not assume the paths on the desktop, in Podman Machine and inside Kubernetes refer to the same filesystem.

Start with limited CPU concurrency and no GPU requirement. Size the scientific demo from measured runtime and peak memory. Kubernetes is an execution target here; it does not add compute capacity to the laptop.

## 4. Container images

| Proposed image | Contents | First use |
|---|---|---|
| `ligand-chem` | RDKit, molscrub, Meeko, Vina, required receptor repair tools, ODDT and its selected backend, relevant project CLI commands | Molecular preparation, docking and fingerprints |
| `ligand-ml` | NumPy, SciPy, pandas, scikit-learn, plotting/report libraries and project ML commands | Regression tests, model fitting, scoring and reports |
| `ligand-runner` | Pinned Nextflow, its supported JVM and basic shell utilities | Kubernetes workflow controller |

Build only `ligand-ml` in M0. Add `ligand-chem` during the chemistry compatibility check and `ligand-runner` for Kubernetes. If ODDT's dependency requirements conflict with the chemistry environment, isolate fingerprinting in a fourth image. Do not replace PLEC silently to make installation easier.

Every image must have a committed Containerfile and a reproducible environment definition. Choose a compatible Python/tool combination through an actual preparation-to-PLEC test, then lock it. Avoid independently installing different versions of the same dependency through both pip and Conda.

Use immutable image tags tied to a code revision, record image identity in run metadata, and pin registry digests for published runs. Include the shell/process utilities required by Nextflow [S12]. Test file ownership with rootless Podman and Kubernetes. A notebook image can be added later for exploration.

Keep datasets, model outputs, downloaded databases and credentials outside image layers. Audit the existing imports before dropping unused TensorFlow, Keras, OpenEye, visualization or XGBoost dependencies. The five configured legacy classifiers should remain usable without installing unused frameworks.

## 5. Pipeline stages and data contracts

| Stage | Inputs | Required outputs and behavior |
|---|---|---|
| Fetch | Versioned source manifest | Cached original structures, molecule records and annotation responses, with source URLs, retrieval dates and SHA-256 checksums |
| Curate | Source records and explicit inclusion rules | Receptor, ligand and annotation tables; exclusions and conflicts with reasons |
| Prepare receptor | Selected structure, chain, state and preparation configuration | Prepared PDB/mmCIF, PDBQT, residue mapping, preparation metadata and structural QC |
| Prepare ligands | Molecular identities and state-generation policy | Prepared SDF and PDBQT with parent, stereoisomer, protomer, tautomer and conformer identifiers |
| Dock | Prepared receptor, ligands and box configuration | All requested poses, scores, seeds, tool versions and per-task status |
| Select and validate poses | Docking output and fixed selection rule | Selected SDF poses, atom mapping, geometry checks and rejection reasons |
| Featurize | Accepted receptor-pose pairs or ligand structures | PLEC or Morgan matrices, ordered row IDs and feature schema |
| Split | Curated molecular identities, scaffolds and annotations | Persisted train/validation/test assignments, independent of feature representation |
| Train | Training features, labels and inner split definitions | Fitted baselines/ensemble, training-only selection results and model metadata |
| Evaluate | Held-out features, labels and fitted models | Per-ligand predictions and continuous scores, metrics, raw confusion counts, ROC/PR data and report |
| Predict | New ligands, selected receptor and a saved model bundle | Predictions from the same preparation/feature configuration, without requiring labels |

Expose stages through a CLI such as `ligand-analysis fetch`, `prepare-receptors`, `prepare-ligands`, `dock`, `featurize`, `train`, `evaluate` and `predict`. Nextflow calls these commands; notebooks may call the same implementation for exploration.

A stage must read declared inputs, write to its task directory and return a meaningful exit status. Publish final outputs into a unique run directory. Record failures explicitly; the run summary must account for every requested input. An empty dataset, missing required labels or failed critical stage must not produce a success report.

Use small YAML/JSON manifests and TSV tables in Git. Use Parquet for larger metadata and compressed SciPy sparse matrices for fingerprints when appropriate. PLEC's sparse representation needs a deliberate conversion to CSR that preserves feature counts; verify equality against the existing dense output on a fixed example [S8].

Minimum metadata:

| Record | Required fields |
|---|---|
| Receptor/structure | UniProt ID, species, PDB/model accession and version, chain, residue mapping, structure source, activation state, mutations, retained components, preparation policy |
| Ligand | Source IDs, isomeric SMILES, InChIKey, stable parent identity, stereochemistry status, state/conformer identity |
| Pharmacology annotation | Ligand and receptor IDs, species, original action term, mapped label, assay context, evidence/publication/source record and conflict status |
| Pose | Parent and state IDs, receptor structure ID, docking engine/version, config hash, seed, pose rank, score and selected/rejected status |
| Feature row | Explicit sample ID, representation/version, receptor-pose linkage and parent/scaffold grouping |
| Run/model | Git commit and dirty status, image identities, dataset version/checksums, complete config, split file, seeds, tool versions and artifact paths |

Pharmacological action belongs to a ligand-receptor/context association. Do not store it as an unconditional property of a molecule. Preserve original terms such as partial agonist, inverse agonist and allosteric modulator. Exclude ambiguous categories from the initial binary demo unless a mapping is explicitly configured. Keep missing labels missing.

The preparation and docking commands should not consume class labels. Join labels to features by validated IDs at training/evaluation time. Every variant and pose of the same parent ligand must remain in the same validation group.

## 6. Molecular modelling protocol

### Ligand preparation

Use RDKit to validate structures and preserve known stereochemistry. Record salt/fragment handling and every standardization operation. Use a fixed Scrubber/molscrub configuration for protonation and tautomer enumeration, and seeded conformer generation. Set explicit limits on generated states and conformers to keep the first demo bounded.

Supply Meeko with valid 3D SDF records containing hydrogens. Meeko's ligand preparation expects these inputs; it does not replace the upstream molecular-state and conformer policy [S5, S6]. Unsupported structures must produce a recorded exclusion.

### Receptor preparation

Choose a well-resolved experimental structure with an identifiable binding site. Define chain selection, alternate locations, mutations, missing atoms/residues, waters, ions, cofactors and fusion partners in configuration. Use PDBFixer selectively for documented repairs [S7]. For the first target, prefer a structure that needs no uncertain binding-pocket reconstruction.

Record protonation assumptions and explicit residue-state overrides, especially in the pocket. Meeko uses residue templates and supports alternative residue states; template matching is not a complete pH-dependent protonation analysis [S14]. Add another open source protonation assessment tool only when the selected target needs it. Do not silently delete unmatched pocket residues or rebuild large loops.

### Docking and pose selection

Use rigid-receptor AutoDock Vina as the first protocol. Record the binding-box center and dimensions, their structural rationale, scoring function, exhaustiveness, number of modes, seed and CPU allocation. Define the site from a reference complex or documented receptor residues, using the same rule for both ligand classes.

Redock the receptor's reference ligand as a structural diagnostic. Compare symmetry-aware heavy-atom RMSD in the receptor-aligned coordinate frame, without independently fitting away the ligand's docking error. Record pocket occupancy and steric clashes. Choose a QC threshold before large runs and investigate failures instead of tuning against held-out class labels. Keep this diagnostic distinct from the classification benchmark.

Preserve all generated poses. Initially choose the best-scoring valid pose using a fixed rule independent of the known class. For multiple molecular states, predefine the selection/aggregation policy and retain the alternatives. A docking score is a pose-ranking input, not an agonist/antagonist label.

Export poses through Meeko while retaining molecular connectivity and atom mapping. Do not infer all bond orders from a generic PDBQT-to-SDF conversion [S5].

### Fingerprints and comparison with lab outputs

Retain the existing PLEC configuration initially: protein depth 4, ligand depth 2 and 65,536 features. Also pin distance cutoff, count-versus-bit behavior, water handling, hydrogen handling and the ODDT toolkit backend. The current call inherits some of these defaults, so make them explicit after checking the installed implementation [S8].

When a small lab reference becomes available, import it as a separate prepared-data input route. Compare identities, protonation, atom mapping, receptor treatment, poses, interaction patterns and fingerprints. Use a comparison table to record which settings match and which differ. Exact reproduction remains unresolved until the lab protocol is recovered; a comparable public-data workflow can proceed now.

Keep preparation and docking behind narrow backend interfaces. A future Schrödinger backend should export the same agreed prepared-structure, pose and metadata contracts. Its results form a separate experiment, with their own provenance.

## 7. Implementation milestones

| Milestone | Work | Acceptance criteria |
|---|---|---|
| **M0. Reproducible software baseline** | Start from current `main`, verify PR #1, inspect host capabilities, audit imports, introduce package/CLI/config foundations and build `ligand-ml` with Podman. Add relevant Copilot instructions and ignores. | Existing 10 tests pass inside the image. No proprietary dependency is required. A seeded synthetic feature example trains, evaluates and writes an explicit test report. Confirm package/CLI works outside the repository working directory. |
| **M1. Explicit data model and local storage** | Implement the schemas, manifest validation, local directory configuration, tiny fixtures and source adapters. Initialize DVC and a local directory remote. Create a curated candidate list for one receptor. | Labels no longer depend on filenames. Duplicate, missing and conflicting identities are reported. A pinned download can be reused from cache. DVC can restore a small tracked artifact from the local remote. |
| **M2. Open source molecular example** | Build `ligand-chem`; prepare one receptor and its reference ligand, dock, export SDF and compute PLEC. Then process approximately 10 to 20 labelled ligands. | The complete chemistry dependency chain works in the image. Outputs preserve molecular identity and 3D coordinates. Structural QC and rejection records exist. Dense/sparse feature equivalence is checked. Every input has an outcome. |
| **M3. Complete Podman workflow** | Add Nextflow modules/profiles, train simple ligand-only and PLEC baselines, expose the legacy ensemble, generate reports and support unlabeled prediction. | One documented command runs the public demo from acquisition through report. A cached rerun works without public database access. Resume reuses valid tasks, and changing a docking parameter reruns affected downstream work. Models can be reloaded and used for prediction. |
| **M4. Complete local Kubernetes workflow** | Create the kind configuration, image loading, namespace, shared storage, service account, runner pod/Job and resource settings. Run the same demo. | Multiple stages actually execute as Kubernetes worker pods. Inputs and outputs survive pod and cluster recreation. Podman/Kubernetes outputs agree within documented tolerances using the same seeds and image builds. Resume survives controller restart. |
| **M5. Meaningful local evaluation** | Expand the curated data; implement grouped/scaffold validation, training-only model selection and matched representation comparisons. Import lab examples if available. | Split leakage checks pass. All representations use identical persisted assignments. Reports show sample/scaffold counts, class support, exclusions and uncertainty. Results distinguish smoke tests from scientific benchmarks. |
| **M6. Online data distribution** | Inventory measured dataset sizes, select an external DVC remote, publish images to a registry and create a small public data release. | A fresh checkout on another machine retrieves an exact dataset version, verifies checksums and reruns the demo without the original workstation's paths or cache. |
| **M7. AlphaFold/GPCRdb comparison** | Import predicted receptor models and confidence/state metadata; run matched comparisons with the established protocol. | Experimental and predicted structures use the same ligand cohort and splits, with structural provenance and pocket QC in the report. Any improvement is measured against the ligand-only baseline. |

M0 through M4 are the immediate local delivery. M5 strengthens the scientific evaluation. M6 is the subsequent storage migration. M7 depends on the local baseline and can be completed before M6 if the models and inputs are already available locally.

Split milestones into small commits or PRs by outcome. Do not combine container migration, scientific-method changes and unrelated refactoring into one large change.

## 8. Local Kubernetes implementation requirements

kind documents Podman support with `KIND_EXPERIMENTAL_PROVIDER=podman`; rootless operation depends on host configuration [S3]. Start with a cluster and two tiny test pods sharing files before attempting chemistry.

1. Build the OCI images with Podman. Export/load the exact builds into kind, configure image pull behavior, and test that worker pods resolve the intended images without a registry. The local Podman image store is not automatically Kubernetes' image store.
2. Keep one persistent data directory outside kind's disposable node storage. Use kind's `extraMounts` to expose it to the node [S4]. The local deployment can use a static, development-only PV/PVC over this directory, with concurrent access verified for the runner and workers. It must satisfy the shared-access contract expected by Nextflow. A single-node directory is not a general multi-node shared filesystem.
3. Mount inputs, working files and outputs at consistent paths in controller and worker pods. Stage materialized input files into the shared directory; avoid symlinks pointing to a DVC cache outside mounted paths.
4. Launch the runner with a namespace-scoped service account and the permissions the selected Nextflow version needs to create and inspect worker pods and logs. Use the Kubernetes executor in this profile and disable Podman task execution there.
5. Persist the Nextflow task work directories and resume metadata alongside outputs. Preserve both during restart tests. Controller working-directory changes must not accidentally hide the run cache.
6. Bound CPU, memory and concurrent tasks. Match Vina's threads and numerical-library threads to task allocation so several pods do not each assume they own the whole machine.

Nextflow's documented filesystem route expects a shared PVC. A future multi-node deployment needs storage with genuine shared access, typically RWX, or a separately designed staging strategy [S2]. DVC remote storage does not replace the live working filesystem.

Document and test UID/GID mapping and persistent files through the actual host/VM/node path. Cluster cleanup must preserve the external data directory. Ordinary analysis pods should run the installed tools directly; they should not build images or start nested container engines.

## 9. Validation and model evaluation

Maintain three distinct validation levels:

| Level | Data | Purpose |
|---|---|---|
| Software regression | Tiny artificial feature matrices and controlled labels | Protect metrics, voting, class ordering, scoring, failures and deterministic behavior |
| Molecular smoke test | Small real public receptor/ligand example | Verify preparation, docking, feature generation, execution and reporting |
| Scientific benchmark | Larger curated receptor-specific data | Estimate performance on held-out chemistry and compare methods |

For the scientific benchmark, start with a dummy classifier, Morgan fingerprints with logistic regression or random forest, and PLEC with a comparable classifier. Include the corrected legacy ensemble as another method. This separates the contribution of structure from the contribution of a different classifier.

Persist splits before fitting. Group all states, conformers, poses, duplicated annotations and receptor-structure representations of a parent ligand together. Use scaffold groups for testing new chemistry. If pooling receptors, define separately whether the goal is new ligands for known receptors or generalization to unseen receptors.

Fit preprocessing, feature selection, hyperparameters, calibration and ensemble weights using training data and inner validation only. The legacy leave-one-out routine can remain for reproduction, but it must not bypass grouped validation in the new benchmark. Test labels must not influence receptor choice, pose choice or model selection.

Report balanced accuracy, MCC, class-specific precision/recall/F1, raw confusion counts, ROC-AUC and PR-AUC where defined. Save continuous scores with their class and score type; a decision margin or weighted ensemble score is not automatically a calibrated probability. If extending the ensemble to ROC, expose and document its continuous voting score.

Guard against folds containing only one class. With insufficient independent scaffolds, report the limitation and leave undefined metrics unset. Do not silently substitute random splitting. For larger datasets, show variation across valid grouped splits or an appropriate group-level uncertainty estimate.

Use focused tests for identity joins, molecular round trips, leakage, score orientation, cache invalidation and partial failures. CI should run small offline tests and a bounded molecular fixture. Large docking runs and downloads belong in explicit benchmark jobs. Do not require bitwise equality across different CPU architectures; record the environment and compare numerical outputs using justified tolerances.

## 10. Storage strategy

| Location | Contents |
|---|---|
| Git | Source, lockfiles, Containerfiles, workflow/config files, schemas, small manifests, DVC pointers, tests and tiny permitted fixtures |
| Local data directory | Downloaded source snapshots, prepared receptors, molecular states, poses, feature matrices and complete run outputs |
| DVC local remote | Selected versioned dataset/model snapshots, initially on a local directory, mounted drive or NAS |
| Container image store | Reusable software environments without scientific datasets |
| Online DVC remote, later | Dataset and model objects needed to reproduce designated releases |
| Image registry, later | Published OCI images referenced by immutable identity |

DVC supports filesystem directories, mounted storage, SSH and S3-compatible remotes [S9]. Begin locally. Choose the online provider only after measuring total size, growth per run, regeneration cost and expected download frequency.

The local working directory, DVC cache, local remote and Nextflow work directory have different roles. Track selected reproducible outputs, not every temporary file. Version models and a representative dataset before deciding whether all intermediate poses merit long-term retention. Changes to large compressed archives can duplicate entire objects; partition sizeable outputs by receptor/run or another useful retrieval unit.

Preserve original downloads and their checksums, curated manifests, final selected data and expensive-to-regenerate artifacts. Make temporary-work cleanup explicit. A local remote on the same disk supports version restoration but is not an independent backup.

M6 should begin with either an available SSH-backed store or a chosen S3-compatible service. Keep credentials in local configuration/environment or Kubernetes Secrets. Share a small public demonstration dataset first, with source and redistribution information in its manifest. Original lab data can remain a separate local dataset.

Online storage migration must leave the scientific stages unchanged. Transfer/versioning handles location changes; preparation, docking and evaluation consume the same materialized file contracts. Storage services may incur costs even when the analysis software is open source.

## 11. AlphaFold extension

Start by downloading existing AlphaFold DB or GPCRdb receptor models. This adds predicted structures without requiring local AlphaFold inference or a GPU. Record accession, model version, sequence mapping, confidence files and state annotations [S10, S11].

Compare three representations on the same ligands and splits: ligand-only Morgan features, experimental-structure PLEC, and predicted-structure PLEC. Add active/inactive GPCRdb models as a separate controlled experiment.

For state comparisons, process each eligible ligand against the same predefined receptor-state panel. Aggregate or concatenate state-specific features using a fixed method. Do not assign agonists only to active structures and antagonists only to inactive structures, which would leak the label into feature construction.

Align structures and map the binding site consistently. Assess confidence near the pocket and account for construct mutations and missing regions. Keep comparisons paired on the common successfully processed cohort, and report any differential exclusions. Record bound reference ligands and model/template provenance when assessing structural independence.

Local structure prediction and receptor-ligand complex prediction are later research additions with separate compute, validation and tool/model-license decisions. They are not dependencies of the first local release.

## 12. Proposed repository organization

| Path | Purpose |
|---|---|
| `src/ligand_analysis/` | CLI, acquisition, schemas, chemistry adapters, fingerprints, training and reporting |
| `code/ml_protocol/` | Existing implementation retained during migration, then reduced to compatibility wrappers or reference notebooks |
| `tests/unit/`, `tests/integration/`, `tests/fixtures/` | Small offline tests and molecular fixtures |
| `workflow/modules/`, `main.nf`, `nextflow.config` | One reusable Nextflow workflow |
| `configs/demo.yaml`, `configs/benchmark.yaml` | Scientific parameters and input manifests |
| `conf/podman.config`, `conf/k8s.config` | Execution settings separated from science settings |
| `containers/chem/`, `containers/ml/`, `containers/runner/` | Containerfiles and environment locks |
| `deploy/kind/`, `deploy/k8s/` | Local cluster, storage, service account and runner definitions |
| `data/manifests/`, `data/schemas/` | Versioned metadata and schema definitions |
| `.dvc/`, selected `*.dvc` files | Dataset/model version references |
| `docs/LOCAL_PIPELINE_PLAN.md`, `docs/protocol.md`, `docs/local-setup.md` | Roadmap, scientific method and tested setup instructions |
| `.github/copilot-instructions.md` | Short project invariants and verification commands |
| `Makefile` or equivalent small wrapper | Tested entry commands and environment checks |

Choose a configurable ignored location for bulky data/work/results. Exclude those paths from both Git and container build contexts. Add files as their milestone needs them; avoid creating an empty framework for every future extension.

## 13. Commands to implement

These are proposed user-facing commands, not commands currently provided by the repository.

| Command | Expected behavior |
|---|---|
| `make doctor` | Check host, tool versions, runtime connection, resources and configured paths |
| `make test` | Run the regression tests inside the project image |
| `make images` | Build the required images with Podman |
| `make fetch-demo` | Materialize and verify the small public input snapshot |
| `make demo-podman` | Execute the complete demo locally and print its report path |
| `make cluster-up` | Create or verify the configured local kind cluster and persistent mounts |
| `make images-load` | Load the built images into the cluster runtime |
| `make demo-k8s` | Stage the inputs, launch the controller and run the same demo on Kubernetes |
| `make data-snapshot` | Version selected data/model artifacts using DVC |
| `make clean-work` | Remove explicitly disposable work while retaining source data and designated results |

Expose the underlying Nextflow command in the docs, for example `nextflow run main.nf -profile demo,podman -params-file configs/demo.yaml`. After the initial run, test the same command with `-resume`. Kubernetes wrappers should print the controller logs, worker status and final artifact location.

## 14. Copilot handoff prompt

Copy this file into `docs/LOCAL_PIPELINE_PLAN.md`, then start a Copilot agent session with the following prompt:

```text
Read docs/LOCAL_PIPELINE_PLAN.md and inspect the current repository and local
execution environment. Implement milestone M0 first.

This project classifies GPCR ligands as agonists or antagonists. PR #1 fixed
metrics, confusion-count ensemble weighting and continuous ROC scoring and
is already merged. Confirm those changes are present and preserve the tests.

The target is one reproducible open source pipeline, first running locally
with Podman, then using the same Nextflow workflow and OCI images on local
Kubernetes. Large data stays outside Git and image layers. DVC begins with
local storage; online storage and AlphaFold integration come later.

For M0, inspect imports and the legacy Dockerfile, remove proprietary and
unused runtime requirements, choose a compatible locked ML environment,
build a Podman image, establish package/CLI/config foundations, and run the
existing regression tests plus a deterministic artificial-feature example.
Use small commits and avoid changing scientific methods in this milestone.

Add concise .github/copilot-instructions.md instructions covering the class
mapping, raw confusion counts, continuous scores, explicit identities,
open source dependencies, data separation and the verified test command.

Never infer a pharmacology label from a filename or docking score. Preserve
0=agonist and 1=antagonist and document the legacy positive-class conventions.
Do not substitute synthetic results for real molecular validation. Record
what was actually executed and which prerequisites remain unavailable.

Keep docs/progress.md updated with completed acceptance criteria, commands
run, results and the next task. Complete M0 and report its evidence before
expanding into later milestones. Ask only when a missing decision or access
actually prevents the current milestone from being completed.
```

Subsequent sessions can use: `Read docs/LOCAL_PIPELINE_PLAN.md and docs/progress.md. Implement the next incomplete milestone, preserve earlier acceptance criteria, and update progress with actual execution evidence.`

## 15. Primary references

Documentation checked on 26 September 2026. Select and lock tested software releases during implementation; documentation examples are not dependency lockfiles.

- **S1.** [Nextflow Podman execution](https://docs.seqera.io/nextflow/container/podman).
- **S2.** [Nextflow Kubernetes execution, shared storage and running in a pod](https://docs.seqera.io/nextflow/kubernetes).
- **S3.** [kind with rootless Podman](https://kind.sigs.k8s.io/docs/user/rootless/).
- **S4.** [kind configuration and extra mounts](https://kind.sigs.k8s.io/docs/user/configuration/).
- **S5.** [AutoDock Vina basic docking and Meeko export workflow](https://autodock-vina.readthedocs.io/en/latest/docking_basic.html).
- **S6.** [Meeko basic ligand preparation](https://meeko.readthedocs.io/en/develop/lig_prep_basic.html) and [Scrubber/Meeko preparation tutorial](https://meeko.readthedocs.io/en/develop/tutorial1.html).
- **S7.** [PDBFixer source and documentation](https://github.com/openmm/pdbfixer).
- **S8.** [ODDT PLEC parameters and sparse representations](https://oddt.readthedocs.io/en/latest/rst/oddt.html#oddt.fingerprints.PLEC).
- **S9.** [DVC local and external remote storage](https://doc.dvc.org/user-guide/data-management/remote-storage).
- **S10.** [AlphaFold DB data, confidence and reuse information](https://alphafold.ebi.ac.uk/faq).
- **S11.** [GPCRdb structure documentation](https://docs.gpcrdb.org/structures.html) and [state-specific models](https://gpcrdb.org/structure/homology_models).
- **S12.** [Nextflow container requirements](https://docs.seqera.io/nextflow/container).
- **S13.** [Nextflow source and Apache 2.0 license](https://github.com/nextflow-io/nextflow).
- **S14.** [Meeko receptor templates and preparation](https://meeko.readthedocs.io/en/develop/rec_overview.html).
