---
goal: Implement the M4 local Kubernetes workflow with kind and Nextflow
version: '1.0'
date_created: 2026-09-27
last_updated: 2026-09-27
owner: ligand-analysis maintainers
status: 'Completed'
tags: [infrastructure, kubernetes, nextflow, m4]
---

# Introduction

![Status: Completed](https://img.shields.io/badge/status-Completed-brightgreen)

Implement milestone M4 from [docs/LOCAL_PIPELINE_PLAN.md](../docs/LOCAL_PIPELINE_PLAN.md): run the existing ADRB2 demo through Nextflow's Kubernetes executor on a local, single-node kind cluster using Podman-built images. Preserve M3's Podman workflow and scientific outputs. This plan covers local development, not production or multi-node Kubernetes.

Acceptance completed on 2026-09-27 with `scripts/k8s-acceptance.sh all`: 20 distinct Kubernetes worker processes, exact Podman/Kubernetes scientific output equivalence, 3,308 persistent-file checksums unchanged across cluster recreation, and 53 cached tasks on resume. The local images were built from the uncommitted `7c04ab7-dirty` worktree; rebuild at a committed revision before publishing reproducible image artifacts.

## 1. Requirements & Constraints

- **REQ-001**: Execute the existing `main.nf` workflow with Nextflow's Kubernetes executor; at least two distinct workflow stages must run as Kubernetes worker Jobs, not in the controller or through the Podman executor.
- **REQ-002**: Use the same `ligand-ml`, `ligand-chem`, and `ligand-runner` image builds and scientific configuration as the M3 Podman demo. Load worker images into kind's containerd image store; do not require an online registry.
- **REQ-003**: Persist source snapshots, published outputs, Nextflow work directories, launch metadata, and resume cache on host storage outside the disposable kind node. They must survive runner Job and kind cluster recreation.
- **REQ-004**: Run controller and workers with one shared PVC mounted at the same container path. Validate concurrent read/write access before running Nextflow.
- **REQ-005**: Preserve label mapping `0 = agonist`, `1 = antagonist`, label-independent preparation/docking, shared persisted split, training-only model selection, raw confusion counts, and continuous class-1 ROC scores. Do not change scientific parameters to make Kubernetes results match.
- **REQ-006**: Record the source revision and exact ML/chem image IDs used by the run. Fail preflight if the requested image tags are missing or do not match the revision being tested.
- **SEC-001**: Give only the controller service account namespace-scoped permissions for the Nextflow Jobs, Pods, pod logs, and events operations. Use a separate worker service account with automounting of its API token disabled. Do not grant cluster-admin or expose controller credentials to scientific task containers.
- **CON-001**: Use a one-control-plane-node kind cluster with Podman as its provider. HostPath storage is only supported by this plan for that single-node development cluster and must not be described as a general shared or multi-node storage solution.
- **CON-002**: Run the Kubernetes setup in Linux, including WSL2 where appropriate. The plan must not rely on host Python, a registry, GPU access, or online workflow services.
- **CON-003**: Keep runtime inputs, work files, reports, images, and credentials outside Git and container image layers. Use the ignored `data/local/k8s/` directory for local Kubernetes storage; cluster deletion must never delete it.
- **GUD-001**: Keep scientific parameters in `configs/demo.yaml`, execution settings in `conf/k8s.config`, and setup/validation instructions in `docs/local-setup.md`.
- **PAT-001**: Reuse the existing DSL2 workflow modules, container labels, stage CLI commands, image lockfiles, and Nextflow strict-syntax requirements. Keep the `podman` profile behavior unchanged.
- **PAT-002**: Make setup and cleanup commands non-interactive, idempotent where possible, and explicit about failures. Cleanup may remove named Kubernetes resources and named temporary probe files only; it must preserve `data/local/k8s/`.

## 2. Implementation Steps

### Implementation Phase 1

- **GOAL-001**: Create a repeatable kind cluster and prove the local shared-storage contract before introducing workflow execution.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-001 | Add `deploy/kind/cluster.yaml.tmpl` and `scripts/kind.sh`. The script must provide `up`, `status`, and `down` commands; validate Podman, kind, kubectl, the Podman connection, and the Linux host path; use `git check-ignore` to verify `data/local/k8s/` is ignored; set `KIND_EXPERIMENTAL_PROVIDER=podman`; and create a one-node cluster. Generate the kind config with an `extraMounts` entry mapping the absolute host directory resolved by joining `git rev-parse --show-toplevel` with `data/local/k8s` to `/mnt/ligand-analysis` in the node. Create the host directory before cluster creation. Load the already-built `localhost/ligand-runner:dev` image into the node for storage-probe pods. `down` must delete only the named kind cluster and must not remove or alter the host directory. | ✅ | 2026-09-27 |
| TASK-002 | Depends on TASK-001. Add `deploy/k8s/namespace.yaml` and `deploy/k8s/storage.yaml`. Define namespace `ligand-analysis`, a static local PersistentVolume whose node path is `/mnt/ligand-analysis`, and a matching `ReadWriteMany` PersistentVolumeClaim named `ligand-analysis-data`. Set both capacities to `50Gi` and both `storageClassName` values to `local-kind-hostpath`; document that RWX is satisfied only for concurrent pods on this single node and does not provide multi-node sharing. Mount this claim at `/workspace` in the controller and worker pods. | ✅ | 2026-09-27 |
| TASK-003 | Depends on TASK-002. Add `scripts/k8s-storage-check.sh` and `deploy/k8s/storage-probe.yaml`. Use the preloaded `localhost/ligand-runner:dev` image with `imagePullPolicy: Never`; do not pull a probe image. The check must wait for the claim to bind, launch two namespace-scoped probe pods that mount the claim concurrently, have one write a uniquely named file under `/workspace/probe/` and the other read and verify its exact contents, then remove only the two probe pods and that named file. Exit nonzero with pod events and logs on any failure. | ✅ | 2026-09-27 |
| TASK-004 | Depends on TASK-001, TASK-002, and TASK-003. Add `deploy/k8s/namespace.yaml` and `deploy/k8s/storage.yaml` usage to `scripts/kind.sh up`: apply namespace, PV, and PVC after cluster creation, then invoke `scripts/k8s-storage-check.sh`. Do not report cluster setup success unless the concurrent storage probe passes. | ✅ | 2026-09-27 |

**Phase 1 completion criteria:** `scripts/kind.sh up` creates the expected one-node cluster, the PVC is Bound, two pods pass the shared-file probe, and `scripts/kind.sh down` leaves files under `data/local/k8s/` unchanged.

### Implementation Phase 2

- **GOAL-002**: Configure the existing workflow to schedule containerized tasks on Kubernetes and make its controller source available through the shared volume.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-005 | Independent of TASK-006 and TASK-007; preserve the M3 profile unchanged. Add `conf/k8s.config` and a `k8s` profile in `nextflow.config`. Set `process.executor = 'k8s'`, `process.maxForks = 2`, `executor.queueSize = 2`, `k8s.namespace = 'ligand-analysis'`, `k8s.serviceAccount = 'workflow-worker'`, `k8s.storageClaimName = 'ligand-analysis-data'`, `k8s.storageMountPath = '/workspace'`, `k8s.launchDir = "/workspace/projects/${params.git_revision}"`, `k8s.workDir = '/workspace/work'`, `k8s.computeResourceType = 'Job'`, `k8s.imagePullPolicy = 'Never'`, and `k8s.cpuLimits = true`. Set `params.image_manifest = '/workspace/image-manifest.json'`. Keep each worker at the existing 2 GB process memory and cap DOCK at `params.dock_cpus` (2 CPUs by default). Set Kubernetes run paths to `/workspace/sources` for `params.sources_dir` and `/workspace/results/adrb2-demo` for `params.outdir`. Preserve `params.containers` names and the existing `podman` and `offline` profiles. Verify Nextflow 26.04 strict lint accepts the config and the k8s profile cannot select the Podman executor. | ✅ | 2026-09-27 |
| TASK-006 | Independent of TASK-005. Update `containers/runner/Containerfile` to include only the workflow launch files needed by the controller at `/opt/ligand-analysis`: `main.nf`, `workflow/`, `configs/`, and `data/manifests/`. Do not copy datasets, source snapshots, work, or results into the image. Add an init container to the runner Job (TASK-008) that copies this revisioned application source into the directory selected by `k8s.launchDir`, `/workspace/projects/${params.git_revision}`, before launching Nextflow. Set `NXF_HOME=/workspace/nextflow-home` and retain it on the PVC. | ✅ | 2026-09-27 |
| TASK-007 | Depends on TASK-006. Add `scripts/k8s-images.sh` with `load` and `verify` commands. For the configured ML, chemistry, and runner image tags, require each OCI revision label to equal the selected Git revision, record each Podman image ID and tag in `data/local/k8s/image-manifest.json` (mounted in Kubernetes as `/workspace/image-manifest.json`), export the exact local image builds, load them with `kind load image-archive`, and verify all three tags are available in the kind node before any Job is submitted. Update `main.nf`'s `imageId(name)` lookup to prefer the validated image-manifest entry when supplied, while retaining the existing Podman-inspect behavior for M3. | ✅ | 2026-09-27 |

**Phase 2 completion criteria:** Nextflow lint passes; `nextflow config -profile k8s` resolves the Kubernetes executor, PVC, shared launch/work paths, resource bounds, and `Never` image-pull policy; both worker image IDs in the manifest match the image builds visible to kind; the Podman profile remains unchanged.

### Implementation Phase 3

- **GOAL-003**: Submit a least-privilege, in-cluster Nextflow controller that launches the existing workflow as Kubernetes worker Jobs.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-008 | Depends on TASK-004 through TASK-007. Add `deploy/k8s/service-account.yaml`, `deploy/k8s/rbac.yaml`, and `deploy/k8s/runner-job.yaml.tmpl`. Create controller service account `nextflow-runner` and worker service account `workflow-worker` with `automountServiceAccountToken: false`; bind a namespace-scoped Role only to `nextflow-runner`. Grant `get`, `list`, `watch`, `create`, and `delete` on batch Jobs; `get`, `list`, `watch`, and `delete` on core Pods (the tested nf-k8s client explicitly deletes task pods during cleanup); `get` on core `pods/log` and `pods/status`; `get` on `jobs/status`; and `get`, `list`, and `watch` on core Events. Mount `ligand-analysis-data` at `/workspace` in the init and controller containers. Set the controller Job's `serviceAccountName` to `nextflow-runner`; run the controller from the directory selected by `k8s.launchDir`, `/workspace/projects/${params.git_revision}`, with persistent `NXF_HOME`; invoke `nextflow run main.nf -profile k8s -params-file configs/demo.yaml`, with `--sources_dir /workspace/sources`, `--outdir /workspace/results/adrb2-demo`, and `-work-dir /workspace/work`. Set `k8s.serviceAccount = 'workflow-worker'` on worker Jobs. Support an explicit resume argument without changing the workflow parameters. Do not mount the Podman socket in any Kubernetes pod. | ✅ | 2026-09-27 |
| TASK-009 | Depends on TASK-008 and TASK-010. Add `scripts/k8s-demo.sh` with `run`, `resume`, and `status` commands. Before creating a uniquely named runner Job, call `scripts/k8s-images.sh verify` and verify the PVC and required service-account permissions with `kubectl auth can-i`. Apply the rendered Job manifest, wait for completion for at most 45 minutes, stream controller logs, print worker Job/Pod status and the final report path, and return the runner Job's nonzero exit status on failure. On failure, print controller and failed worker logs and Kubernetes events. Never delete the PVC, PV, host data directory, or prior successful results. | ✅ | 2026-09-27 |
| TASK-010 | Depends on TASK-005 and TASK-008. Configure worker requests/limits of 2 CPUs and 2 GiB for DOCK and 1 CPU and 2 GiB for every other process; configure the controller requests/limits as 1 CPU and 1 GiB in `deploy/k8s/runner-job.yaml.tmpl`. Match Vina `task.cpus` to `params.dock_cpus`, set scientific library thread counts to one unless allocated by the task, apply the `ligand-chem` or `ligand-ml` image from the existing process labels, and have `scripts/kind.sh up` reject Podman Machine configurations below 4 CPUs, 6 GiB memory, or 20 GiB free disk. | ✅ | 2026-09-27 |

**Phase 3 completion criteria:** One invocation of `scripts/k8s-demo.sh run` executes the workflow in namespace `ligand-analysis`, schedules distinct labeled workflow stages as Kubernetes worker Jobs, writes a successful report, captures worker status/logs, and records the source revision and image IDs. No task executes with the local or Podman executor.

### Implementation Phase 4

- **GOAL-004**: Demonstrate durable resume and equivalent scientific outputs between the existing Podman workflow and Kubernetes.

| Task | Description | Completed | Date |
|------|-------------|-----------|------|
| TASK-011 | Depends on TASK-009 and TASK-010. Add `scripts/k8s-acceptance.sh` to run the deterministic M4 acceptance sequence using `configs/demo.yaml` and the same Git revision and image IDs for both profiles. Run the demo once through `scripts/nextflow.sh -profile podman -params-file configs/demo.yaml` and once through `scripts/k8s-demo.sh run`; archive comparison evidence under ignored `data/local/k8s/acceptance/`. Assert exact equality for input checksums, requested-input outcome identifiers/statuses, selected ligand IDs, feature row IDs and feature values, split assignments, predictions, and raw confusion counts. Assert Vina scores differ by no more than 0.01 kcal/mol and matched selected-pose heavy-atom RMSD is at most 0.10 Å. Record the comparison method, tolerance, run paths, image IDs, and any failed assertion. | ✅ | 2026-09-27 |
| TASK-012 | Depends on TASK-011. Extend `scripts/k8s-acceptance.sh` with durability checks. Save checksums for the PVC-backed source snapshot, successful report, work directory and Nextflow resume metadata; delete and recreate only the named kind cluster; confirm the host checksums are unchanged; recreate namespace/storage resources and reload the same image builds; then run `scripts/k8s-demo.sh resume`. Require Nextflow to report cached worker tasks and require the final scientific artifacts to satisfy TASK-011 comparisons. Preserve the original acceptance evidence and all demo results. | ✅ | 2026-09-27 |
| TASK-013 | Depends on TASK-011 and TASK-012. Update `docs/local-setup.md`, `README.md`, and `docs/progress.md`. Document Linux/WSL2 prerequisites, pinned tool versions, Podman kind provider setup, exact commands for cluster lifecycle, image load, storage check, full demo, resume, status and cleanup, UID/GID behavior, mount paths, CPU/memory limits, logs, failure recovery, the single-node-only HostPath limitation, and which M4 commands and acceptance criteria were actually executed. Mark M4 done in `docs/progress.md` only after TASK-011 and TASK-012 pass; otherwise list exact failed/unavailable evidence and retain an in-progress status. | ✅ | 2026-09-27 |

**Phase 4 completion criteria:** TASK-011 comparisons pass at the specified tolerances; TASK-012 proves inputs, outputs, work data, and resume state survive kind deletion/recreation and reuses valid tasks; documentation records actual evidence without claiming unexecuted chemistry or Kubernetes validation.

## 3. Alternatives

- **ALT-001**: Use Nextflow Fusion instead of shared storage. Rejected because the M4 architecture explicitly uses ordinary local filesystem storage and does not depend on hosted workflow services or Fusion.
- **ALT-002**: Run workflow stages with `podman kube play` or retain the Podman executor inside the controller. Rejected because this does not verify Nextflow's Kubernetes executor or actual Kubernetes worker pods.
- **ALT-003**: Use a remote registry for workflow images. Rejected because the local milestone must work without a registry; Podman-built image archives are loaded directly into kind's containerd runtime.
- **ALT-004**: Use a Kubernetes `hostPath` volume as production-grade shared storage. Rejected because it is node-local and suitable here only for the explicitly single-node development cluster.

## 4. Dependencies

- **DEP-001**: M3's `main.nf`, DSL2 workflow modules, Podman demo, and scientific acceptance baseline must remain available and pass before M4 implementation begins.
- **DEP-002**: Linux or WSL2 with a running Podman service, `kind` configured for the Podman provider, `kubectl`, and enough local CPU, memory, and disk to build/load images and run the demo.
- **DEP-003**: `ligand-ml`, `ligand-chem`, and `ligand-runner` images built from one source revision with their existing committed environment lockfiles.
- **DEP-004**: Nextflow Kubernetes executor requirements: a namespace-accessible kubeconfig/service account and a shared PVC; use the Kubernetes executor's `Job` compute resource type and the official k8s configuration keys.
- **DEP-005**: An existing or fetchable ADRB2 demo snapshot plus the DVC-tracked scientific inputs required by `configs/demo.yaml`; all materialized run data stays under `data/local/k8s/`.

## 5. Files

- **FILE-001**: `deploy/kind/cluster.yaml.tmpl` — generated single-node kind configuration with a persistent node mount.
- **FILE-002**: `deploy/k8s/namespace.yaml`, `deploy/k8s/storage.yaml`, and `deploy/k8s/storage-probe.yaml` — namespace, static local PV/PVC, and shared-storage probe.
- **FILE-003**: `deploy/k8s/service-account.yaml`, `deploy/k8s/rbac.yaml`, and `deploy/k8s/runner-job.yaml.tmpl` — in-cluster controller identity, scoped permissions, and parameterized runner Job.
- **FILE-004**: `scripts/kind.sh`, `scripts/k8s-storage-check.sh`, `scripts/k8s-images.sh`, `scripts/k8s-demo.sh`, and `scripts/k8s-acceptance.sh` — local lifecycle, image, execution, and acceptance commands.
- **FILE-005**: `conf/k8s.config` and `nextflow.config` — Kubernetes executor profile and bounded execution settings; preserve existing Podman profile.
- **FILE-006**: `containers/runner/Containerfile` — revision-matched workflow launch source, excluding runtime data.
- **FILE-007**: `main.nf` — Kubernetes-compatible run metadata image-ID lookup with unchanged Podman fallback.
- **FILE-008**: `docs/local-setup.md`, `README.md`, and `docs/progress.md` — setup instructions and evidence-based M4 status.
- **FILE-009**: `data/local/k8s/` — ignored local persistent state for PVC-backed sources, work, reports, cache, image manifest, and acceptance evidence; never commit this directory.

## 6. Testing

- **TEST-001**: `scripts/kind.sh up`, `kubectl get nodes -o wide`, `kubectl -n ligand-analysis get pvc ligand-analysis-data`, and `scripts/k8s-storage-check.sh` prove kind is Podman-backed, the claim is Bound, and two pods concurrently share the same file.
- **TEST-002**: `scripts/k8s-images.sh load` followed by `scripts/k8s-images.sh verify` proves the ML and chemistry worker images and the runner image with the expected source revision are present in kind before scheduling.
- **TEST-003**: Run `podman run --rm -v "${PWD}:/mnt/w:ro" -w /mnt/w ligand-runner:dev nextflow lint main.nf workflow nextflow.config conf`; validate `nextflow config -profile k8s`; validate all YAML manifests with `kubectl apply --dry-run=client`; and run the existing offline ML and chemistry regression commands from README.md without regressions.
- **TEST-004**: `scripts/k8s-demo.sh run` must produce a successful report, execute multiple named process stages in Kubernetes worker Jobs, include input outcome records, and record source/image identity in run metadata.
- **TEST-005**: `scripts/k8s-acceptance.sh` compares Podman and Kubernetes demo outputs: exact input checksums/outcome identities, selected ligand IDs, features, split assignments, predictions and raw confusion counts; Vina score delta <= 0.01 kcal/mol; matched selected-pose heavy-atom RMSD <= 0.10 Å.
- **TEST-006**: `scripts/k8s-acceptance.sh` deletes/recreates the named kind cluster without changing host-persisted file checksums, then confirms `scripts/k8s-demo.sh resume` reuses cached tasks and regenerates equivalent scientific artifacts.
- **TEST-007**: Run the acceptance workflow with an unavailable image tag, an unbound PVC, and a deliberately denied namespace permission; each case must fail before scientific tasks start and report the failed prerequisite without success-shaped output.

## 7. Risks & Assumptions

- **RISK-001**: Podman machine bind mounts and kind `extraMounts` may expose different host paths or ownership on Windows/WSL2. Stop before workflow execution if storage probes or UID/GID checks fail; document the actual tested host path and mapping.
- **RISK-002**: Nextflow 26.04 Kubernetes executor behavior or RBAC requirements may differ from assumptions. Validate required operations against the selected version and reduce permissions to the smallest tested namespace-scoped set.
- **RISK-003**: Podman image IDs, imported containerd IDs, or locally resolved image names may differ. Keep the exact exported image archives, verify node-visible tags and revision labels, and do not fall back to registry pulls.
- **RISK-004**: Small demo runs do not establish biological generalization. Report only the M3 software/molecular smoke-test evidence and preserve its explicit limitation on scientific claims.
- **ASSUMPTION-001**: M3 acceptance remains green and its deterministic ADRB2 demo can run from the pinned configuration and same source revision before M4 comparison.
- **ASSUMPTION-002**: The Podman Machine can be configured with at least 4 CPUs, 6 GiB memory, and 20 GiB free disk, and exposes a host directory that persists after kind node deletion.
- **ASSUMPTION-003**: The static HostPath PV can be declared `ReadWriteMany` for concurrent pods on one node; this must be proven by TEST-001 and is not an assertion of multi-node RWX behavior.

## 8. Related Specifications / Further Reading

- [Local pipeline plan](../docs/LOCAL_PIPELINE_PLAN.md), especially sections 3, 7, 8, 12, and 13.
- [Progress and M3 execution evidence](../docs/progress.md).
- [Scientific protocol](../docs/protocol.md).
- [Nextflow Kubernetes executor documentation](https://docs.seqera.io/nextflow/kubernetes).
- [Nextflow Kubernetes configuration reference](https://docs.seqera.io/nextflow/reference/config/k8s).
- [kind configuration documentation](https://kind.sigs.k8s.io/docs/user/configuration/).
