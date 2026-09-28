#!/usr/bin/env bash
# Run Nextflow for this repository inside the ligand-runner image on Linux (rootless Podman).
# The runner talks to the user's Podman API socket (systemctl --user enable --now podman.socket),
# so every task runs as a sibling container. The repository is mounted at its own absolute path
# so work directories resolve identically for the controller and the task containers.
# All arguments are passed to "nextflow run main.nf", e.g.
#   scripts/nextflow.sh -profile podman -params-file configs/demo.yaml
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
socket=$(podman info --format '{{.Host.RemoteSocket.Path}}')
socket=${socket#unix://}
revision=${LIGAND_ANALYSIS_REVISION:-$(git -C "$repo" describe --always --dirty 2>/dev/null || echo unknown)}
exec podman run --rm --user root --security-opt label=disable \
    -v "$socket:/run/podman/podman.sock" -v "$repo:$repo" -w "$repo" \
    localhost/ligand-runner:dev nextflow run main.nf "$@" --git_revision "$revision"
