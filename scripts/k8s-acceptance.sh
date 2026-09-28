#!/usr/bin/env bash
# End-to-end M4 acceptance on Linux with Podman, kind and kubectl installed.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
state="$repo/data/local/k8s"
podman_results="$repo/data/local/results/adrb2-demo"
k8s_results="$state/results/adrb2-demo"
evidence="$state/acceptance"
image_revision=$(podman image inspect --format '{{index .Labels "org.opencontainers.image.revision"}}' \
    localhost/ligand-ml:dev)
[[ -n "$image_revision" && "$image_revision" != unknown ]] ||
    { echo 'ML image does not declare a concrete source revision' >&2; exit 1; }
export LIGAND_ANALYSIS_REVISION="$image_revision"
export LIGAND_REVISION="$image_revision"

snapshot_hashes() {
    local root=$1
    if [[ ! -f "$root/source_index.tsv" ]]; then
        echo "Missing pinned source index: $root/source_index.tsv" >&2
        return 1
    fi
    while IFS=$'\t' read -r _id _url sha _rest; do
        [[ "$sha" =~ ^[0-9a-f]{64}$ ]] || continue
        [[ -f "$root/objects/$sha" ]] || { echo "Missing source object $sha" >&2; return 1; }
        printf '%s  %s\n' "$sha" "$root/objects/$sha"
    done < "$root/source_index.tsv" | sha256sum -c --status
    (cd "$root" && find objects -type f -print0 | sort -z | xargs -0 sha256sum)
}

state_hashes() {
    (cd "$state" && find sources results work nextflow-home projects -type f -print0 |
        sort -z | xargs -0 sha256sum)
}

compare() {
    [[ -f "$podman_results/report/report.json" && -f "$k8s_results/report/report.json" ]] ||
        { echo 'Both demo reports are required for comparison' >&2; return 1; }
    snapshot_hashes "$repo/data/sources/adrb2" > "$evidence/podman-snapshot.sha256"
    snapshot_hashes "$state/sources/adrb2" > "$evidence/k8s-snapshot.sha256"
    (cd "$repo/data/sources/adrb2" && find objects -type f -print0 |
        sort -z | xargs -0 sha256sum) > "$evidence/source-checksums-podman.txt"
    (cd "$state/sources/adrb2" && find objects -type f -print0 |
        sort -z | xargs -0 sha256sum) > "$evidence/source-checksums-k8s.txt"
    diff -u "$evidence/source-checksums-podman.txt" "$evidence/source-checksums-k8s.txt"
    if ! podman run --rm --network=none -v "$repo:/comparison:ro" -w /comparison \
        localhost/ligand-chem:dev python scripts/k8s_compare.py \
        /comparison/data/local/results/adrb2-demo \
        /comparison/data/local/k8s/results/adrb2-demo \
        > "$evidence/comparison.json" 2> "$evidence/comparison-error.txt"; then
        cat "$evidence/comparison-error.txt" >&2
        return 1
    fi
    rm -f "$evidence/comparison-error.txt"
    cat "$evidence/comparison.json"
}

mkdir -p "$evidence"
case "${1:-all}" in
    compare)
        compare
        ;;
    all)
        [[ -f "$repo/data/sources/adrb2/source_index.tsv" ]] ||
            { echo 'Restore the ADRB2 snapshot with DVC before running M4 acceptance' >&2; exit 1; }
        if [[ -e "$state/sources/adrb2" ]]; then
            snapshot_hashes "$state/sources/adrb2" > /dev/null
        else
            mkdir -p "$state/sources"
            cp -a "$repo/data/sources/adrb2" "$state/sources/adrb2"
        fi
        snapshot_hashes "$repo/data/sources/adrb2" > /dev/null
        snapshot_hashes "$state/sources/adrb2" > /dev/null
        compare_sources=$(diff -u <(snapshot_hashes "$repo/data/sources/adrb2") \
            <(snapshot_hashes "$state/sources/adrb2")) || {
            echo "Staged snapshot does not match the Podman input: $compare_sources" >&2
            exit 1
        }
        scripts/kind.sh up
        scripts/k8s-images.sh load
        scripts/nextflow.sh -profile podman,offline -params-file configs/demo.yaml \
            --sources_dir data/sources --outdir data/local/results/adrb2-demo
        scripts/k8s-demo.sh run
        runner_job=$(kubectl --context kind-ligand-analysis -n ligand-analysis get jobs \
            -l app.kubernetes.io/component=nextflow-controller,ligand-analysis/mode=run \
            --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')
        [[ -n "$runner_job" ]] || { echo 'Kubernetes runner Job was not recorded' >&2; exit 1; }
        worker_summary="$repo/data/local/k8s-runs/$runner_job/worker-pods.summary.txt"
        [[ -s "$worker_summary" ]] || { echo "No worker pod evidence: $worker_summary" >&2; exit 1; }
        cp "$worker_summary" "$evidence/worker-pods.summary.txt"
        if ! awk '$1 != "none" && NF >= 4 && $2 == "workflow-worker" { process[$1] = 1 }
                  END { for (name in process) count++; exit(count < 2) }' "$worker_summary"; then
            echo 'Fewer than two distinct Nextflow Kubernetes worker processes were observed' >&2
            exit 1
        fi
        compare
        state_hashes > "$evidence/before-restart.sha256"
        scripts/kind.sh down
        (cd "$state" && sha256sum -c "$evidence/before-restart.sha256")
        scripts/kind.sh up
        scripts/k8s-images.sh load
        scripts/k8s-demo.sh resume
        compare
        if ! grep -q $'\tCACHED\t' "$k8s_results/pipeline_info/trace.tsv"; then
            echo 'Resume did not reuse any Kubernetes worker task' >&2
            exit 1
        fi
        echo "M4 acceptance passed: $evidence/comparison.json"
        ;;
    *)
        echo 'Usage: scripts/k8s-acceptance.sh [all|compare]' >&2
        exit 2
        ;;
esac
