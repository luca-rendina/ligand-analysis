#!/usr/bin/env bash
set -Eeuo pipefail

readonly KIND_CONTEXT=kind-ligand-analysis
readonly NAMESPACE=ligand-analysis
readonly RUNNER_IMAGE=localhost/ligand-runner:dev

fail() {
    printf 'k8s-storage-check: ERROR: %s\n' "$*" >&2
    exit 1
}

for command_name in awk grep kubectl mktemp od podman tr; do
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "Required command not found: $command_name"
done

probe_started=0
temp_dir=
probe_file=
writer_pod=
reader_pod=
cleanup() {
    local status=$?
    trap - EXIT

    if [[ "$status" -ne 0 ]]; then
        printf 'k8s-storage-check: Failure diagnostics:\n' >&2
        if [[ "$probe_started" -eq 1 ]]; then
            for pod in "$writer_pod" "$reader_pod"; do
                kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" describe pod "$pod" >&2 || true
                printf '%s\n' "--- logs: $pod ---" >&2
                kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" logs "$pod" --all-containers=true >&2 || true
            done
        else
            printf 'No probe pod pair was created.\n' >&2
        fi
        kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" get events \
            --sort-by=.lastTimestamp >&2 || true
    fi

    if [[ "$probe_started" -eq 1 ]]; then
        if kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" get pod "$writer_pod" \
            -o jsonpath='{.status.phase}' 2>/dev/null | grep -qx Running; then
            if ! kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" exec "$writer_pod" -- \
                rm -f -- "$probe_file"; then
                printf 'k8s-storage-check: Could not remove probe file %s through the writer pod.\n' "$probe_file" >&2
                status=1
            fi
        fi
        if ! kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" delete pod \
            "$writer_pod" "$reader_pod" --ignore-not-found --wait=true --timeout=30s; then
            printf 'k8s-storage-check: Could not remove both named probe pods.\n' >&2
            status=1
        fi
    fi
    if [[ -n "$temp_dir" ]] && ! rm -rf -- "$temp_dir"; then
        printf 'k8s-storage-check: Could not remove temporary directory %s.\n' "$temp_dir" >&2
        status=1
    fi
    exit "$status"
}
trap cleanup EXIT

export KIND_EXPERIMENTAL_PROVIDER=podman
kubectl --context "$KIND_CONTEXT" get namespace "$NAMESPACE" >/dev/null ||
    fail "Namespace '$NAMESPACE' is unavailable in context '$KIND_CONTEXT'. Run scripts/kind.sh up first."
podman image exists "$RUNNER_IMAGE" ||
    fail "Required local image is missing: $RUNNER_IMAGE."

probe_uid=$(podman run --rm --entrypoint /usr/bin/id "$RUNNER_IMAGE" -u) ||
    fail "Cannot determine the UID used by $RUNNER_IMAGE."
probe_gid=$(podman run --rm --entrypoint /usr/bin/id "$RUNNER_IMAGE" -g) ||
    fail "Cannot determine the GID used by $RUNNER_IMAGE."
[[ "$probe_uid" =~ ^[0-9]+$ && "$probe_gid" =~ ^[0-9]+$ ]] ||
    fail "Runner image returned invalid numeric UID/GID: '$probe_uid:$probe_gid'."

kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" wait \
    --for=jsonpath='{.status.phase}'=Bound pvc/ligand-analysis-data \
    --timeout=120s || fail "PersistentVolumeClaim '$NAMESPACE/ligand-analysis-data' did not bind."

probe_id=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
[[ "$probe_id" =~ ^[0-9a-f]{32}$ ]] || fail 'Could not generate a unique probe identifier.'
probe_file="/workspace/probe/kind-storage-$probe_id"
success_prefix="SHARED_STORAGE_VERIFIED content=ligand-analysis-kind-storage-$probe_id uid_gid=$probe_uid:$probe_gid ownership="
writer_pod="ligand-analysis-probe-writer-$probe_id"
reader_pod="ligand-analysis-probe-reader-$probe_id"
temp_dir=$(mktemp -d "${TMPDIR:-/tmp}/ligand-analysis-storage-check.XXXXXX") ||
    fail 'Cannot create a temporary directory for the probe manifest.'
probe_manifest="$temp_dir/storage-probe.yaml"
probe_template="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)/deploy/k8s/storage-probe.yaml"

[[ -r "$probe_template" ]] || fail "Missing probe manifest: $probe_template"
if ! awk -v probe_id="$probe_id" -v probe_uid="$probe_uid" -v probe_gid="$probe_gid" '
    {
        gsub(/__PROBE_ID__/, probe_id)
        gsub(/__RUN_AS_USER__/, probe_uid)
        gsub(/__RUN_AS_GROUP__/, probe_gid)
        print
    }
' "$probe_template" > "$probe_manifest"; then
    fail "Could not render probe manifest from $probe_template."
fi

probe_started=1
kubectl --context "$KIND_CONTEXT" apply -f "$probe_manifest"
kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" wait \
    --for=condition=Ready "pod/$writer_pod" --timeout=120s
kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" wait \
    --for=jsonpath='{.status.phase}'=Succeeded "pod/$reader_pod" --timeout=120s
reader_logs=$(kubectl --context "$KIND_CONTEXT" -n "$NAMESPACE" logs "$reader_pod")
printf '%s\n' "$reader_logs"
verified_line=$(printf '%s\n' "$reader_logs" | grep -F -- "$success_prefix" || true)
case "$verified_line" in
    "${success_prefix}preserved") owner_mode=preserved ;;
    "${success_prefix}not-preserved-drvfs") owner_mode=not-preserved-drvfs ;;
    *) fail 'Reader pod did not confirm exact shared-file contents, UID/GID, and file ownership.' ;;
esac

if [[ "$owner_mode" == not-preserved-drvfs ]]; then
    printf 'NOTE: the shared mount is WSL drvfs (9p); files appear as 0:0 mode 777 and Linux ownership is not stored. UID:GID %s:%s can create, read and delete files.\n' "$probe_uid" "$probe_gid"
fi
printf 'Shared PVC probe passed (runner UID:GID %s:%s, file ownership %s).\n' "$probe_uid" "$probe_gid" "$owner_mode"
