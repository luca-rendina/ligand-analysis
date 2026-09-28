#!/usr/bin/env bash
set -Eeuo pipefail

readonly CLUSTER_NAME=ligand-analysis
readonly NAMESPACE=ligand-analysis
readonly RUNNER_IMAGE=localhost/ligand-runner:dev
readonly STORAGE_DIR_REL=data/local/k8s

usage() {
    printf 'Usage: %s {up|status|down}\n' "${0##*/}" >&2
}

fail() {
    printf 'kind: ERROR: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

[[ $# -eq 1 ]] || { usage; exit 2; }
command_name=$1
case "$command_name" in
    up|status|down) ;;
    *) usage; exit 2 ;;
esac

case "$(uname -s)" in
    Linux) ;;
    *)
        fail "Run this script in Linux or WSL2, not Windows/Git Bash. Podman Machine interprets bind-mount source paths on its Linux VM; use a path visible to that VM."
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
if [[ -d "$repo_root/.git" ]]; then
    git_dir="$repo_root/.git"
elif [[ -f "$repo_root/.git" ]]; then
    gitfile=
    IFS= read -r gitfile < "$repo_root/.git" || [[ -n "$gitfile" ]] ||
        fail "Cannot read the linked-worktree pointer at $repo_root/.git."
    gitfile=${gitfile%$'\r'}
    [[ "$gitfile" == 'gitdir: '* ]] ||
        fail "Unrecognized linked-worktree pointer in $repo_root/.git."
    git_dir=${gitfile#gitdir: }
    if [[ "$git_dir" =~ ^([A-Za-z]):[/\\](.*)$ ]]; then
        windows_drive=${BASH_REMATCH[1],,}
        windows_git_dir=$(printf '%s' "${BASH_REMATCH[2]}" | tr '\\' '/')
        git_dir="/mnt/$windows_drive/$windows_git_dir"
    elif [[ "$git_dir" != /* ]]; then
        git_dir="$repo_root/$git_dir"
    fi
else
    fail "Cannot find Git metadata at $repo_root/.git."
fi
[[ -d "$git_dir" ]] ||
    fail "Git metadata directory is unavailable at '$git_dir'. Ensure a Windows gitdir path is mounted in Podman Machine under /mnt/<drive>/."
git_dir=$(cd -- "$git_dir" && pwd -P)
git_worktree=$(git --git-dir="$git_dir" --work-tree="$repo_root" rev-parse --show-toplevel) ||
    fail "Cannot resolve the worktree using Git metadata at '$git_dir'."
git_worktree=$(cd -- "$git_worktree" && pwd -P)
[[ "$git_worktree" == "$repo_root" ]] ||
    fail "Git metadata resolves to '$git_worktree', not the script worktree '$repo_root'."
storage_dir="$repo_root/$STORAGE_DIR_REL"
kind_context="kind-$CLUSTER_NAME"

for required_command in awk df git grep podman tr wc; do
    require_command "$required_command"
done

export KIND_EXPERIMENTAL_PROVIDER=podman

cluster_exists() {
    kind get clusters | grep -Fxq "$CLUSTER_NAME"
}

check_node_count() {
    local node_count
    node_count=$(kubectl --context "$kind_context" get nodes --no-headers 2>/dev/null | wc -l | tr -d ' ')
    [[ "$node_count" == 1 ]] || fail "Expected exactly one node in kind cluster '$CLUSTER_NAME'; found $node_count."
    podman ps -a --format '{{.Names}}' | grep -Fxq "$CLUSTER_NAME-control-plane" ||
        fail "kind node '$CLUSTER_NAME-control-plane' is not present in Podman's container list; refusing to treat another provider's cluster as the expected cluster."
}

check_podman_resources() {
    local host_info host_cpus host_mem_bytes graph_root remote_host host_kernel
    local machine_listing machine_name listed_name is_default is_running
    local machine_info machine_state machine_cpus machine_memory_mib
    local disk_output disk_available_kib
    local required_mem_bytes=6442450944
    local required_disk_kib=20971520

    # Measured on the Podman host itself (the Linux VM for Podman Machine), which is what kind gets.
    host_info=$(podman info --format '{{.Host.CPUs}}|{{.Host.MemTotal}}|{{.Store.GraphRoot}}|{{.Host.ServiceIsRemote}}|{{.Host.Kernel}}') ||
        fail 'Cannot read Podman host CPU, memory, storage, and connection information.'
    IFS='|' read -r host_cpus host_mem_bytes graph_root remote_host host_kernel <<< "$host_info"
    [[ "$host_cpus" =~ ^[0-9]+$ && "$host_mem_bytes" =~ ^[0-9]+$ && -n "$graph_root" && "$remote_host" =~ ^(true|false)$ ]] ||
        fail "Podman returned incomplete host resource information: '$host_info'."
    (( host_cpus >= 4 )) ||
        fail "Podman host exposes $host_cpus CPUs; at least 4 are required."
    (( host_mem_bytes >= required_mem_bytes )) ||
        fail "Podman host has $host_mem_bytes bytes of memory; at least 6442450944 bytes (6 GiB) are required."

    machine_name=
    if [[ "$remote_host" == true ]]; then
        machine_listing=$(podman machine list --format '{{.Name}}|{{.Default}}|{{.Running}}') ||
            fail 'Podman is remote but its Podman Machine cannot be listed.'
        while IFS='|' read -r listed_name is_default is_running; do
            [[ "$is_default" == true ]] || continue
            [[ -z "$machine_name" ]] ||
                fail 'Podman reports multiple default machines; cannot identify the active one.'
            machine_name=${listed_name%\*}
            [[ "$is_running" == true ]] ||
                fail "Default Podman Machine '$machine_name' is not running."
        done <<< "$machine_listing"
        [[ -n "$machine_name" ]] ||
            fail 'Podman is remote, but no default Podman Machine is available to measure its disk space.'

        # WSL2 ignores the machine's configured CPUs/memory (WSL uses .wslconfig), so there
        # the measured values above are authoritative. Other providers apply the configuration.
        if [[ "$host_kernel" != *[Mm]icrosoft* && "$host_kernel" != *WSL* ]]; then
            machine_info=$(podman machine inspect "$machine_name" \
                --format '{{.Name}}|{{.State}}|{{.Resources.CPUs}}|{{.Resources.Memory}}') ||
                fail "Cannot inspect configured resources for Podman Machine '$machine_name'."
            IFS='|' read -r listed_name machine_state machine_cpus machine_memory_mib <<< "$machine_info"
            [[ "$listed_name" == "$machine_name" && "$machine_state" == running &&
                "$machine_cpus" =~ ^[0-9]+$ && "$machine_memory_mib" =~ ^[0-9]+$ ]] ||
                fail "Podman Machine '$machine_name' returned incomplete resource configuration: '$machine_info'."
            (( machine_cpus >= 4 )) ||
                fail "Podman Machine '$machine_name' is configured with $machine_cpus CPUs; at least 4 are required."
            (( machine_memory_mib >= 6144 )) ||
                fail "Podman Machine '$machine_name' is configured with $machine_memory_mib MiB RAM; at least 6144 MiB (6 GiB) are required."
        fi
        disk_output=$(podman machine ssh "$machine_name" df -Pk "$graph_root" "$repo_root") ||
            fail "Cannot measure free disk space on Podman Machine '$machine_name' for its image store and worktree."
    else
        disk_output=$(df -Pk "$graph_root" "$repo_root") ||
            fail 'Cannot measure free disk space on the Podman host image store and worktree.'
    fi
    disk_available_kib=$(printf '%s\n' "$disk_output" | awk '
        NR > 1 && $4 ~ /^[0-9]+$/ {
            if (minimum == "" || $4 < minimum) minimum = $4
            rows++
        }
        END {
            if (rows == 0) exit 1
            print minimum
        }
    ') || fail "Could not parse available disk space from Podman-host df output: $disk_output"
    (( disk_available_kib >= required_disk_kib )) ||
        fail "Podman host has $disk_available_kib KiB free on the least-available image-store/worktree filesystem; at least 20971520 KiB (20 GiB) are required."
    printf 'Podman resources passed: %s CPUs, %s bytes memory, %s KiB minimum free disk.\n' \
        "$host_cpus" "$host_mem_bytes" "$disk_available_kib"
}

case "$command_name" in
    status)
        require_command kind
        require_command kubectl
        if ! cluster_exists; then
            printf "kind cluster '%s' is not present.\n" "$CLUSTER_NAME"
            exit 1
        fi
        check_node_count
        printf "kind cluster '%s' (Podman provider):\n" "$CLUSTER_NAME"
        kubectl --context "$kind_context" get nodes -o wide
        kubectl --context "$kind_context" -n "$NAMESPACE" get pvc ligand-analysis-data
        printf 'Persistent host directory (preserved by down): %s\n' "$storage_dir"
        ;;
    down)
        require_command kind
        if cluster_exists; then
            kind delete cluster --name "$CLUSTER_NAME"
        else
            printf "kind cluster '%s' is already absent.\n" "$CLUSTER_NAME"
        fi
        printf 'Left persistent host directory untouched: %s\n' "$storage_dir"
        ;;
    up)
        require_command grep
        require_command mktemp

        git --git-dir="$git_dir" --work-tree="$repo_root" check-ignore -q -- "$STORAGE_DIR_REL" ||
            fail "$STORAGE_DIR_REL is not ignored by Git; refusing to put persistent runtime data in the worktree."

        podman info >/dev/null 2>&1 ||
            fail 'Cannot connect to Podman. Start the Podman service/machine and configure its Linux connection.'
        podman_host_os=$(podman info --format '{{.Host.OS}}') ||
            fail 'Cannot determine the operating system of the Podman host.'
        [[ "$podman_host_os" == linux ]] ||
            fail "Podman host must be Linux, got '$podman_host_os'."
        check_podman_resources
        require_command kind
        require_command kubectl
        podman image exists "$RUNNER_IMAGE" ||
            fail "Required local image is missing: $RUNNER_IMAGE. Build it before running kind up."

        mkdir -p -- "$storage_dir" || fail "Cannot create persistent host directory: $storage_dir"
        case "$storage_dir" in
            *'"'*|*'\'*|*':'*|*$'\n'*)
                fail "Host path contains a quote, backslash, colon, or newline that cannot be represented safely for kind/Podman: $storage_dir"
                ;;
        esac
        [[ "$storage_dir" == /* ]] || fail "Expected an absolute Linux host path, got '$storage_dir'."

        # Podman Machine resolves bind mounts on its Linux VM, not on a Windows client.
        # The sentinel ensures the daemon sees this exact WSL/Linux directory, not an
        # automatically created empty directory at a similarly named remote path.
        preflight_token="kind-path-check-$$-$RANDOM"
        preflight_file=$(mktemp "$storage_dir/.kind-path-check.XXXXXX") ||
            fail "Cannot create a temporary host-path preflight file in $storage_dir."
        trap 'rm -f -- "$preflight_file"' EXIT
        printf '%s' "$preflight_token" > "$preflight_file"
        preflight_name=${preflight_file##*/}
        if ! podman run --rm --security-opt label=disable \
            -v "$storage_dir:/kind-preflight:ro" \
            -e "PREFLIGHT_FILE=$preflight_name" \
            -e "PREFLIGHT_CONTENT=$preflight_token" \
            "$RUNNER_IMAGE" sh -ec \
            'test "$(cat "/kind-preflight/$PREFLIGHT_FILE")" = "$PREFLIGHT_CONTENT"'; then
            fail "Podman cannot read a host-created sentinel at $storage_dir. On Windows/WSL2, keep the repository on a Podman-Machine-shared drive (often /mnt/<drive>/...) or use a Podman service that shares the WSL filesystem."
        fi
        rm -f -- "$preflight_file"
        trap - EXIT

        if ! cluster_exists; then
            template="$repo_root/deploy/kind/cluster.yaml.tmpl"
            [[ -r "$template" ]] || fail "Missing kind cluster template: $template"
            generated_config=$(mktemp "${TMPDIR:-/tmp}/ligand-analysis-kind.XXXXXX.yaml") ||
                fail 'Cannot create temporary kind configuration.'
            trap 'rm -f -- "$generated_config"' EXIT
            if ! awk -v host_path="$storage_dir" '
                { sub(/\r$/, "") }
                (position = index($0, "__HOST_DATA_PATH__")) {
                    $0 = substr($0, 1, position - 1) "\"" host_path "\"" substr($0, position + length("__HOST_DATA_PATH__"))
                    replaced++
                }
                { print }
                END { if (replaced != 1) exit 1 }
            ' "$template" > "$generated_config"; then
                fail "Could not render host path into $template."
            fi
            kind create cluster --name "$CLUSTER_NAME" --config "$generated_config"
            rm -f -- "$generated_config"
            trap - EXIT
        else
            printf "Using existing kind cluster '%s'.\n" "$CLUSTER_NAME"
        fi

        kubectl --context "$kind_context" wait --for=condition=Ready node \
            --all --timeout=120s || fail "kind cluster '$CLUSTER_NAME' did not become ready."
        check_node_count
        # kind's Podman provider cannot resolve localhost/ names for docker-image; load an archive.
        runner_archive=$(mktemp "${TMPDIR:-/tmp}/ligand-analysis-runner.XXXXXX.tar") ||
            fail 'Cannot create a temporary runner image archive.'
        trap 'rm -f -- "$runner_archive"' EXIT
        podman save --format docker-archive -o "$runner_archive" "$RUNNER_IMAGE" ||
            fail "Cannot export $RUNNER_IMAGE to an image archive."
        kind load image-archive "$runner_archive" --name "$CLUSTER_NAME" ||
            fail "Cannot load $RUNNER_IMAGE into kind cluster '$CLUSTER_NAME'."
        rm -f -- "$runner_archive"
        trap - EXIT
        podman exec "$CLUSTER_NAME-control-plane" crictl inspecti "$RUNNER_IMAGE" >/dev/null ||
            fail "$RUNNER_IMAGE is not visible to containerd in kind node '$CLUSTER_NAME-control-plane'."
        kubectl --context "$kind_context" apply -f "$repo_root/deploy/k8s/namespace.yaml"
        kubectl --context "$kind_context" apply -f "$repo_root/deploy/k8s/storage.yaml"
        "$repo_root/scripts/k8s-storage-check.sh"
        printf "kind cluster '%s' is ready and shared storage passed the concurrent UID/GID probe.\n" "$CLUSTER_NAME"
        ;;
esac
