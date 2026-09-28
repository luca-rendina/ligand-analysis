#!/usr/bin/env bash
# Put the locally built workflow images into the kind node and record their identity.
#   scripts/k8s-images.sh load     export, validate and load the ML, chem and runner images;
#                                  write data/local/k8s/image-manifest.json
#   scripts/k8s-images.sh verify   check the manifest against Podman and the kind node
# Each manifest entry records the Podman image ID (= config digest, which verify compares with
# the node) and node_digest, the manifest digest the kubelet reports as a pod's imageID.
# Every image must carry the OCI revision label of the selected Git revision (LIGAND_REVISION,
# default: git describe --always --dirty, as in the README build commands). Nothing is pulled
# from a registry: the exact local builds are saved with Podman, checked, and loaded with
# "kind load image-archive". The archives stay in data/local/k8s-images/ (ignored by Git and
# outside the PVC). The manifest is mounted in Kubernetes as /workspace/image-manifest.json.
# Overrides: LIGAND_REVISION, LIGAND_KIND_CLUSTER, LIGAND_ML_IMAGE, LIGAND_CHEM_IMAGE, LIGAND_RUNNER_IMAGE.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
source "$repo/scripts/worktree-git.sh"
# Linux Git (e.g. in the Podman Machine) may not see a Windows linked worktree; then the revision
# must be passed from the Windows checkout in LIGAND_REVISION. Never let Linux Git rewrite the
# index that Windows Git shares.
export GIT_OPTIONAL_LOCKS=0
git_ok=1
configure_worktree_git "$repo" 2>/dev/null || { git_ok=0; unset GIT_DIR GIT_WORK_TREE; }
cluster=${LIGAND_KIND_CLUSTER:-ligand-analysis}
data_dir="$repo/data/local/k8s"
archive_dir="$repo/data/local/k8s-images"
manifest="$data_dir/image-manifest.json"
keys=(ml chem runner)
declare -A images=(
    [ml]=${LIGAND_ML_IMAGE:-localhost/ligand-ml:dev}
    [chem]=${LIGAND_CHEM_IMAGE:-localhost/ligand-chem:dev}
    [runner]=${LIGAND_RUNNER_IMAGE:-localhost/ligand-runner:dev}
)
export KIND_EXPERIMENTAL_PROVIDER=podman

PROG=k8s-images
die() { echo "k8s-images: $*" >&2; exit 1; }

# LIGAND_REVISION (e.g. "git describe --always --dirty" run in the Windows checkout) takes
# precedence. If Linux Git can also read the worktree, a different commit is an error; a
# difference in the -dirty suffix alone (line-ending views can differ) is a warning.
resolve_revision() {
    local explicit=${LIGAND_REVISION:-} described="" a b
    if (( git_ok )); then
        described=$(git -C "$repo" describe --always --dirty 2>/dev/null) || described=""
    fi
    if [[ -n $explicit ]]; then
        [[ $explicit =~ ^[0-9a-f]{7,40}(-dirty)?$ ]] || die "LIGAND_REVISION must look like git describe --always --dirty output, got '$explicit'"
        if [[ -n $described && $described != "$explicit" ]]; then
            a=${explicit%-dirty} b=${described%-dirty}
            [[ $a == "$b"* || $b == "$a"* ]] || die "LIGAND_REVISION=$explicit, but this worktree is at $described"
            echo "$PROG: warning: Linux Git reports $described; using LIGAND_REVISION=$explicit" >&2
        fi
        printf '%s\n' "$explicit"
    elif [[ -n $described ]]; then
        printf '%s\n' "$described"
    else
        die "cannot read the Git revision from this shell; pass it from the checkout, e.g. LIGAND_REVISION=\$(git describe --always --dirty) (PowerShell: \$env:LIGAND_REVISION = git describe --always --dirty)"
    fi
}

# Whether a data path is ignored by Git; without usable Git, fall back to the data/local/ rule
# in .gitignore.
# A Git error (e.g. "dubious ownership" or unreadable worktree metadata) also falls back.
ignored() {
    local rc=128
    if (( git_ok )); then
        git -C "$repo" check-ignore -q "$repo/$1" 2>/dev/null && rc=0 || rc=$?
    fi
    case $rc in
        0) return 0 ;;
        1) return 1 ;;
        *) sed 's/\r$//' "$repo/.gitignore" 2>/dev/null | grep -qxE '/?data/local/?\**' ;;
    esac
}

require_tools() {
    local tool
    for tool in "$@"; do
        command -v "$tool" >/dev/null 2>&1 || die "required tool not found: $tool"
    done
}

selected_revision() {
    local revision
    revision=$(resolve_revision) || exit 1
    if [[ $revision == *-dirty ]]; then
        echo "k8s-images: warning: revision $revision has uncommitted changes; its images are not reproducible from Git" >&2
    fi
    printf '%s\n' "$revision"
}

check_settings() {
    local key
    [[ $cluster =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?$ ]] || die "invalid kind cluster name '$cluster'"
    for key in "${keys[@]}"; do
        # A fully qualified localhost/ reference keeps the same name in Podman and containerd.
        [[ ${images[$key]} =~ ^localhost/[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] \
            || die "image for '$key' must be a localhost/<name>:<tag> reference: '${images[$key]}'"
    done
    ignored data/local/k8s/ || die "data/local/k8s/ is not ignored by Git"
    ignored data/local/k8s-images/ || die "data/local/k8s-images/ is not ignored by Git"
}

kind_node() {
    local nodes
    local clusters
    clusters=$(kind get clusters 2>/dev/null) || die "cannot list kind clusters"
    grep -qx "$cluster" <<<"$clusters" || die "kind cluster '$cluster' does not exist (run scripts/kind.sh up)"
    nodes=$(kind get nodes --name "$cluster" 2>/dev/null) || die "cannot list the nodes of kind cluster '$cluster'"
    [[ $(wc -w <<<"$nodes") -eq 1 ]] || die "cluster '$cluster' must have exactly one node, found: $nodes"
    printf '%s\n' "$nodes"
}

# Image ID (config digest) as bare lowercase hex: Podman versions print it with or without the
# "sha256:" prefix. The manifest and all comparisons use this form.
podman_id() {
    local out
    out=$(podman image inspect --format '{{.Id}}' "$1" 2>/dev/null) || die "image $1 is not in local Podman storage (build it first, see README.md)"
    printf '%s\n' "${out#sha256:}"
}

podman_label() {
    podman image inspect --format "{{index .Labels \"$2\"}}" "$1"
}

# Image ID as seen by the kubelet (the config digest, equal to the Podman image ID), or empty.
node_id() {
    local out
    out=$(podman exec "$1" crictl inspecti -o go-template --template '{{.status.id}}' "$2" 2>/dev/null) || return 0
    printf '%s\n' "${out#sha256:}"
}

# Manifest digest that the kubelet reports as a pod's imageID (kind imports the archive under a
# generated name), or empty.
node_digest() {
    local out
    out=$(podman exec "$1" crictl inspecti -o go-template --template '{{range .status.repoDigests}}{{.}} {{end}}' "$2" 2>/dev/null) || return 0
    grep -o '@sha256:[0-9a-f]\{64\}' <<<"$out" | head -n 1 | cut -c2- || true
}

check_local_image() {
    local key=$1 revision=$2 image=${images[$1]} id label title
    id=$(podman_id "$image")
    [[ $id =~ ^[0-9a-f]{64}$ ]] || die "unexpected Podman image ID for $image: $id"
    label=$(podman_label "$image" org.opencontainers.image.revision)
    [[ $label == "$revision" ]] || die "$image has revision label '$label', expected '$revision' (rebuild it with --build-arg REVISION=$revision)"
    title=$(podman_label "$image" org.opencontainers.image.title)
    [[ $title == "ligand-$key" ]] || die "$image is titled '$title', expected 'ligand-$key'"
    printf '%s\n' "$id"
}

# The archive must hold exactly this image: one manifest entry with this tag, whose config blob
# hashes to the image ID.
validate_archive() {
    local archive=$1 image=$2 id=$3 entry config_hash
    entry=$(tar -xOf "$archive" manifest.json) || die "$archive has no manifest.json"
    [[ $entry == "[{\"Config\":\"$id.json\",\"RepoTags\":[\"$image\"],"* ]] \
        || die "$archive does not contain exactly $image ($id): $entry"
    [[ $(grep -o '"Config"' <<<"$entry" | wc -l) -eq 1 ]] || die "$archive contains more than one image"
    config_hash=$(tar -xOf "$archive" "$id.json" | sha256sum | cut -d' ' -f1)
    [[ $config_hash == "$id" ]] || die "$archive: image config hashes to $config_hash, not $id"
}

export_archive() {
    local image=$1 id=$2 archive="$archive_dir/$3-$2.tar" tmp
    if [[ -f $archive && -f $archive.sha256 ]] \
        && [[ $(sha256sum "$archive" | cut -d' ' -f1) == "$(cat "$archive.sha256")" ]]; then
        validate_archive "$archive" "$image" "$id"
        echo "k8s-images: reusing $archive" >&2
    else
        tmp="$archive.partial.$$"
        rm -f "$tmp"
        echo "k8s-images: saving $image ($id) to $archive" >&2
        podman save --format docker-archive -o "$tmp" "$image"
        validate_archive "$tmp" "$image" "$id"
        sha256sum "$tmp" | cut -d' ' -f1 >"$archive.sha256"
        mv -f "$tmp" "$archive"
    fi
    printf '%s\n' "$archive"
}

cmd_load() {
    require_tools podman kind tar sha256sum
    check_settings
    local revision node key id archive now tmp
    revision=$(selected_revision)
    node=$(kind_node)
    mkdir -p "$archive_dir" "$data_dir"
    declare -A ids archives sums digests
    for key in "${keys[@]}"; do
        ids[$key]=$(check_local_image "$key" "$revision")
    done
    for key in "${keys[@]}"; do
        archive=$(export_archive "${images[$key]}" "${ids[$key]}" "$key")
        archives[$key]=${archive#"$repo"/}
        sums[$key]=$(cat "$archive.sha256")
        echo "k8s-images: loading ${images[$key]} into $node" >&2
        kind load image-archive --name "$cluster" "$archive"
        [[ $(node_id "$node" "${images[$key]}") == "${ids[$key]}" ]] \
            || die "after loading, $node does not resolve ${images[$key]} to ${ids[$key]}"
        digests[$key]=$(node_digest "$node" "${images[$key]}")
        [[ ${digests[$key]} =~ ^sha256:[0-9a-f]{64}$ ]] || die "no manifest digest for ${images[$key]} in $node"
    done
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    tmp="$manifest.partial.$$"
    {
        printf '{\n  "schema_version": 1,\n  "git_revision": "%s",\n  "kind_cluster": "%s",\n  "kind_node": "%s",\n  "created_utc": "%s",\n  "images": {\n' \
            "$revision" "$cluster" "$node" "$now"
        for key in "${keys[@]}"; do
            printf '    "%s": {"image": "%s", "id": "%s", "revision": "%s", "archive": "%s", "archive_sha256": "%s", "node_digest": "%s"}%s\n' \
                "$key" "${images[$key]}" "${ids[$key]}" "$revision" "${archives[$key]}" "${sums[$key]}" "${digests[$key]}" \
                "$([[ $key == "${keys[-1]}" ]] || echo ,)"
        done
        printf '  }\n}\n'
    } >"$tmp"
    mv -f "$tmp" "$manifest"
    echo "k8s-images: wrote $manifest" >&2
    cmd_verify
}

manifest_field() {
    sed -n "s/^  \"$1\": \"\\([^\"]*\\)\",\$/\\1/p" "$manifest"
}

manifest_entry() {
    sed -n "s/^    \"$1\": {\"image\": \"\\([^\"]*\\)\", \"id\": \"\\([0-9a-f]*\\)\", \"revision\": \"\\([^\"]*\\)\".*/\\1 \\2 \\3/p" "$manifest"
}

cmd_verify() {
    require_tools podman kind
    check_settings
    local revision node key entry image id label ok failed=0
    revision=$(selected_revision)
    [[ -f $manifest ]] || die "no image manifest at $manifest (run scripts/k8s-images.sh load)"
    [[ $(manifest_field git_revision) == "$revision" ]] \
        || die "image manifest is for revision '$(manifest_field git_revision)', not '$revision' (run scripts/k8s-images.sh load)"
    [[ $(manifest_field kind_cluster) == "$cluster" ]] \
        || die "image manifest was written for kind cluster '$(manifest_field kind_cluster)', not '$cluster'"
    node=$(kind_node)
    for key in "${keys[@]}"; do
        entry=$(manifest_entry "$key")
        image='' id='' label=''
        read -r image id label <<<"$entry" || true
        if [[ -z $entry || $image != "${images[$key]}" || $label != "$revision" || ! $id =~ ^[0-9a-f]{64}$ ]]; then
            echo "k8s-images: manifest entry '$key' does not describe ${images[$key]} at revision $revision: '$entry'" >&2
            failed=1
            continue
        fi
        ok=1
        if [[ $(podman_id "$image") != "$id" ]]; then
            echo "k8s-images: $image no longer resolves to $id in Podman (rebuilt or retagged; run scripts/k8s-images.sh load)" >&2
            ok=0
        elif [[ $(podman_label "$image" org.opencontainers.image.revision) != "$revision" ]]; then
            echo "k8s-images: $image has no revision label $revision" >&2
            ok=0
        fi
        if [[ $(node_id "$node" "$image") != "$id" ]]; then
            echo "k8s-images: $node resolves $image to '$(node_id "$node" "$image")', expected $id (run scripts/k8s-images.sh load)" >&2
            ok=0
        fi
        if [[ $ok -eq 1 ]]; then
            echo "k8s-images: ok $key $image $id"
        else
            failed=1
        fi
    done
    [[ $failed -eq 0 ]] || die "image verification failed"
    echo "k8s-images: all images match revision $revision in $node"
}

case ${1:-} in
    load) cmd_load ;;
    verify) cmd_verify ;;
    *) echo "usage: $0 load|verify" >&2; exit 2 ;;
esac