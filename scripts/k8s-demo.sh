#!/usr/bin/env bash
# Run the ADRB2 demo on the local kind cluster with the in-cluster Nextflow controller.
#   scripts/k8s-demo.sh run                 fresh run (earlier results are moved aside, not deleted)
#   scripts/k8s-demo.sh resume [SESSION]    -resume the last (or the given) session of this revision
#   scripts/k8s-demo.sh status              controller Jobs, worker Jobs/Pods, storage and report
# Before submitting the uniquely named runner Job it verifies the images (scripts/k8s-images.sh
# verify), the PVC and the service-account permissions. It streams the controller log, waits at
# most 45 minutes, and exits with the controller's status. Evidence (rendered Job, logs, worker
# Job/Pod observations, worker-jobs.json/.tsv, events, run.env) is kept in data/local/k8s-runs/<job>/, and run.env is also
# copied to data/local/k8s-runs/last-<mode>.env. Env overrides: LIGAND_REVISION, LIGAND_KIND_CLUSTER,
# LIGAND_KUBE_CONTEXT, LIGAND_RUNNER_IMAGE. It never deletes the PVC,
# the PV, data/local/k8s/ or earlier results. Scientific parameters come from configs/demo.yaml.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
source "$repo/scripts/worktree-git.sh"
# Linux Git (e.g. in the Podman Machine) may not see a Windows linked worktree; then the revision
# must be passed from the Windows checkout in LIGAND_REVISION. Never let Linux Git rewrite the
# index that Windows Git shares.
export GIT_OPTIONAL_LOCKS=0
git_ok=1
configure_worktree_git "$repo" 2>/dev/null || { git_ok=0; unset GIT_DIR GIT_WORK_TREE; }
namespace=ligand-analysis
claim=ligand-analysis-data
cluster=${LIGAND_KIND_CLUSTER:-ligand-analysis}
context=${LIGAND_KUBE_CONTEXT:-kind-$cluster}
runner_image=${LIGAND_RUNNER_IMAGE:-localhost/ligand-runner:dev}
template="$repo/deploy/k8s/runner-job.yaml.tmpl"
data_dir="$repo/data/local/k8s"
runs_dir="$repo/data/local/k8s-runs"
timeout_seconds=2700
controller_sa="system:serviceaccount:$namespace:nextflow-runner"
worker_sa="system:serviceaccount:$namespace:workflow-worker"

PROG=k8s-demo
die() { echo "k8s-demo: $*" >&2; exit 1; }

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

kc() { kubectl --context "$context" -n "$namespace" "$@"; }

require_tools() {
    local tool
    for tool in kubectl sed date timeout; do
        command -v "$tool" >/dev/null 2>&1 || die "required tool not found: $tool"
    done
}

selected_revision() {
    resolve_revision
}

check_can() {
    local who=$1 expected=$2; shift 2
    local answer
    answer=$(kubectl --context "$context" auth can-i --as "$who" "$@" 2>/dev/null || true)
    if [[ $answer != "$expected" ]]; then
        echo "k8s-demo: permission check failed: can-i --as $who $* returned '$answer', expected '$expected'" >&2
        return 1
    fi
}

preflight() {
    local phase failed=0 automount active
    LIGAND_REVISION=$revision LIGAND_KIND_CLUSTER=$cluster LIGAND_RUNNER_IMAGE=$runner_image "$repo/scripts/k8s-images.sh" verify \
        || die "image preflight failed; no Job was submitted"
    kubectl --context "$context" get namespace "$namespace" >/dev/null \
        || die "namespace $namespace does not exist in context $context (run scripts/kind.sh up)"
    # The identities are declarative and idempotent; applying them also restores them after the
    # cluster was recreated. The permission checks below still verify the result.
    kubectl --context "$context" apply -f "$repo/deploy/k8s/service-account.yaml" -f "$repo/deploy/k8s/rbac.yaml" >/dev/null \
        || die "cannot apply deploy/k8s/service-account.yaml and rbac.yaml"
    phase=$(kc get pvc "$claim" -o jsonpath='{.status.phase}' 2>/dev/null || true)
    [[ $phase == Bound ]] || die "PVC $claim is '${phase:-missing}', not Bound (run scripts/kind.sh up)"
    for sa in nextflow-runner workflow-worker; do
        automount=$(kc get serviceaccount "$sa" -o jsonpath='{.automountServiceAccountToken}' 2>/dev/null) \
            || die "service account $sa is missing (kubectl apply -f deploy/k8s/service-account.yaml)"
        [[ $automount == false ]] || die "service account $sa must set automountServiceAccountToken: false"
    done
    # What the nf-k8s client needs, and nothing broader.
    check_can "$controller_sa" yes -n "$namespace" create jobs.batch || failed=1
    check_can "$controller_sa" yes -n "$namespace" get jobs.batch --subresource=status || failed=1
    local verb
    for verb in get list watch delete; do
        check_can "$controller_sa" yes -n "$namespace" "$verb" jobs.batch || failed=1
        check_can "$controller_sa" yes -n "$namespace" "$verb" pods || failed=1
    done
    check_can "$controller_sa" yes -n "$namespace" get pods --subresource=status || failed=1
    check_can "$controller_sa" yes -n "$namespace" get pods --subresource=log || failed=1
    check_can "$controller_sa" yes -n "$namespace" list events || failed=1
    check_can "$controller_sa" no -n "$namespace" get secrets || failed=1
    check_can "$controller_sa" no -n "$namespace" create pods || failed=1
    check_can "$controller_sa" no -n "$namespace" create pods --subresource=exec || failed=1
    check_can "$controller_sa" no -n default create jobs.batch || failed=1
    check_can "$controller_sa" no -n "$namespace" '*' '*' || failed=1
    check_can "$worker_sa" no -n "$namespace" create jobs.batch || failed=1
    check_can "$worker_sa" no -n "$namespace" get pods || failed=1
    [[ $failed -eq 0 ]] || die "RBAC preflight failed (kubectl apply -f deploy/k8s/service-account.yaml -f deploy/k8s/rbac.yaml)"
    active=$(kc get jobs -l app.kubernetes.io/component=nextflow-controller \
        -o jsonpath='{range .items[?(@.status.active)]}{.metadata.name}{" "}{end}')
    [[ -z $active ]] || die "a controller Job is still active: $active (wait for it or check scripts/k8s-demo.sh status)"
    echo "k8s-demo: preflight passed (images, PVC $claim Bound, service accounts and permissions)"
}

render() {
    local job=$1 mode=$2 revision=$3 resume=$4
    sed -e 's/\r$//' -e "s|__JOB_NAME__|$job|g" -e "s|__MODE__|$mode|g" -e "s|__GIT_REVISION__|$revision|g" \
        -e "s|__RUNNER_IMAGE__|$runner_image|g" -e "s|__RESUME__|$resume|g" "$template"
}

pod_of() {
    kc get pods -l "job-name=$1" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

# Prints "running" or "exited" once the controller container can be followed, else "failed" or
# "timeout". Containers that cannot start (e.g. ErrImageNeverPull) count as failed.
wait_for_controller() {
    local job=$1 deadline=$2 pod state init waiting
    while (( SECONDS < deadline )); do
        pod=$(pod_of "$job")
        if [[ -n $pod ]]; then
            init=$(kc get pod "$pod" -o jsonpath='{.status.initContainerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)
            if [[ -n $init && $init != 0 ]]; then
                echo failed; return
            fi
            waiting=$(kc get pod "$pod" -o jsonpath='{.status.initContainerStatuses[0].state.waiting.reason} {.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null || true)
            if [[ $waiting =~ (Err|Invalid|CreateContainer|BackOff) ]]; then
                echo "k8s-demo: runner pod $pod cannot start: $waiting" >&2
                echo failed; return
            fi
            state=$(kc get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null || true)
            case $state in
                *running*) echo running; return ;;
                *terminated*) echo exited; return ;;
            esac
        fi
        if [[ -n $(kc get job "$job" -o jsonpath='{.status.failed}' 2>/dev/null || true) ]]; then
            echo failed; return
        fi
        sleep 3
    done
    echo timeout
}

worker_selector() { printf 'nextflow.io/runName=%s' "$1"; }

watch_workers() {
    local job=$1 dir=$2
    kc get jobs -l "$(worker_selector "$job")" --watch \
        -o custom-columns='NAME:.metadata.name,PROCESS:.metadata.labels.nextflow\.io/processName,TASK:.metadata.labels.nextflow\.io/taskName,ACTIVE:.status.active,SUCCEEDED:.status.succeeded,FAILED:.status.failed' \
        >"$dir/worker-jobs.watch.txt" 2>&1 &
    watchers+=($!)
    kc get pods -l "$(worker_selector "$job")" --watch \
        -o custom-columns='NAME:.metadata.name,PROCESS:.metadata.labels.nextflow\.io/processName,PHASE:.status.phase,NODE:.spec.nodeName,SA:.spec.serviceAccountName,IMAGE:.spec.containers[0].image,IMAGE_ID:.status.containerStatuses[0].imageID,EXIT:.status.containerStatuses[0].state.terminated.exitCode' \
        >"$dir/worker-pods.watch.txt" 2>&1 &
    watchers+=($!)
}

stop_watchers() {
    local pid
    for pid in "${watchers[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    watchers=()
}

report_failure() {
    local job=$1 dir=$2 pod pods name
    pod=$(pod_of "$job")
    echo "k8s-demo: ---- runner Job $job failed; diagnostics ----" >&2
    kc describe job "$job" >"$dir/runner-job.describe.txt" 2>&1 || true
    if [[ -n $pod ]]; then
        kc describe pod "$pod" >"$dir/runner-pod.describe.txt" 2>&1 || true
        echo "k8s-demo: init container (source) log:" >&2
        kc logs "$pod" -c source 2>&1 | tee "$dir/source.log" >&2 || true
        echo "k8s-demo: last controller log lines:" >&2
        kc logs "$pod" -c controller --tail=100 2>&1 >&2 || true
    fi
    pods=$(kc get pods -l "$(worker_selector "$job")" --field-selector=status.phase=Failed \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)
    for name in $pods; do
        echo "k8s-demo: failed worker pod $name:" >&2
        kc logs "$name" --all-containers 2>&1 | tee "$dir/worker-$name.log" >&2 || true
        kc describe pod "$name" >"$dir/worker-$name.describe.txt" 2>&1 || true
    done
    echo "k8s-demo: namespace events:" >&2
    kc get events --sort-by=.lastTimestamp 2>&1 | tail -n 40 >&2 || true
}

# Summarises the worker pods seen by the watch: one line per process, service account, image and
# image ID reported by the kubelet. Fails when a pod did not run as workflow-worker or ran an image
# whose digest is not in the image manifest.
print_workers() {
    local job=$1 dir=$2 observed
    echo "k8s-demo: worker Jobs/Pods of this run still present (all of them with k8s.cleanup = false; otherwise only failed ones):"
    kc get jobs,pods -l "$(worker_selector "$job")" -o wide 2>&1 | tee "$dir/workers.final.txt" || true
    # Durable copy of the worker Jobs and Pods: they disappear with the cluster (kind.sh down).
    kc get jobs,pods -l "$(worker_selector "$job")" -o json >"$dir/worker-jobs.json" 2>/dev/null || true
    kc get jobs -l "$(worker_selector "$job")" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.nextflow\.io/processName}{"\t"}{.metadata.labels.nextflow\.io/taskName}{"\t"}{.spec.template.spec.serviceAccountName}{"\t"}{.status.succeeded}{"\t"}{.status.failed}{"\n"}{end}' \
        >"$dir/worker-jobs.tsv" 2>/dev/null || true
    worker_jobs_succeeded=$(awk -F'\t' '$5 == 1' "$dir/worker-jobs.tsv" | wc -l)
    echo "k8s-demo: worker Jobs recorded in $dir/worker-jobs.tsv (name process task service-account succeeded failed): $worker_jobs_succeeded succeeded"
    observed=$(awk '$1 != "NAME" && NF >= 7 && $7 != "<none>" {print $2, $5, $6, $7}' \
        "$dir/worker-pods.watch.txt" | sort -u)
    printf '%s\n' "$observed" >"$dir/worker-pods.summary.txt"
    echo "k8s-demo: worker pods observed (process service-account image image-id; $dir/worker-pods.summary.txt):"
    printf '%s\n' "${observed:-none}"
    echo "k8s-demo: distinct worker processes: $(awk 'NF {print $1}' <<<"$observed" | sort -u | wc -l)"
    if awk 'NF && $2 != "workflow-worker" {found=1} END {exit !found}' <<<"$observed"; then
        echo "k8s-demo: a worker pod did not use service account workflow-worker" >&2
        return 1
    fi
    # Pod imageIDs must be the manifest digests of the images recorded in the run metadata.
    local digest unknown=0
    while read -r digest; do
        if ! grep -q "\"node_digest\": \"$digest\"" "$dir/image-manifest.json"; then
            echo "k8s-demo: a worker ran image $digest, which is not in the image manifest" >&2
            unknown=1
        fi
    done < <(grep -o '@sha256:[0-9a-f]\{64\}' <<<"$observed" | cut -c2- | sort -u)
    [[ $unknown -eq 0 ]] || return 1
    [[ -z $observed ]] || echo "k8s-demo: all worker pods ran as workflow-worker with image-manifest images"
}

# Machine-readable record of a run for scripts such as k8s-acceptance.sh: KEY=value lines in
# <run dir>/run.env, copied to data/local/k8s-runs/last-<mode>.env (written atomically).
write_record() {
    local dir=$1 mode=$2 status=$3 code=$4 tmp
    tmp="$runs_dir/.last-$mode.env.$$"
    {
        printf 'JOB=%s\nMODE=%s\nGIT_REVISION=%s\nRESUME=%s\nRUN_DIR=%s\n' "$job" "$mode" "$revision" "$resume" "$dir"
        printf 'STATUS=%s\nEXIT_CODE=%s\n' "$status" "$code"
        printf 'RESULTS_DIR=%s\nREPORT=%s\n' "$data_dir/results/adrb2-demo" "$data_dir/results/adrb2-demo/report/report.html"
        printf 'IMAGE_MANIFEST=%s\nCONTROLLER_LOG=%s\nNEXTFLOW_LOG=%s\n' \
            "$dir/image-manifest.json" "$dir/controller.log" "$data_dir/logs/$job/nextflow.log"
        printf 'LAUNCH_DIR=%s\n' "$data_dir/projects/$revision"
        printf 'WORKER_JOBS=%s\nWORKER_JOBS_JSON=%s\nWORKER_JOBS_SUCCEEDED=%s\n' \
            "$dir/worker-jobs.tsv" "$dir/worker-jobs.json" "${worker_jobs_succeeded:-}"
    } >"$tmp"
    cp "$tmp" "$dir/run.env"
    mv -f "$tmp" "$runs_dir/last-$mode.env"
}

submit() {
    local mode=$1 resume=$2 revision job dir deadline state code init pod report workers_ok
    require_tools
    [[ -f $template ]] || die "missing $template"
    [[ $runner_image =~ ^localhost/[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || die "invalid runner image '$runner_image'"
    [[ -z $resume || $resume == last || $resume =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] \
        || die "resume takes an optional Nextflow session ID (UUID), got '$resume'"
    ignored data/local/k8s-runs/ || die "data/local/k8s-runs/ is not ignored by Git"
    revision=$(selected_revision)
    preflight
    if [[ $mode == resume && ! -d "$data_dir/projects/$revision/.nextflow" ]]; then
        echo "k8s-demo: warning: no resume cache found at $data_dir/projects/$revision/.nextflow on the host" >&2
    fi
    # Also the Nextflow run name: lower case, unique, at most 63 characters.
    job="nf-$mode-$(cut -c1-12 <<<"${revision%-dirty}")"
    [[ $revision == *-dirty ]] && job+=d
    job+="-$(date -u +%Y%m%d-%H%M%S)"
    dir="$runs_dir/$job"
    mkdir -p "$dir"
    render "$job" "$mode" "$revision" "$resume" >"$dir/runner-job.yaml"
    grep -q '__[A-Z_]*__' "$dir/runner-job.yaml" && die "unrendered placeholder in $dir/runner-job.yaml"
    cp "$data_dir/image-manifest.json" "$dir/image-manifest.json"
    watchers=()
    worker_jobs_succeeded=""
    trap 'stop_watchers' EXIT
    watch_workers "$job" "$dir"
    kubectl --context "$context" apply -f "$dir/runner-job.yaml"
    write_record "$dir" "$mode" submitted ""
    echo "k8s-demo: submitted $job (revision $revision, evidence in $dir)"
    deadline=$((SECONDS + timeout_seconds + 120))
    state=$(wait_for_controller "$job" "$deadline")
    pod=$(pod_of "$job")
    if [[ -n $pod ]]; then
        kc logs "$pod" -c source >"$dir/source.log" 2>&1 || true
        cat "$dir/source.log"
    fi
    if [[ $state == running || $state == exited ]]; then
        timeout $((deadline - SECONDS)) kubectl --context "$context" -n "$namespace" logs -f "$pod" -c controller || true
        while (( SECONDS < deadline )); do
            [[ -n $(kc get job "$job" -o jsonpath='{.status.succeeded}{.status.failed}' 2>/dev/null || true) ]] && break
            sleep 3
        done
    fi
    stop_watchers
    [[ -z $pod ]] || kc logs "$pod" -c controller >"$dir/controller.log" 2>&1 || true
    kc get events --sort-by=.lastTimestamp >"$dir/events.txt" 2>&1 || true
    code=$(kc get pod "$pod" -o jsonpath='{.status.containerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)
    init=$(kc get pod "$pod" -o jsonpath='{.status.initContainerStatuses[0].state.terminated.exitCode}' 2>/dev/null || true)
    [[ -n $code || -z $init || $init == 0 ]] || code=$init
    workers_ok=1
    print_workers "$job" "$dir" || workers_ok=0
    if [[ $(kc get job "$job" -o jsonpath='{.status.succeeded}' 2>/dev/null || true) == 1 && $code == 0 && $workers_ok == 1 ]]; then
        report="$data_dir/results/adrb2-demo/report/report.html"
        [[ -f $report ]] || die "the controller succeeded but $report is missing on the host; check the storage mount"
        echo "$code" >"$dir/exit-code"
        write_record "$dir" "$mode" succeeded "$code"
        echo "k8s-demo: $job succeeded"
        echo "k8s-demo: report: /workspace/results/adrb2-demo/report/report.html (host: $report)"
        return 0
    fi
    report_failure "$job" "$dir"
    if [[ -z $(kc get job "$job" -o jsonpath='{.status.succeeded}{.status.failed}' 2>/dev/null || true) ]]; then
        # Never started or still running past the deadline: stop only this runner Job.
        echo "k8s-demo: deleting the unfinished runner Job $job" >&2
        kc delete job "$job" --wait=false >&2 || true
    fi
    [[ $code =~ ^[0-9]+$ && $code != 0 ]] || code=1
    echo "$code" >"$dir/exit-code"
    write_record "$dir" "$mode" failed "$code"
    echo "k8s-demo: $job failed with exit status $code (evidence in $dir)" >&2
    return "$code"
}

cmd_status() {
    require_tools
    local latest
    echo "== controller Jobs"
    kc get jobs -l app.kubernetes.io/component=nextflow-controller -L ligand-analysis/mode,ligand-analysis/git-revision \
        --sort-by=.metadata.creationTimestamp || true
    latest=$(kc get jobs -l app.kubernetes.io/component=nextflow-controller --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true)
    if [[ -n $latest ]]; then
        echo "== worker Jobs/Pods of $latest"
        kc get jobs,pods -l "$(worker_selector "$latest")" -o wide || true
        echo "== last controller log lines of $latest"
        kc logs "job/$latest" -c controller --tail=20 2>&1 || true
    fi
    echo "== storage"
    kc get pvc "$claim" || true
    echo "== image manifest"
    cat "$data_dir/image-manifest.json" 2>/dev/null || echo "none (run scripts/k8s-images.sh load)"
    echo "== report"
    if [[ -f $data_dir/results/adrb2-demo/report/report.html ]]; then
        echo "$data_dir/results/adrb2-demo/report/report.html"
    else
        echo "none yet"
    fi
}

case ${1:-} in
    run) [[ $# -eq 1 ]] || die "usage: $0 run"; submit run "" ;;
    resume) [[ $# -le 2 ]] || die "usage: $0 resume [SESSION_ID]"; submit resume "${2:-last}" ;;
    status) cmd_status ;;
    *) echo "usage: $0 run | resume [SESSION_ID] | status" >&2; exit 2 ;;
esac