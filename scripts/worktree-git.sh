#!/usr/bin/env bash
# Make Windows linked-worktree Git metadata usable from a Linux Podman Machine.
configure_worktree_git() {
    local repo=$1 pointer drive rest git_dir
    if [[ -d "$repo/.git" ]]; then
        git_dir="$repo/.git"
    elif [[ -f "$repo/.git" ]]; then
        IFS= read -r pointer < "$repo/.git" || [[ -n "$pointer" ]] ||
            { echo "Cannot read linked-worktree pointer: $repo/.git" >&2; return 1; }
        pointer=${pointer%$'\r'}
        [[ "$pointer" == 'gitdir: '* ]] ||
            { echo "Invalid linked-worktree pointer: $repo/.git" >&2; return 1; }
        git_dir=${pointer#gitdir: }
        if [[ "$git_dir" =~ ^([A-Za-z]):[/\\](.*)$ ]]; then
            drive=${BASH_REMATCH[1],,}
            rest=${BASH_REMATCH[2]//\\//}
            git_dir="/mnt/$drive/$rest"
        elif [[ "$git_dir" != /* ]]; then
            git_dir="$repo/$git_dir"
        fi
    else
        echo "Missing Git metadata: $repo/.git" >&2
        return 1
    fi
    [[ -d "$git_dir" ]] ||
        { echo "Git metadata is not visible from Linux: $git_dir" >&2; return 1; }
    export GIT_DIR="$git_dir" GIT_WORK_TREE="$repo"
    local actual
    actual=$(git -C "$repo" rev-parse --show-toplevel) || return 1
    [[ "$(cd "$actual" && pwd -P)" == "$(cd "$repo" && pwd -P)" ]] ||
        { echo "Git metadata belongs to another worktree: $actual" >&2; return 1; }
}
