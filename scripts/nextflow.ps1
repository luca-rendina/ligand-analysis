<#
.SYNOPSIS
Run Nextflow for this repository inside the ligand-runner image on Windows with a Podman machine.

.DESCRIPTION
Nextflow does not run natively on Windows. This wrapper starts the ligand-runner container and
gives it the Podman machine's API socket, so Nextflow launches every task as a sibling container
(ligand-ml, ligand-chem) on the same Podman machine. The repository is mounted at the path the
machine sees (C:\x\y -> /mnt/c/x/y) so work directories resolve identically for the controller
and for the task containers. All arguments are passed to "nextflow run main.nf".

.EXAMPLE
.\scripts\nextflow.ps1 -profile podman -params-file configs/demo.yaml
.\scripts\nextflow.ps1 -profile podman,offline -params-file configs/demo.yaml -resume
#>
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($repo -notmatch '^[A-Za-z]:\\') { throw "Expected a Windows drive path, got $repo" }
$machinePath = '/mnt/' + $repo.Substring(0, 1).ToLower() + ($repo.Substring(2) -replace '\\', '/')
$socket = (podman info --format '{{.Host.RemoteSocket.Path}}') -replace '^unix://', ''
if (-not $socket) { throw 'Cannot find the Podman machine API socket; is the machine running (podman machine start)?' }
$revision = git -C $repo describe --always --dirty 2>$null
if (-not $revision) { $revision = 'unknown' }
podman run --rm --user root --security-opt label=disable `
    -v "${socket}:/run/podman/podman.sock" `
    -v "${repo}:${machinePath}" -w $machinePath `
    localhost/ligand-runner:dev nextflow run main.nf @args --git_revision $revision
exit $LASTEXITCODE
