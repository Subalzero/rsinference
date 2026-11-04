[CmdletBinding()]
param(
    # all = cpp-windows then rust-windows in one container.
    [ValidateSet('all', 'cpp', 'rust')]
    [string] $Job = 'all',
    [string] $Image = 'infer-win-ci',
    # The container is a Hyper-V VM; its memory must be free host commit when it starts.
    [string] $Memory = '6g',
    [int] $Cpus = [Math]::Min(8, [Environment]::ProcessorCount)
)

<#
.SYNOPSIS
Run the Windows jobs of .github/workflows/develop.yml locally in a Windows container.

.DESCRIPTION
Requires Docker Desktop in Windows-containers mode. The working tree (including
uncommitted changes) is mounted read-only and copied into the container, excluding
local build output. Collected bridge DLLs land in scripts/docker-ci/out/bridge.
#>

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ctx = Join-Path $PSScriptRoot 'docker-ci'
$out = Join-Path $ctx 'out'
$name = "infer-ci-$Job"

# Native stderr (docker/curl progress) must not become terminating errors under Windows
# PowerShell 5.1 when output is redirected; failures are checked via exit codes instead.
$ErrorActionPreference = 'Continue'

docker build -t $Image $ctx
if ($LASTEXITCODE -ne 0) { throw "docker build failed with exit code $LASTEXITCODE." }

if ($Job -ne 'rust') { Remove-Item -Recurse -Force (Join-Path $out 'bridge') -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force $out | Out-Null
if (docker ps -aq --filter "name=^$name$") { docker rm -f $name | Out-Null }

docker run --rm --name $name --cpus $Cpus --memory $Memory --storage-opt size=120G `
    -v "${repo}:C:\src:ro" -v "${out}:C:\out" `
    $Image pwsh -NoProfile -File C:\ci\run-ci.ps1 -Job $Job
exit $LASTEXITCODE
