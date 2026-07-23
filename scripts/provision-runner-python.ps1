<#
.SYNOPSIS
    Provision a system-wide Python for the self-hosted GPU runner.

.DESCRIPTION
    The test-cuda CI job (see .github/workflows/ci.yml) uses a system-wide
    Python at C:\Python<ver>\python.exe because the runner service runs as
    NT AUTHORITY\NETWORK SERVICE, which cannot read a per-user install. This
    script is the single source of truth for that path: it installs the
    requested CPython via uv, copies it to the system location, grants
    NETWORK SERVICE read+execute, and verifies the result.

    Run from an elevated (admin) PowerShell on the runner box. Idempotent:
    re-running with the same -Version skips the work if the target is already
    a working interpreter. Changing -Version re-provisions to the new folder
    and prints a reminder to update env.PYTHON in ci.yml to match.

    Requires `uv` on PATH (https://docs.astral.sh/uv/).

.PARAMETER Version
    CPython version to provision (major.minor). Default "3.13" to match the
    current env.PYTHON in ci.yml. Must be available via `uv python install`.

.PARAMETER InstallRoot
    System root to install into. Default "C:\Python<ver-stripped>" (e.g.
    C:\Python313). The runner service path in ci.yml must point here.

.EXAMPLE
    # Default: provision 3.13 to C:\Python313
    .\scripts\provision-runner-python.ps1

.EXAMPLE
    # Provision 3.14 when bumping the CI Python
    .\scripts\provision-runner-python.ps1 -Version 3.14
#>

[CmdletBinding()]
param(
    [string]$Version = "3.13",
    [string]$InstallRoot = ""
)

$ErrorActionPreference = "Stop"

# --- preflight ---------------------------------------------------------------

# uv must be available to fetch the standalone CPython build.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found on PATH. Install it: https://docs.astral.sh/uv/getting-started/installation/"
}

# icacls + writing under C:\ require elevation. Detect early with a clear
# message rather than failing mid-copy with an opaque ACL error.
$currentUser = [Security.Principal.WindowsPrincipal]::new(
    [Security.Principal.WindowsIdentity]::GetCurrent()
)
if (-not $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "This script must run as Administrator (needed for icacls and C:\ writes). Re-launch PowerShell elevated."
}

# Derive the install root from the version: "3.13" -> "C:\Python313".
if (-not $InstallRoot) {
    $verNoDot = $Version -replace '\.', ''
    $InstallRoot = "C:\Python$verNoDot"
}
$targetExe = Join-Path $InstallRoot "python.exe"

Write-Host "[provision] target: $targetExe (version $Version)"

# --- idempotency check -------------------------------------------------------

# If the target already resolves to the requested version, there's nothing
# to do. This makes re-runs cheap and safe (no re-copy of a 100MB tree).
if (Test-Path $targetExe) {
    $existing = & $targetExe --version 2>$null
    if ($LASTEXITCODE -eq 0 -and $existing -match $Version) {
        Write-Host "[provision] $InstallRoot already at $existing - nothing to do."
        return
    }
    Write-Host "[provision] $InstallRoot exists but is '$existing' - replacing."
}

# --- install + copy ----------------------------------------------------------

# uv's standalone CPython lands under its data dir. The exact folder name
# encodes version + platform (e.g. cpython-3.13.x-windows-x86_64-none), so
# resolve it via `uv python dir` + a glob rather than hard-coding it.
Write-Host "[provision] installing CPython $Version via uv..."
& uv python install $Version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "uv python install failed for $Version" }

$uvPythonDir = & uv python dir
if ($LASTEXITCODE -ne 0) { throw "could not resolve uv python dir" }

# Match the major.minor prefix; uv uses cpython-<ver>-windows-x86_64-none.
$srcPattern = Join-Path $uvPythonDir "cpython-$Version*-windows-*-none"
$candidates = Get-Childitem -Path $srcPattern -Directory -ErrorAction SilentlyContinue
if (-not $candidates) {
    throw "No uv CPython install found matching $srcPattern. Run 'uv python install $Version' and check the output."
}
# If multiple patch versions exist, take the most recently written one.
$src = $candidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "[provision] source: $($src.FullName)"

if (Test-Path $InstallRoot) {
    Write-Host "[provision] removing stale $InstallRoot..."
    Remove-Item -Recurse -Force $InstallRoot
}

Write-Host "[provision] copying to $InstallRoot..."
Copy-Item -Recurse $src.FullName $InstallRoot

# --- ACL: let the runner service account read + execute ----------------------

# NETWORK SERVICE is the identity the GitHub Actions Windows runner service
# runs as; without RX it can't invoke python.exe. (OI)(CI) makes it inherit
# to the whole tree (site-packages etc.).
Write-Host "[provision] granting NETWORK SERVICE read+execute on $InstallRoot..."
& icacls $InstallRoot /grant 'NETWORK SERVICE:(OI)(CI)RX' /T | Out-Null
if ($LASTEXITCODE -ne 0) { throw "icacls failed - check the path and your elevation." }

# --- verify ------------------------------------------------------------------

$final = & $targetExe --version
if ($LASTEXITCODE -ne 0 -or $final -notmatch $Version) {
    throw "Verification failed: $targetExe reported '$final' (expected $Version)."
}
Write-Host "[provision] OK: $final at $targetExe"

# Point the operator at the one remaining manual step so the new interpreter
# is actually used by CI. Use a sentinel string that grep-able from docs.
$ciPath = ".github/workflows/ci.yml"
if (Test-Path $ciPath) {
    $ciHasPath = Select-String -Path $ciPath -Pattern ([regex]::Escape($InstallRoot.Replace('\', '\\'))) -Quiet
    if (-not $ciHasPath) {
        Write-Host ""
        Write-Host "[provision] REMINDER: update env.PYTHON in $ciPath to point at $targetExe (currently points elsewhere)."
    }
}
