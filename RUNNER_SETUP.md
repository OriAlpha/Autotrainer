# Self-hosted GPU runner setup

This repo's `test-cuda` CI job runs the `cuda`-marked pytest tests on a
self-hosted runner with an actual GPU. This is the only CI job that can
catch CUDA-path bugs — `torch.cuda.is_available()` returning True with the
driver present but `device_count()` zero, stub defects, OS-specific
directory-permission issues, and so on.

This doc walks through registering a machine as the runner, **one time**.
Any maintainer who wants to provide a GPU runner for the project can follow
it; the steps are OS-agnostic in shape, with Windows specifics where they
bite.

---

## Prerequisites

- An **NVIDIA GPU and driver** capable of the CUDA version you intend to run.
- **Admin on the box** (needed to install the runner as a service).
- A system-wide Python install that the runner service account can read.
  On Windows, the runner service runs as `NT AUTHORITY\NETWORK SERVICE`,
  which **cannot read `C:\Users\<you>\...`** — so Python must live somewhere
  all accounts can read. See "Install Python system-wide" below.

You do **not** need a pre-installed torch — the `test-cuda` CI job installs
the CUDA-enabled torch wheel itself. The runner's only job is to provide the
GPU + driver + Python.

### Install Python system-wide (Windows, one-time)

We deliberately do **not** use `actions/setup-python` on a self-hosted
runner. On a self-hosted Windows box it downloads a Python zip and runs
`setup.ps1`, which (a) needs the PowerShell execution policy loosened to
`RemoteSigned` system-wide and (b) may trip Windows Defender quarantining the
downloaded `python.exe`. A pre-installed system-wide Python sidesteps both.

The fastest path is to copy a uv-managed Python to a system path and grant
NETWORK SERVICE read access. The repo ships a script that does all of this
idempotently — run it from an admin PowerShell:

```powershell
# Default: provisions 3.13 to C:\Python313 (matches env.PYTHON in ci.yml).
.\scripts\provision-runner-python.ps1

# Bumping the CI Python? Pass the new version; it reprovisions and reminds
# you to update env.PYTHON in ci.yml to match.
.\scripts\provision-runner-python.ps1 -Version 3.14
```

The script (`scripts/provision-runner-python.ps1`) installs the CPython via
`uv python install`, copies it to `C:\Python<ver>`, runs
`icacls /grant 'NETWORK SERVICE:(OI)(CI)RX' /T`, and verifies the result.
It is idempotent — re-running with the same `-Version` is a no-op if the
target is already the right interpreter. Requires `uv` on PATH and an
elevated shell.

<details><summary>Manual equivalent (what the script does under the hood)</summary>

```powershell
# From an admin PowerShell. Pick the version that matches env.PYTHON in
# .github/workflows/ci.yml (and keep the two in sync if you change either).
uv python install 3.13
Copy-Item -Recurse C:\Users\<you>\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none C:\Python313
icacls C:\Python313 /grant 'NETWORK SERVICE:(OI)(CI)RX' /T
& "C:\Python313\python.exe" --version   # should print Python 3.13.x
```

To change the Python version later: re-run with `-Version <new>` (or re-do
the copy + icacls to a new folder, e.g. `C:\Python314`), then update
`env.PYTHON` in `.github/workflows/ci.yml` to match.

</details>

## Step 1 — Create the runner in GitHub

GitHub will give you a config token + the exact download commands. **The
token is short-lived (~1 hour), so do step 2 immediately after step 1.**

1. Open: https://github.com/OriAlpha/Autotrainer/settings/actions/runners/new
2. Choose **Self-hosted** (not the GitHub-hosted Linux/Windows/macOS options).
3. Select your **operating system** and **architecture**.
4. You'll see a panel with four blocks: Download, Configure, Run, Install.
   Leave that page open — you'll paste the commands in step 2.

## Step 2 — Run the setup commands on the box

Open a terminal **as Administrator** (on Windows: right-click PowerShell →
"Run as administrator") and run the commands from the GitHub panel, in this
order:

```powershell
# 1. Create a directory for the runner (anywhere outside the repo).
mkdir actions-runner; cd actions-runner

# 2. Download the latest runner package (use the URL GitHub gives you - it
#    has a versioned filename like actions-runner-win-x64-2.xxx.0.zip).
#    The Invoke-WebRequest command GitHub shows is correct as-is.

# 3. Extract:
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory("$PWD\actions-runner-win-x64-*.zip", "$PWD")

# 4. Configure - use the EXACT ./config.cmd line GitHub shows (it carries
#    your short-lived token). When it asks for a runner name, pick something
#    descriptive of the machine. When it asks for labels, enter:
#
#       self-hosted,gpu
#
#    (Comma-separated, no spaces. The `gpu` label is what ci.yml selects on;
#    `self-hosted` is the default that lets you scope it further later.)
#
#    When it asks for work folder, accept the default (_work).

# 5. Install as a service so it survives logout/reboot (Windows):
./run.cmd

# (Optional) Auto-start on boot:
./svc.sh install
```

After step 5, refresh https://github.com/OriAlpha/Autotrainer/settings/actions/runners
— you should see your runner with a green "Idle" dot and labels `self-hosted, gpu`.

## Step 3 — Verify it works

Trigger a test-cuda run by pushing any commit to a PR branch (or pushing to
main). The job log should show the GPU sanity check passing:

```
GPU OK: <your GPU name> (capability (..., ...))
```

followed by the `cuda`-marked tests running. If you instead see "queued"
forever, the labels don't match (re-run `./config.cmd` with `self-hosted,gpu`).
If the GPU sanity check fails, the runner is registered but torch can't see
the GPU — check `CUDA_VISIBLE_DEVICES` isn't set in the service environment.

## Security notes

Self-hosted runners execute PR code from anyone who can open a PR against
this repo. For a public repo, **that's anyone on the internet.** Mitigations:

- The runner runs in the service account's context. **Do not put secrets on
  this machine** that PR code could exfiltrate.
- Consider runner groups and ephemeral container-based runners for stronger
  isolation when adding collaborators.
- Treat the runner host as untrusted: it should not have access to your
  GitHub credentials, signing keys, or other repos' secrets.

If you ever want to disable the job without unregistering the runner,
re-add `if: false` to the `test-cuda` job in `.github/workflows/ci.yml`.

## Troubleshooting

- **"Runner is offline"**: the service isn't running. Start it with
  `Start-Service actions.runner.*` (run `Get-Service actions.runner.*` to
  find the exact name) or re-run `./run.cmd` in the runner directory.
- **Job hangs at "queued"**: no runner matches the `[self-hosted, gpu]`
  label combo. Check the runner's labels on the settings page.
- **GPU sanity check fails with "No CUDA GPU visible"**: the service runs
  under a different account that may have different env. Open Services.msc,
  find the GitHub Actions runner service, check its environment / "Log On"
  tab. The simplest fix is to run the runner interactively (`./run.cmd`
  from your own shell) instead of as a service — same GPU visibility as
  your dev session.
