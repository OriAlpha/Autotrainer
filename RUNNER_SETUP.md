# Self-hosted GPU runner setup

This repo's `test-cuda` CI job runs the `cuda`-marked pytest tests on a
self-hosted runner with an actual GPU. This is the only CI job that can
catch CUDA-path bugs — `torch.cuda.is_available()` returning True with the
driver present but `device_count()` zero, `_pretend_cuda` stub defects,
OS-specific directory-permission issues, and so on. Three such bugs were
found in PRs #1–#3 by running tests on a real GPU, never by CPU CI.

This doc walks through registering your machine as the runner, **one time**.

---

## Prerequisites (already met on this box)

- NVIDIA GPU and driver: **RTX 5070 Laptop GPU, driver 610.74** (CUDA 13.3 capable).
- Admin on the box (needed to install the runner as a Windows service).

You do **not** need a pre-installed Python or torch — the `test-cuda` CI job
uses `actions/setup-python` to install a known Python into the runner's
workspace, then installs the cu128 nightly torch itself. The runner's only
job is to provide the GPU + driver.

## Step 1 — Create the runner in GitHub

GitHub will give you a config token + the exact download commands. **The
token is short-lived (~1 hour), so do step 2 immediately after step 1.**

1. Open: https://github.com/OriAlpha/Autotrainer/settings/actions/runners/new
2. Choose **Self-hosted** (not the GitHub-hosted Linux/Windows/macOS options).
3. Select **Windows** as the operating system, **x64** as the architecture.
4. You'll see a panel with four blocks: Download, Configure, Run, Install.
   Leave that page open — you'll paste the commands in step 2.

## Step 2 — Run the setup commands on this box

Open **PowerShell as Administrator** (right-click → "Run as administrator")
and run the commands from the GitHub panel, in this order:

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
#    your short-lived token). When it asks for a runner name, use something
#    like `suhas-rtx5070`. When it asks for labels, enter:
#
#       self-hosted,gpu
#
#    (Comma-separated, no spaces. The `gpu` label is what ci.yml selects on;
#    `self-hosted` is the default that lets you scope it further later.)
#
#    When it asks for work folder, accept the default (_work).

# 5. Install as a Windows service so it survives logout/reboot:
./run.cmd

# (Optional) Auto-start on boot:
./svc.sh install
```

After step 5, refresh https://github.com/OriAlpha/Autotrainer/settings/actions/runners
— you should see your runner with a green "Idle" dot and labels `self-hosted, gpu`.

## Step 3 — Verify it works

Trigger a test-cuda run by pushing any commit to a PR branch (or pushing to
main). The job log should show:

```
GPU OK: NVIDIA GeForce RTX 5070 Laptop GPU (capability (12, 0))
```

followed by the `cuda`-marked tests running. If you instead see "queued"
forever, the labels don't match (re-run `./config.cmd` with `self-hosted,gpu`).
If the GPU sanity check fails, the runner is registered but torch can't see
the GPU — check `CUDA_VISIBLE_DEVICES` isn't set in the service environment.

## Security notes

Self-hosted runners execute PR code from anyone who can open a PR against
this repo. For a public repo, **that's anyone on the internet.** Mitigations:

- This is your personal repo, but consider it before adding collaborators.
- The runner runs in the service account's context. Don't put secrets on
  this machine that PR code could exfiltrate.
- For stronger isolation, GitHub supports runner groups and ephemeral
  container-based runners; out of scope here but worth knowing.

If you ever want to disable the job without unregistering the runner,
re-add `if: false` to the `test-cuda` job in `.github/workflows/ci.yml`.

## Troubleshooting

- **"Runner is offline"**: the Windows service isn't running. Start it with
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
