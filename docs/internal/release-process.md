# Internal release process

**Releasing is now "merge a version bump."** Everything after the merge — the
tag, the GitHub Release, the release notes, the PyPI upload — is automatic.

> This replaces the old manual runbook, which said merging "does NOT
> automatically create a Git tag or GitHub Release" and told you to run
> `gh release create` by hand. That has been false since PR #20. Following the
> old steps today would fail CI on the version-parity test and then hand-create
> a release the automation had already made.

---

## 1. Bump four files

All four, in one PR. The first three are **enforced by
`tests/test_version.py`** — miss one and CI goes red before anything is tagged,
which is the point.

| File | Change |
|---|---|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `src/autotrainer/__init__.py` | `__version__ = "X.Y.Z"` |
| `SECURITY.md` | support table row → `\| X.Y.x \| :white_check_mark: \|` |
| `CHANGELOG.md` | move `## [Unreleased]` items into `## [X.Y.Z] - YYYY-MM-DD` |

The changelog step is not optional either: the release job pulls its notes from
the `## [X.Y.Z]` section and **fails the job if that section is missing or
empty**, rather than publishing a release with a blank body.

```bash
git checkout main && git pull
git checkout -b release/X.Y.Z
# ...edit the four files...
git commit -am "chore(release): X.Y.Z"
git push -u origin release/X.Y.Z
gh pr create --base main
```

## 2. Merge the PR

Branch protection requires an approving review, and GitHub will not let you
approve your own PR. As a solo maintainer you have two options:

- **Settings → Branches → Edit `main` → "Allow administrators to bypass
  required pull requests"**, merge, then turn it back off.
- Or drop `required_approving_review_count` to 0 permanently and rely on the
  status checks as the gate.

Relax the narrowest thing you can and restore it immediately — leaving
`enforce_admins` off is a much wider hole than it looks, since it disables
every protection for admins, not just the review requirement.

## 3. Everything else happens on its own

On push to `main`:

```
version-info          reads the version, decides stable vs prerelease
      |
   release            tag exists already?  yes -> stop (the common case)
      |                                     no -> write notes, create the
      |                                            tag + GitHub Release
   publish            build sdist+wheel, upload via trusted publishing
```

`release` runs only after `lint`, `test`, `typecheck`, and `test-tf` pass. It
is serialized (`concurrency: release-main`) so two merges landing together
can't race to create the same tag.

**Why `publish` runs in the same workflow run** rather than being triggered by
the release it just created: GitHub does not start workflow runs for events
raised with `GITHUB_TOKEN`. A release created by CI fires nothing. Wiring it
the obvious way would produce green runs, correct-looking release pages, and
nothing on PyPI.

## 4. Prereleases

*(Pending PR #23 — inert until the two setup steps below are done.)*

A version that is not exactly `X.Y.Z` — `0.14.0rc1`, `0.14.0a1`, `.dev`,
`.post` — is marked as a prerelease on GitHub and published to **TestPyPI**
instead of PyPI. Use it to validate a real installable wheel (the SLURM runbook
needs this) without spending a version number, since PyPI versions are
immutable and can never be replaced.

A prerelease has no changelog section of its own, so notes fall back to
`[Unreleased]`, retitled.

**One-time setup, not yet done:**

1. Configure a trusted publisher on test.pypi.org: repo `OriAlpha/Autotrainer`,
   workflow `ci.yml`, environment `testpypi`.
2. Create the `testpypi` environment in repo settings.

## 5. Verify

```bash
gh run list --branch main -L 1
gh release view "vX.Y.Z"
pip index versions autotrainer
```

In the `release` job log, the line you want is either `vX.Y.Z is new - cutting
a release` or, for an ordinary push, `vX.Y.Z is already tagged - nothing to
release`. Note that `grep` on the raw log also matches the *echoed script
source*; the real output lines are the ones without the `\033[36;1m` escape.

## 6. When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `test` red on `test_version_matches_pyproject` | bumped one file, not all | bump the rest |
| `release` fails at the notes step | no `## [X.Y.Z]` section, or it's empty | write the changelog section |
| `release` fails at `gh release create` with 422 | tag already exists | someone/something already released it; nothing to do |
| Release exists, PyPI does not | upload failed after the release was created | `workflow_dispatch` **from the tag** (see below) |
| Nothing happened at all | version wasn't bumped | expected — the tag already existed |

The retry path is `workflow_dispatch`, and it is restricted to tag refs on
purpose: dispatching from `main` would build whatever `main` is *now* and
upload it under the version in `pyproject.toml`, quietly shipping post-release
commits as the released version.

Note that `workflow_dispatch` only works from refs whose workflow file already
carries the trigger — so tags cut before PR #21 can't use it.

## 7. What is still manual

- Deciding the version number (SemVer 0.x: minor for new public API or
  observable behaviour change, patch for fixes).
- Writing the changelog entries.
- Merging the PR.
- Head branches are **not** auto-deleted on merge; that's a repo setting nobody
  has turned on, so branches accumulate.
