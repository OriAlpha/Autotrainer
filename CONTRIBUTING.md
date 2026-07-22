# Contributing to autotrainer

Thanks for your interest! Contributions of all kinds are welcome: bug reports,
docs fixes, new backends, and features. This project follows the
[Code of Conduct](CODE_OF_CONDUCT.md) - please be respectful in all
interactions.

## Development setup

```bash
git clone https://github.com/OriAlpha/autotrainer
cd autotrainer

# Using uv (recommended):
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -e ".[dev,torch,sklearn,boosting,tune]"

# Or with standard pip:
pip install -e ".[dev,torch,sklearn,boosting,tune]"
```

The `tf` extra is optional and heavy - install it only when working on
TensorFlow code paths.

## Before opening a PR

All contributions go through pull requests: **branch off `main`, push, open a
PR, wait for CI, squash-merge.** Don't push directly to `main`.

### The CI gates

Every PR runs these jobs in `.github/workflows/ci.yml`:

| Job | What |
|---|---|
| `lint` | `ruff check` + `ruff format --check` |
| `typecheck` | `mypy src/autotrainer` (strict) |
| `test (3.9/3.11/3.13)` | `pytest tests/ -m "not cuda"` on Ubuntu, three Python versions |
| `test-tf` | TensorFlow-specific tests (single version; `tf` is heavy) |
| `docs` | pdoc API reference build (deployed to Pages on merge to `main`) |
| `test-cuda` | `pytest -m "cuda"` on a self-hosted GPU runner |

All must pass before merge (except `test-cuda`, which is best-effort — see below).

### Run the gates locally before pushing

```bash
ruff check src/ tests/          # lint
ruff format --check src/ tests/ # formatting
mypy src/autotrainer            # type checking
pytest tests/ -m "not cuda"     # tests + coverage (75% floor), skipping GPU tests
```

`pytest` runs with coverage automatically (configured in `pyproject.toml`).
If coverage drops below 75%, the run fails - add tests for any new behavior.

### The `cuda` pytest marker and the GPU runner

Tests that need a real CUDA GPU are marked `@pytest.mark.cuda` (see
`pyproject.toml`). On a CPU-only box they're deselected by `-m "not cuda"`
(above), so plain `pytest tests/` will show them as skipped/deselected rather
than failed - that's expected, not a problem.

The `test-cuda` CI job runs only the `cuda`-marked subset on a self-hosted
GPU runner (`runs-on: [self-hosted, gpu]`). It's **not required** for merge
(no branch-protection rule includes it), so PRs don't block if the runner is
offline. See [`RUNNER_SETUP.md`](RUNNER_SETUP.md) for one-time runner
registration, and [`NEXT_STEPS.md`](NEXT_STEPS.md) for the engineering
backlog. If you add CUDA-dependent behavior, mark the test `@pytest.mark.cuda`
and gate it with `skipif(not _has_cuda())` so it skips cleanly on CPU.

### Pre-commit hooks (optional)

You can also install the [pre-commit](https://pre-commit.com/) hooks to run
ruff automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

## Guidelines

- **One feature or fix per pull request** - keeps review focused.
- **Add tests for new behavior.** Untested public functions won't merge.
- **Keep coverage at or above 75%.** The CI enforces this.
- **New framework backends** go in `src/autotrainer/backends/` and are routed
  from the dispatcher in `__init__.py`. Follow the pattern of existing
  backends: auto-detect the environment, print every decision the backend
  makes, and keep everything overridable.
- **Lazy-import optional frameworks** inside function bodies, not at module
  top level - the base `pip install autotrainer` must work with no ML
  framework installed.
- **Update `CHANGELOG.md`** under an `[Unreleased]` heading.

## Public API and deprecation policy

The public API is exactly what `autotrainer.__all__` exports. Submodules
(`autotrainer.tuning`, `autotrainer.fitting`, ...) and `_`-prefixed helpers
are internal and may change without notice.

From 1.0 onward:

- Removing or changing public behavior requires a **deprecation cycle**: the
  old form keeps working and emits a `DeprecationWarning` for at least one
  minor release before removal, with the replacement named in the warning.
- Breaking changes land only in **major** versions; new features in minor
  versions; fixes in patches ([SemVer](https://semver.org/)).
- On-disk formats (the `fit()` checkpoint) carry a `format_version` and are
  rejected loudly - never silently misread - when incompatible.

## Commit messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <short description in imperative mood>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`.

Examples:

```
feat: add MirroredStrategy auto-selection for multi-GPU TensorFlow
fix: handle empty CUDA_VISIBLE_DEVICES in GPU count detection
docs: document SLURM environment variables in .env.example
```

## Reporting bugs

[Open an issue](https://github.com/OriAlpha/autotrainer/issues) and include
the output of `autotrainer doctor` and `autotrainer info` - they capture most
environment details needed to reproduce issues. The bug report template will
prompt you for these.

## Reporting security issues

**Do not open a public issue for security problems.** See
[SECURITY.md](SECURITY.md) for private reporting instructions.
