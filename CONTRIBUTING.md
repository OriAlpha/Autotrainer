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

All three gates must pass. Run them locally before pushing:

```bash
ruff check src/ tests/          # lint
ruff format --check src/ tests/ # formatting
mypy src/autotrainer            # type checking
pytest tests/                   # tests + coverage (75% floor)
```

`pytest` runs with coverage automatically (configured in `pyproject.toml`).
If coverage drops below 75%, the run fails - add tests for any new behavior.

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
