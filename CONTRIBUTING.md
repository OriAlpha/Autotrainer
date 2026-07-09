# Contributing to autotrainer

Thanks for your interest! Contributions of all kinds are welcome: bug reports,
docs fixes, new backends, and features.

## Development setup

```bash
git clone https://github.com/OriAlpha/autotrainer
cd autotrainer
pip install -e ".[dev]" scikit-learn xgboost lightgbm
pytest tests/ -v
```

## Guidelines

- Run `pytest tests/` before opening a PR; add tests for new behavior.
- One feature or fix per pull request.
- New framework backends go in `src/autotrainer/backends/` and are routed
  from the dispatcher in `__init__.py`. Follow the pattern of existing
  backends: auto-detect the environment, print every decision the backend
  makes, and keep everything overridable.
- Update `CHANGELOG.md` under an `[Unreleased]` heading.

## Reporting bugs

Please include the output of `autotrainer doctor` and `autotrainer info` -
they capture most environment details needed to reproduce issues.
