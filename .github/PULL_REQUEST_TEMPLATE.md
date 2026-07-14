## Summary

<!-- Brief description of what this PR changes and why. -->

## Type of change

<!-- Check one: -->
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing behavior to change)
- [ ] Documentation / community files only
- [ ] Refactor / tooling (no behavior change)

## Checklist

- [ ] I have read **CONTRIBUTING.md**.
- [ ] `ruff check src/ tests/` and `ruff format --check src/ tests/` pass.
- [ ] `mypy src/autotrainer` passes.
- [ ] `pytest tests/` passes (coverage stays at or above 75%).
- [ ] I added tests for any new behavior.
- [ ] I updated **CHANGELOG.md** under `[Unreleased]`.
- [ ] New framework backends are placed in `src/autotrainer/backends/` and
      routed from the dispatcher in `__init__.py`.

## Related issues

<!-- e.g. Closes #42. Leave blank if none. -->
