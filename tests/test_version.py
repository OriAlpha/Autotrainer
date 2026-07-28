"""Tests that the version string is declared consistently in every place.

The release job in ci.yml reads the version from pyproject.toml alone: it is
what decides whether a tag is new, what names the GitHub release, and what
version the wheel is built under. `__version__` is a second, hand-maintained
copy, so bumping one and forgetting the other cuts a release whose package
reports the *previous* version at runtime.

That mistake is unfixable after the fact - PyPI versions are immutable, so a
one-character typo costs a whole patch release to correct. These tests run in
the `test` job, which gates `release`, so drift fails the build before a tag
exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import autotrainer

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject():
    # Parsed with a regex rather than tomllib because the test matrix includes
    # 3.9 and tomllib landed in 3.11. This is deliberately the same expression
    # the release job uses, so the test fails on anything that would confuse it.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "(.*)"$', text, re.M)
    assert match is not None, 'no top-level `version = "..."` in pyproject.toml'
    assert autotrainer.__version__ == match.group(1), (
        f"__init__.py says {autotrainer.__version__}, pyproject.toml says {match.group(1)}"
    )


def test_security_policy_covers_current_version():
    # SECURITY.md promises support for the current minor line only, so the
    # table goes stale silently on every minor bump.
    major, minor, _ = autotrainer.__version__.split(".", 2)
    supported = f"| {major}.{minor}.x"
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert supported in text, (
        f"SECURITY.md has no `{supported}` row for the current version {autotrainer.__version__}"
    )
