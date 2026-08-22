"""Tests that the docs describe the package that actually exists.

Documentation drift is a proven failure mode here, not a hypothetical one: a
single review turned up a `TypeError` citing a release that had never shipped,
a docs page claiming to list all of ``__all__`` while omitting two exports, and
a link to a heading that had been renamed. None of it was catchable by reading,
because each claim was locally plausible - you only see the mismatch by
checking it against the code.

These tests check the mechanical claims (does this link resolve, is this export
mentioned at all). They cannot check that the prose is *true*, so they are a
floor, not a ceiling.
"""

from __future__ import annotations

import re
import subprocess
import types
from pathlib import Path

import pytest

import autotrainer

ROOT = Path(__file__).resolve().parent.parent

# Rendered pdoc output: generated, not authored, and not in the repo on a
# fresh checkout.
EXCLUDED_PREFIXES = ("docs/build/",)

# Where the public API is expected to be discussed. The API reference is
# generated from docstrings, so it is not part of this check.
API_DOC_FILES = ("README.md", "docs/README.md")
API_DOC_GLOBS = ("docs/guide/*.md",)

LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


def _tracked_markdown() -> list[Path]:
    """Every markdown file git knows about, minus generated output."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "*.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if proc.returncode != 0:  # pragma: no cover - sdist / export, not a checkout
        pytest.skip("not a git checkout")
    return [ROOT / line for line in proc.stdout.split() if not line.startswith(EXCLUDED_PREFIXES)]


def _slug(heading_text: str) -> str:
    """GitHub's heading -> anchor transform, near enough for link checking.

    Lowercase, drop punctuation, spaces become hyphens. Underscores survive
    (``train_step`` stays ``train_step``), which is why ``\\w`` is used rather
    than ``[a-z0-9]``.
    """
    text = heading_text.replace("`", "")
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text.strip().lower())


def _anchors(path: Path) -> set[str]:
    return {_slug(m.group(2)) for m in HEADING.finditer(path.read_text(encoding="utf-8"))}


def _links(path: Path):
    """(label, path_part, anchor) for each inline link that isn't external."""
    for label, target in LINK.findall(path.read_text(encoding="utf-8")):
        # `[text](path "title")` - the title is not part of the target.
        target = target.split()[0] if target.split() else ""
        if not target or target.startswith(("http://", "https://", "mailto:", "#!")):
            continue
        path_part, _, anchor = target.partition("#")
        yield label, path_part, anchor


def test_relative_links_resolve():
    """Every relative markdown link points at a file that exists."""
    broken = []
    for md in _tracked_markdown():
        for label, path_part, _anchor in _links(md):
            if not path_part:
                continue
            target = (md.parent / path_part).resolve()
            if not target.exists():
                broken.append(f"{md.relative_to(ROOT)}: [{label}] -> {path_part}")
    assert not broken, "broken relative links:\n  " + "\n  ".join(broken)


def test_link_anchors_resolve():
    """Every `#anchor` points at a heading that exists in the target file."""
    broken = []
    for md in _tracked_markdown():
        for label, path_part, anchor in _links(md):
            if not anchor:
                continue
            target = (md.parent / path_part).resolve() if path_part else md
            if not target.is_file() or target.suffix != ".md":
                continue
            if _slug(anchor) not in _anchors(target):
                broken.append(f"{md.relative_to(ROOT)}: [{label}] -> {path_part}#{anchor}")
    assert not broken, "links to nonexistent headings:\n  " + "\n  ".join(broken)


def test_public_api_is_mentioned_in_the_docs():
    """Nothing in `__all__` is undocumented enough to be invisible.

    This is a presence check, not a quality one - it catches the case where a
    new export ships and no prose ever mentions it (`augment_batch` and
    `TrainingMonitor` both did exactly that).
    """
    paths = [ROOT / name for name in API_DOC_FILES]
    for pattern in API_DOC_GLOBS:
        paths.extend(sorted(ROOT.glob(pattern)))
    prose = "".join(p.read_text(encoding="utf-8") for p in paths if p.is_file())

    missing = [name for name in autotrainer.__all__ if name != "__version__" and name not in prose]
    assert not missing, (
        "exported but never mentioned in README/docs: "
        + ", ".join(missing)
        + "\n(add prose, or drop it from __all__ if it is not really public)"
    )


def test_documented_attributes_are_actually_public():
    """The reverse of the check above: nothing is taught as public API while
    `__all__` says it is internal.

    `ThroughputMonitor` shipped exactly that way - a worked example in
    `docs/guide/monitors.md` for a name missing from `__all__`, which the
    README declares the boundary of the stable API. The presence check above
    only looks one way, so it could never catch it.
    """
    paths = [ROOT / name for name in API_DOC_FILES]
    for pattern in API_DOC_GLOBS:
        paths.extend(sorted(ROOT.glob(pattern)))
    prose = "".join(p.read_text(encoding="utf-8") for p in paths if p.is_file())

    # Only `autotrainer.X` call/attribute forms - prose naming a class in
    # passing is not a promise that it is importable.
    documented = set(re.findall(r"\bautotrainer\.([A-Za-z_][A-Za-z0-9_]*)", prose))
    leaked = sorted(
        name
        for name in documented
        if hasattr(autotrainer, name)
        and not name.startswith("_")
        and name not in autotrainer.__all__
        # Submodules are reachable but were never claimed as API surface.
        and not isinstance(getattr(autotrainer, name), types.ModuleType)
    )
    assert not leaked, (
        "documented as `autotrainer.X` but absent from __all__: "
        + ", ".join(leaked)
        + "\n(add it to __all__ if it is public, or stop documenting it as such)"
    )


def test_docs_do_not_promise_unreleased_versions():
    """No doc or message points at a version newer than the current one.

    The `train_loader=` removal notice said "removed in 1.0" for two releases
    while 1.0 had never shipped, sending anyone who hit it looking for a
    release that did not exist.
    """
    major, minor, *_ = (int(p) for p in autotrainer.__version__.split(".")[:2])
    sources = _tracked_markdown() + sorted((ROOT / "src" / "autotrainer").rglob("*.py"))
    offenders = []
    for path in sources:
        # The changelog and roadmap legitimately discuss future versions.
        if path.name == "CHANGELOG.md" or path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\bremoved in (\d+)\.(\d+)", text):
            claimed = (int(match.group(1)), int(match.group(2)))
            if claimed > (major, minor):
                offenders.append(
                    f"{path.relative_to(ROOT)}: claims 'removed in "
                    f"{match.group(1)}.{match.group(2)}' but this is "
                    f"{autotrainer.__version__}"
                )
    assert not offenders, "\n  ".join(["references to unshipped versions:", *offenders])
