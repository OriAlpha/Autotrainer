"""Print one version's CHANGELOG section, for use as GitHub Release notes.

Used by the `release` job in ci.yml. Kept as a script rather than inline
shell because the version string contains dots (which are regex
metacharacters) and the section runs to the next `## [` heading - both are
easy to get subtly wrong in a one-liner, and getting them wrong produces
release notes that are empty or run into the previous version.

    python .github/scripts/changelog_section.py 0.13.0
    python .github/scripts/changelog_section.py 0.14.0rc1 --fallback-unreleased

Prereleases need the fallback: an RC ships whatever is currently unreleased,
and Keep a Changelog has no place to write a `## [0.14.0rc1]` heading for it.
Without the flag a missing section is still a hard failure - shipping a
release with empty notes is the thing this script exists to prevent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

UNRELEASED = "Unreleased"


def section(text: str, version: str) -> str | None:
    """The body of ``## [version] ...`` up to the next version heading."""
    match = re.search(
        rf"^## \[{re.escape(version)}\].*?(?=^## \[|\Z)",
        text,
        re.S | re.M,
    )
    return match.group(0).strip() if match else None


def _has_content(body: str) -> bool:
    """True when the section is more than just its heading."""
    lines = [line for line in body.splitlines()[1:] if line.strip()]
    return bool(lines)


def resolve(text: str, version: str, *, fallback_unreleased: bool) -> tuple[str, str]:
    """Return (notes, source) or raise ValueError with the reason."""
    body = section(text, version)
    if body is not None and _has_content(body):
        return body, version
    if body is not None:
        raise ValueError(f"the [{version}] section is empty")
    if not fallback_unreleased:
        raise ValueError(f"no CHANGELOG section for {version}")

    body = section(text, UNRELEASED)
    if body is None:
        raise ValueError(f"no [{version}] section and no [{UNRELEASED}] to fall back to")
    if not _has_content(body):
        raise ValueError(f"no [{version}] section and [{UNRELEASED}] is empty")
    # Retitle so the release page names the prerelease rather than saying
    # "Unreleased", which is true of the changelog but not of the release.
    heading, _, rest = body.partition("\n")
    del heading
    return f"## [{version}] - prerelease of unreleased changes\n{rest}", UNRELEASED


def main(argv: list[str]) -> int:
    args = argv[1:]
    fallback = "--fallback-unreleased" in args
    if fallback:
        args.remove("--fallback-unreleased")
    if len(args) != 1:
        print(
            f"usage: {Path(argv[0]).name} VERSION [--fallback-unreleased]",
            file=sys.stderr,
        )
        return 2
    version = args[0]
    changelog = Path("CHANGELOG.md")
    if not changelog.is_file():
        print("CHANGELOG.md not found (run from the repo root)", file=sys.stderr)
        return 1
    try:
        body, source = resolve(
            changelog.read_text(encoding="utf-8"), version, fallback_unreleased=fallback
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if source != version:
        print(f"note: using the [{source}] section for {version}", file=sys.stderr)
    # The changelog carries non-ASCII (R^2, arrows); force UTF-8 rather than
    # inherit a locale-dependent stdout encoding.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
