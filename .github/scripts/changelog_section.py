"""Print one version's CHANGELOG section, for use as GitHub Release notes.

Used by the `release` job in ci.yml. Kept as a script rather than inline
shell because the version string contains dots (which are regex
metacharacters) and the section runs to the next `## [` heading - both are
easy to get subtly wrong in a one-liner, and getting them wrong produces
release notes that are empty or run into the previous version.

    python .github/scripts/changelog_section.py 0.13.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def section(text: str, version: str) -> str | None:
    """The body of ``## [version] ...`` up to the next version heading."""
    match = re.search(
        rf"^## \[{re.escape(version)}\].*?(?=^## \[|\Z)",
        text,
        re.S | re.M,
    )
    return match.group(0).strip() if match else None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} VERSION", file=sys.stderr)
        return 2
    version = argv[1]
    changelog = Path("CHANGELOG.md")
    if not changelog.is_file():
        print("CHANGELOG.md not found (run from the repo root)", file=sys.stderr)
        return 1
    body = section(changelog.read_text(encoding="utf-8"), version)
    if body is None:
        print(f"no CHANGELOG section for {version}", file=sys.stderr)
        return 1
    # The changelog carries non-ASCII (R^2, arrows); force UTF-8 rather than
    # inherit a locale-dependent stdout encoding.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
