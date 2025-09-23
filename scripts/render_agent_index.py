#!/usr/bin/env python3
"""
Generate a Documentation Index section in docs/PROJECT_OVERVIEW.md.

Usage:
    uv run python scripts/render_agent_index.py

The script scans:
- Core docs in docs/ (PROJECT_OVERVIEW.md, REPO_MAP.md, DATA_ARTIFACTS.md, ROADMAP.md)
- Per-repo guides in docs/guides/
- Process references in docs/processes/

It then writes a categorized bullet list between markers in PROJECT_OVERVIEW.md.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable

DOCS_DIR = pathlib.Path("docs")
OVERVIEW_PATH = DOCS_DIR / "PROJECT_OVERVIEW.md"
START_MARKER = "<!-- START INDEX -->"
END_MARKER = "<!-- END INDEX -->"

CORE_DOCS = [
    ("Project Overview", DOCS_DIR / "PROJECT_OVERVIEW.md", "Mission, architecture snapshot."),
    ("Repo Map", DOCS_DIR / "REPO_MAP.md", "Purpose + entry points for each repo."),
    ("Data Artifacts", DOCS_DIR / "DATA_ARTIFACTS.md", "Existing outputs and planned schemas."),
    ("Roadmap", DOCS_DIR / "ROADMAP.md", "Near/long-term priorities."),
]


@dataclass
class DocEntry:
    title: str
    path: pathlib.Path
    description: str | None = None

    def to_markdown(self, base_path: pathlib.Path | None = None) -> str:
        rel_path = self.path if base_path is None else self.path.relative_to(base_path)
        base = f"[{self.title}]({rel_path.as_posix()})"
        if self.description:
            return f"- {base} — {self.description}"
        return f"- {base}"


def _list_guides() -> list[DocEntry]:
    guides_dir = DOCS_DIR / "guides"
    entries: list[DocEntry] = []
    if guides_dir.exists():
        for path in sorted(guides_dir.glob("AGENT_GUIDE_*.md")):
            title = path.stem.replace("AGENT_GUIDE_", "").replace("_", " ")
            entries.append(DocEntry(title=title, path=path, description="Per-repo onboarding guide."))
    return entries


def _list_process_docs() -> list[DocEntry]:
    proc_dir = DOCS_DIR / "processes"
    entries: list[DocEntry] = []
    if proc_dir.exists():
        for path in sorted(proc_dir.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            title = path.stem.replace("_", " ")
            entries.append(DocEntry(title=title, path=path))
    return entries


def build_index() -> str:
    lines: list[str] = []
    lines.append("## Documentation Index")
    lines.append(START_MARKER)
    lines.append("- **Core Docs**")
    for entry in CORE_DOCS:
        lines.append(f"  {DocEntry(*entry).to_markdown(DOCS_DIR.parent)}")

    guides = _list_guides()
    if guides:
        lines.append("\n- **Per-Repo Guides**")
        for entry in guides:
            lines.append(f"  {entry.to_markdown(DOCS_DIR.parent)}")

    processes = _list_process_docs()
    if processes:
        lines.append("\n- **Process References**")
        for entry in processes:
            lines.append(f"  {entry.to_markdown(DOCS_DIR.parent)}")

    lines.append(END_MARKER)
    lines.append("")
    return "\n".join(lines)


def replace_section(original: str, new_section: str) -> str:
    if START_MARKER in original and END_MARKER in original:
        before, rest = original.split(START_MARKER, 1)
        _, after = rest.split(END_MARKER, 1)
        return f"{before}{new_section}{after}"
    else:
        # Append at end if markers missing
        return original.strip() + "\n\n" + new_section + "\n"


def main() -> None:
    if not OVERVIEW_PATH.exists():
        raise SystemExit(f"Missing {OVERVIEW_PATH}")

    new_section = build_index()
    original = OVERVIEW_PATH.read_text()
    updated = replace_section(original, new_section)
    OVERVIEW_PATH.write_text(updated)
    print("Updated documentation index in PROJECT_OVERVIEW.md")


if __name__ == "__main__":
    main()
