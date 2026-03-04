#!/usr/bin/env python3
# Auto-generate a minimal Jupyter Book _toc.yml based on the repository layout.
# - Prefers a "modules/" directory containing sub-directories for chapters.
# - Otherwise, uses top-level markdown files and directories as chapters.
# - Uses natural (human) sorting.
# - No third-party dependencies.
#
# USAGE:
#     python generate_toc.py
#
# It writes _toc.yml in the current working directory.

import os
import sys
from pathlib import Path
import re

EXCLUDE_DIRS = {
    "_build", ".git", ".github", ".venv", "venv", "env",
    "node_modules", ".ipynb_checkpoints", "__pycache__",
    ".mypy_cache", ".pytest_cache","settings"
}

INCLUDE_EXTS = {".md", ".ipynb"}


def natural_key(s: str):
    """Split a string into ints/strings for natural sorting (2 < 10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def rel_no_ext(path: Path, root: Path) -> str:
    """Return POSIX-style relative path without extension, for Jupyter Book TOC."""
    p = path.relative_to(root).as_posix()
    pl = p.lower()
    if pl.endswith(".md") or pl.endswith(".ipynb"):
        p = p[: p.rfind(".")]
    return p


def find_root_file(root: Path):
    """Pick a root (book 'root') file: prefer index.md, else README.md, else first top-level .md."""
    index = root / "index.md"
    if index.exists():
        return index
    readme = root / "README.md"
    if readme.exists():
        return readme
    top_md = sorted([p for p in root.glob("*.md") if p.name.lower() != "_toc.md"], key=lambda p: p.name.lower())
    return top_md[0] if top_md else None


def list_markdown_files(d: Path):
    """List immediate markdown/ipynb files inside directory d (no recursion), sorted naturally."""
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in INCLUDE_EXTS]
    files.sort(key=lambda p: natural_key(p.name))
    return files


def build_chapter_for_dir(d: Path, repo_root: Path):
    """Create a chapter dict for directory d. Use intro.md/index.md as chapter file if present."""
    candidates = list_markdown_files(d)
    intro = d / "intro.md"
    index = d / "index.md"
    chapter_file = None
    if intro.exists():
        chapter_file = intro
    elif index.exists():
        chapter_file = index
    elif candidates:
        chapter_file = candidates[0]

    if not chapter_file:
        return None  # no content

    sections = [p for p in candidates if p != chapter_file]

    chapter = {"file": rel_no_ext(chapter_file, repo_root)}
    if sections:
        chapter["sections"] = [{"file": rel_no_ext(p, repo_root)} for p in sections]
    return chapter


def discover_chapters(repo_root: Path):
    """Discover chapters/sections from the repo layout.

    Strategy:
      1) If 'modules/' exists, treat each subdir of modules as a chapter.
      2) Else: treat top-level dirs (with md files) as chapters; top-level md files as chapters.
    """
    chapters = []
    modules_dir = repo_root / "modules"
    if modules_dir.exists() and modules_dir.is_dir():
        subdirs = [p for p in modules_dir.iterdir() if p.is_dir() and p.name not in EXCLUDE_DIRS and not p.name.startswith(".")]
        subdirs.sort(key=lambda p: natural_key(p.name))
        for sd in subdirs:
            chap = build_chapter_for_dir(sd, repo_root)
            if chap:
                chapters.append(chap)
        if chapters:
            return chapters

    # Generic fallback
    # 1) top-level directories with md content
    top_dirs = [p for p in repo_root.iterdir() if p.is_dir() and p.name not in EXCLUDE_DIRS and not p.name.startswith(".")]
    top_dirs.sort(key=lambda p: natural_key(p.name))
    for d in top_dirs:
        chap = build_chapter_for_dir(d, repo_root)
        if chap:
            chapters.append(chap)

    # 2) top-level md files (except the chosen root) — we'll filter the root later.
    top_md = [p for p in repo_root.glob("*.md")]
    top_md.sort(key=lambda p: natural_key(p.name))

    return chapters, top_md


def build_yaml(repo_root: Path, root_file: Path, chapters):
    """Render a minimal YAML string for Jupyter Book TOC."""
    lines = []
    lines.append("format: jb-book")
    root_str = rel_no_ext(root_file, repo_root) if root_file else "index"
    lines.append(f"root: {root_str}")
    lines.append("")
    lines.append("chapters:")

    def emit_chapter(chap, indent="  "):
        lines.append(f"{indent}- file: {chap['file']}")
        if "sections" in chap and chap["sections"]:
            lines.append(f"{indent}  sections:")
            for sec in chap["sections"]:
                lines.append(f"{indent}{indent}  - file: {sec['file']}")

    # If discover_chapters returned a tuple, we are in generic mode with extra top-level mds.
    if isinstance(chapters, tuple):
        chapters, top_md = chapters
        if root_file:
            top_md = [p for p in top_md if p.resolve() != root_file.resolve()]
        for chap in chapters:
            emit_chapter(chap)
        for md in top_md:
            lines.append(f"  - file: {rel_no_ext(md, repo_root)}")
    else:
        for chap in chapters:
            emit_chapter(chap)

    return "\n".join(lines) + "\n"


essential_hint = """
# Hints:
# - Put an index.md at your repo root to be the landing page.
# - In each module folder, create an intro.md (or index.md) to serve as the chapter page.
# - The script does not descend recursively for sections; it only lists files directly inside a chapter directory.
"""


def main():
    repo_root = Path(".").resolve()
    toc_path = repo_root / "_toc.yml"

    root_file = find_root_file(repo_root)
    if not root_file:
        sys.stderr.write("WARNING: No root 'index.md' or 'README.md' found at repo root.\n")
        sys.stderr.write("         TOC will set 'root: index' (ensure index.md exists or update TOC).\n")

    chapters = discover_chapters(repo_root)
    yaml_text = build_yaml(repo_root, root_file, chapters)
    toc_path.write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {toc_path}")


if __name__ == "__main__":
    main()
