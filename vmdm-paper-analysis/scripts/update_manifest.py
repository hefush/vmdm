#!/usr/bin/env python3
"""Build FILE_MANIFEST.tsv for source files intended for git tracking."""

from __future__ import annotations

import fnmatch
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "FILE_MANIFEST.tsv"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".home",
    ".cache",
    ".mplconfig",
    ".pip-cache",
    ".conda-env",
    ".conda-pkgs",
    ".mamba-root",
    "__pycache__",
    "figs_png",
}

EXCLUDE_PATTERNS = [
    "*.pyc",
    "figures/**/*.tiff",
    "figures/**/*rebuilt*.pptx",
    "figures/**/Fig2-1.pdf",
    "figures/**/PRESS_Figure2_combined.pdf",
    "figures/**/Fig3.pdf",
    "figures/**/Fig3-1.pdf",
    "figures/**/Fig3.png",
    "figures/**/Fig4.pdf",
    "figures/**/Fig4-1.pdf",
    "figures/**/Fig4.png",
    "figures/**/Fig5-1.pdf",
    "figures/**/Fig5-1.png",
    "figures/**/subFig*.pdf",
    "figures/**/subFig*.png",
    "figures/**/sTable4_subFig4_direct_sputum_predictions.xlsx",
    "supplementary_tables/Supplementary_Table_5_rebuilt.xlsx",
]


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if path == OUT:
        return True
    if any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts):
        return True
    return any(fnmatch.fnmatch(rel, pattern) for pattern in EXCLUDE_PATTERNS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        if is_excluded(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        rows.append((rel, path.stat().st_size, sha256(path)))

    with OUT.open("w", encoding="utf-8", newline="") as handle:
        handle.write("path\tsize_bytes\tsha256\n")
        for rel, size, digest in rows:
            handle.write(f"{rel}\t{size}\t{digest}\n")

    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Files: {len(rows)}")
    print(f"Bytes: {sum(size for _, size, _ in rows)}")
    for rel, size, _digest in sorted(rows, key=lambda row: row[1], reverse=True)[:10]:
        print(f"{size}\t{rel}")


if __name__ == "__main__":
    main()
