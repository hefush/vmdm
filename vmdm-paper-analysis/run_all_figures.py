#!/usr/bin/env python3
"""Run figure-generation scripts for the VMDM paper-analysis repository."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TASKS = {
    "figure2": (
        ROOT / "figures" / "figure2",
        [sys.executable, "plot_figure2.py"],
    ),
    "figure3": (
        ROOT / "figures" / "figure3",
        [sys.executable, "plot_figure3.py"],
    ),
    "figure4": (
        ROOT / "figures" / "figure4",
        [sys.executable, "plot_figure4.py"],
    ),
    "figure5_and_supplementary_figure4": (
        ROOT / "figures" / "figure5_and_supplementary_figure4",
        [sys.executable, "plot_figure5_and_suppfig4.py"],
    ),
    "supplementary_figure1": (
        ROOT / "figures" / "supplementary_figure1",
        [sys.executable, "plot_supplementary_figure1.py"],
    ),
    "supplementary_figure2": (
        ROOT / "figures" / "supplementary_figure2",
        [sys.executable, "plot_supplementary_figure2.py"],
    ),
    "supplementary_figure3": (
        ROOT / "figures" / "supplementary_figure3",
        [sys.executable, "plot_supplementary_figure3.py"],
    ),
    "supplementary_figure5": (
        ROOT / "figures" / "supplementary_figure5",
        [sys.executable, "plot_supplementary_figure5.py"],
    ),
}


def env() -> dict[str, str]:
    run_env = os.environ.copy()
    run_env.setdefault("MPLBACKEND", "Agg")
    run_env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    run_env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    run_env.setdefault("ARIAL_FONT_PATH", str(ROOT / "fonts" / "Arial.ttf"))
    return run_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available tasks and exit.")
    parser.add_argument("--only", nargs="+", choices=sorted(TASKS), help="Run only selected tasks.")
    parser.add_argument("--skip", nargs="+", choices=sorted(TASKS), default=[], help="Skip selected tasks.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a task fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for name, (cwd, command) in TASKS.items():
            print(f"{name}\t{cwd.relative_to(ROOT)}\t{' '.join(command)}")
        return

    selected = args.only if args.only else list(TASKS)
    selected = [task for task in selected if task not in set(args.skip)]
    failed = []
    for task in selected:
        cwd, command = TASKS[task]
        (cwd / "figs_png").mkdir(exist_ok=True)
        print(f"\n== {task} ==")
        print("cwd:", cwd.relative_to(ROOT))
        print("cmd:", " ".join(command))
        result = subprocess.run(command, cwd=cwd, env=env())
        if result.returncode != 0:
            failed.append(task)
            if not args.keep_going:
                raise SystemExit(result.returncode)

    if failed:
        print("\nFailed tasks:", ", ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
