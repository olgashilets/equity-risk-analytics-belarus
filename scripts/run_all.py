# -*- coding: utf-8 -*-
"""Run all three risk analytics modules in sequence."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("01_clustering", "run_clustering.py"),
    ("02_cvar_portfolio", "run_cvar_portfolio.py"),
    ("03_factor_stress_testing", "run_factor_analysis.py"),
]


def main() -> None:
    for folder, script in STEPS:
        print(f"\n=== Running {folder}/{script} ===")
        subprocess.run([sys.executable, script], cwd=ROOT / folder, check=True)


if __name__ == "__main__":
    main()
