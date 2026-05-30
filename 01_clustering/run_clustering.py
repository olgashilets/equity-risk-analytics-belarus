# -*- coding: utf-8 -*-
"""Run the asset clustering case study."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
DATA_PATH = PROJECT_DIR / "data" / "input" / "data.xlsx"
OUTPUT_DIR = PROJECT_DIR / "outputs"

sys.path.insert(0, str(SRC_DIR))

from clustering import build_interpretation, load_prices, run_clustering, save_outputs  # noqa: E402


def main() -> None:
    prices = load_prices(DATA_PATH)
    result = run_clustering(prices)
    save_outputs(result, OUTPUT_DIR)
    interpretation = build_interpretation(result)
    (OUTPUT_DIR / "interpretation.txt").write_text(interpretation, encoding="utf-8")
    print(interpretation)


if __name__ == "__main__":
    main()
