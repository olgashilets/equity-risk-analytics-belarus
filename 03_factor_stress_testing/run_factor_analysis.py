# -*- coding: utf-8 -*-
"""Run macro factor sensitivity and stress testing."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data" / "input"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

sys.path.insert(0, str(SRC_DIR))

from factor_model import (  # noqa: E402
    build_model_dataset,
    calculate_vif,
    estimate_factor_models,
    load_macro_factors,
    load_portfolio_returns,
    make_interpretation,
    make_plots,
    run_stress_test,
)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    portfolio_returns = load_portfolio_returns(DATA_DIR / "portfolio_returns.csv")
    macro = load_macro_factors(DATA_DIR / "macro_factors.csv")
    model_data, portfolio_cols = build_model_dataset(portfolio_returns, macro)

    model_data.to_csv(TABLES_DIR / "model_dataset.csv", index=False, encoding="utf-8-sig")
    vif = calculate_vif(model_data[[
        "inflation_mom_pct",
        "usd_byn_depreciation_pct",
        "refinancing_rate_change_pp",
        "industrial_prod_yoy_change_pp",
    ]])
    vif.to_csv(TABLES_DIR / "vif_results.csv", index=False, encoding="utf-8-sig")

    regression_results, fitted_values, equations = estimate_factor_models(model_data, portfolio_cols)
    regression_results.to_csv(TABLES_DIR / "regression_results.csv", index=False, encoding="utf-8-sig")
    fitted_values.to_csv(TABLES_DIR / "fitted_values.csv", index=False, encoding="utf-8-sig")
    (TABLES_DIR / "factor_equations.txt").write_text("\n".join(equations), encoding="utf-8")

    stress = run_stress_test(regression_results)
    stress.to_csv(TABLES_DIR / "stress_test_results.csv", index=False, encoding="utf-8-sig")
    make_plots(regression_results, fitted_values, stress, FIGURES_DIR)

    interpretation = make_interpretation(stress, vif)
    (OUTPUT_DIR / "interpretation.txt").write_text(interpretation, encoding="utf-8")

    print("Done. Results saved to outputs/.")
    print("\n".join(equations))
    print("\n" + interpretation)


if __name__ == "__main__":
    main()
