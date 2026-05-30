# -*- coding: utf-8 -*-
"""Run the CVaR portfolio optimization module.

The input universe is predefined in data/input/selected_asset_prices.csv.
No clustering is performed in this module.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
DATA_PATH = PROJECT_DIR / "data" / "input" / "selected_asset_prices.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

sys.path.insert(0, str(SRC_DIR))

from data_loading import compute_log_returns, load_selected_prices  # noqa: E402
from optimization import combine_weights, equal_weight, min_cvar_weights, min_volatility_weights  # noqa: E402
from portfolio_metrics import portfolio_returns, summarize_portfolios  # noqa: E402
from visualization import plot_cumulative_returns, plot_cvar_histogram, plot_weights_comparison  # noqa: E402


def build_interpretation(metrics: pd.DataFrame, weights_all: pd.DataFrame) -> str:
    m = metrics.set_index("portfolio")
    cvar_name = "CVaR 95%"
    minvol_name = "Minimum Volatility"
    eq_name = "Equal Weight"

    cvar_tail = m.loc[cvar_name, "cvar_95_monthly_pct"]
    minvol_tail = m.loc[minvol_name, "cvar_95_monthly_pct"]
    eq_tail = m.loc[eq_name, "cvar_95_monthly_pct"]
    cvar_return = m.loc[cvar_name, "annual_return_pct"]
    minvol_return = m.loc[minvol_name, "annual_return_pct"]
    cvar_vol = m.loc[cvar_name, "annual_volatility_pct"]
    minvol_vol = m.loc[minvol_name, "annual_volatility_pct"]

    top = weights_all[cvar_name].sort_values(ascending=False)
    top_text = ", ".join([f"{asset} ({weight:.1%})" for asset, weight in top[top > 0.01].head(5).items()])

    return f"""CVaR portfolio interpretation
=============================

This module focuses only on portfolio optimization. The 7-asset universe is a predefined input from the clustering module; no clustering is repeated here.

The CVaR 95% portfolio minimizes the expected loss in the adverse 5% tail of the empirical monthly return distribution. In the current sample, its monthly CVaR is {cvar_tail:.2f}% versus {minvol_tail:.2f}% for the minimum-volatility portfolio and {eq_tail:.2f}% for the equal-weight benchmark.

The largest positive weights in the CVaR portfolio are: {top_text}.

The CVaR portfolio has annual return {cvar_return:.2f}% and annual volatility {cvar_vol:.2f}%. The minimum-volatility portfolio has annual return {minvol_return:.2f}% and annual volatility {minvol_vol:.2f}%. This difference shows that minimizing volatility and minimizing tail loss are related but not identical risk objectives.
"""


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_selected_prices(DATA_PATH)
    returns = compute_log_returns(prices)

    prices.to_csv(TABLES_DIR / "selected_asset_prices_clean.csv", encoding="utf-8-sig")
    returns.to_csv(TABLES_DIR / "log_returns_selected_assets.csv", encoding="utf-8-sig")

    w_equal = equal_weight(returns)
    w_min_vol = min_volatility_weights(returns)
    w_cvar = min_cvar_weights(returns, alpha=0.05)
    weights_all = combine_weights(w_equal, w_min_vol, w_cvar)

    if not np.allclose(weights_all.sum(axis=0).values, 1.0, atol=1e-6):
        raise AssertionError("At least one portfolio weight vector does not sum to 1.")
    if (weights_all < -1e-8).any().any():
        raise AssertionError("Negative weights found in a long-only portfolio.")

    weights_all.to_csv(TABLES_DIR / "portfolio_weights_all.csv", encoding="utf-8-sig")
    w_cvar.rename("weight").reset_index().rename(columns={"index": "asset"}).to_csv(
        TABLES_DIR / "optimal_weights_cvar.csv", index=False, encoding="utf-8-sig"
    )
    w_min_vol.rename("weight").reset_index().rename(columns={"index": "asset"}).to_csv(
        TABLES_DIR / "optimal_weights_min_vol.csv", index=False, encoding="utf-8-sig"
    )

    port_ret = pd.DataFrame({
        w_equal.name: portfolio_returns(returns, w_equal),
        w_min_vol.name: portfolio_returns(returns, w_min_vol),
        w_cvar.name: portfolio_returns(returns, w_cvar),
    })
    port_ret.to_csv(TABLES_DIR / "portfolio_returns.csv", encoding="utf-8-sig")

    metrics = summarize_portfolios(port_ret, confidence=0.95, risk_free_rate_annual=0.0, log_returns=True)
    metrics_rounded = metrics.copy()
    for col in metrics_rounded.columns:
        if col != "portfolio":
            metrics_rounded[col] = metrics_rounded[col].astype(float).round(4)
    metrics_rounded.to_csv(TABLES_DIR / "portfolio_comparison.csv", index=False, encoding="utf-8-sig")

    plot_cumulative_returns(port_ret, FIGURES_DIR / "cumulative_returns.png", log_returns=True)
    plot_cvar_histogram(port_ret, FIGURES_DIR / "cvar_histogram.png", confidence=0.95)
    plot_weights_comparison(weights_all[["Minimum Volatility", "CVaR 95%"]], FIGURES_DIR / "weights_comparison.png")

    interpretation = build_interpretation(metrics, weights_all)
    (OUTPUT_DIR / "interpretation.txt").write_text(interpretation, encoding="utf-8")

    print("Done. Results saved to outputs/.")
    print(metrics_rounded.to_string(index=False))


if __name__ == "__main__":
    main()
