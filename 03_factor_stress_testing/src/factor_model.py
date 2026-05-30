# -*- coding: utf-8 -*-
"""Macro factor sensitivity model for portfolio returns."""
from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=RuntimeWarning)

FACTOR_COLS = [
    "inflation_mom_pct",
    "usd_byn_depreciation_pct",
    "refinancing_rate_change_pp",
    "industrial_prod_yoy_change_pp",
]

FACTOR_LABELS = {
    "inflation_mom_pct": "Inflation MoM, %",
    "usd_byn_depreciation_pct": "BYN depreciation vs USD, MoM, %",
    "refinancing_rate_change_pp": "Refinancing rate change, p.p.",
    "industrial_prod_yoy_change_pp": "Industrial growth change, p.p.",
}

STRESS_SHOCKS = {
    "inflation_mom_pct": 1.0,
    "usd_byn_depreciation_pct": 5.0,
    "refinancing_rate_change_pp": 1.0,
    "industrial_prod_yoy_change_pp": -2.0,
}


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def load_portfolio_returns(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if "date" not in df.columns:
        raise ValueError("portfolio_returns.csv must contain a date column.")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_macro_factors(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if "date" not in df.columns:
        raise ValueError("macro_factors.csv must contain a date column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "usd_byn_depreciation_pct" not in df.columns and "usd_byn_avg_geometric" in df.columns:
        df["usd_byn_depreciation_pct"] = df["usd_byn_avg_geometric"].astype(float).pct_change() * 100
    if "refinancing_rate_change_pp" not in df.columns and "refinancing_rate_pct" in df.columns:
        df["refinancing_rate_change_pp"] = df["refinancing_rate_pct"].astype(float).diff()
    if "industrial_prod_yoy_pct" not in df.columns and "industrial_prod_yoy_index_pct" in df.columns:
        df["industrial_prod_yoy_pct"] = df["industrial_prod_yoy_index_pct"].astype(float) - 100
    if "industrial_prod_yoy_change_pp" not in df.columns and "industrial_prod_yoy_pct" in df.columns:
        df["industrial_prod_yoy_change_pp"] = df["industrial_prod_yoy_pct"].astype(float).diff()

    missing = [col for col in FACTOR_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"macro_factors.csv is missing required factors: {missing}")
    return df


def build_model_dataset(portfolio_returns: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    model_data = portfolio_returns.merge(macro, on="date", how="inner")
    portfolio_cols = [c for c in portfolio_returns.columns if c != "date"]
    model_data = model_data.dropna(subset=FACTOR_COLS + portfolio_cols).sort_values("date").reset_index(drop=True)
    if len(model_data) < 10:
        raise ValueError("Too few observations after merging portfolio and macro data.")
    return model_data, portfolio_cols


def calculate_vif(x: pd.DataFrame) -> pd.DataFrame:
    x_clean = x.astype(float).dropna()
    values = x_clean.values
    rows = []
    for i, col in enumerate(x_clean.columns):
        try:
            vif = float(variance_inflation_factor(values, i))
        except Exception:
            vif = np.nan
        rows.append({"factor": col, "factor_label": FACTOR_LABELS.get(col, col), "vif": vif})
    return pd.DataFrame(rows)


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def robust_results_with_fallback(model):
    """Return robust covariance results.

    HC3 can become non-finite when a design matrix contains extreme leverage
    points. In this dataset the refinancing-rate change is very sparse, so HC3
    may produce infinite standard errors. A professional workflow should not
    silently report such p-values; therefore we use HC3 when finite and fall back
    to HC1 otherwise, while recording the covariance type in outputs.
    """
    for cov_type in ["HC3", "HC1"]:
        robust = model.get_robustcov_results(cov_type=cov_type)
        bse = np.asarray(robust.bse, dtype=float)
        pvalues = np.asarray(robust.pvalues, dtype=float)
        if np.isfinite(bse).all() and np.isfinite(pvalues).all():
            return robust, cov_type
    return model, "OLS"


def estimate_factor_models(model_data: pd.DataFrame, portfolio_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    results_rows = []
    fitted_frames = []
    equations = []

    X = sm.add_constant(model_data[FACTOR_COLS].astype(float), has_constant="add")

    for portfolio in portfolio_cols:
        y = model_data[portfolio].astype(float) * 100.0
        model = sm.OLS(y, X, missing="drop").fit()
        robust, cov_type = robust_results_with_fallback(model)
        params = pd.Series(robust.params, index=X.columns)
        pvals = pd.Series(robust.pvalues, index=X.columns)
        tvals = pd.Series(robust.tvalues, index=X.columns)
        conf = pd.DataFrame(robust.conf_int(), index=X.columns, columns=["ci_low_robust", "ci_high_robust"])

        for term in X.columns:
            results_rows.append({
                "portfolio": portfolio,
                "term": term,
                "term_label": "Intercept" if term == "const" else FACTOR_LABELS.get(term, term),
                "coef": params[term],
                "p_value_robust": pvals[term],
                "t_stat_robust": tvals[term],
                "ci_low_robust": conf.loc[term, "ci_low_robust"],
                "ci_high_robust": conf.loc[term, "ci_high_robust"],
                "robust_cov_type": cov_type,
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "nobs": int(model.nobs),
                "significance": significance_stars(pvals[term]),
            })

        y_hat = model.predict(X)
        fitted_frames.append(pd.DataFrame({
            "date": model_data["date"],
            "portfolio": portfolio,
            "actual_return_pct": y,
            "fitted_return_pct": y_hat,
            "residual_pct": y - y_hat,
        }))

        rhs = f"{params['const']:.4f}{significance_stars(pvals['const'])}"
        for term in FACTOR_COLS:
            sign = "+" if params[term] >= 0 else "-"
            rhs += f" {sign} {abs(params[term]):.4f}{significance_stars(pvals[term])}·{term}"
        equations.append(f"{portfolio}: R_t(%) = {rhs} + ε_t | R²={model.rsquared:.3f}, adj.R²={model.rsquared_adj:.3f}, n={int(model.nobs)}, cov={cov_type}")
        equations.append("    Robust p-values: " + ", ".join([f"{term}={pvals[term]:.3f}" for term in X.columns]))

    return pd.DataFrame(results_rows), pd.concat(fitted_frames, ignore_index=True), equations


def run_stress_test(reg_results: pd.DataFrame) -> pd.DataFrame:
    betas = reg_results[reg_results["term"].isin(FACTOR_COLS)].pivot(index="portfolio", columns="term", values="coef")
    rows = []
    for portfolio in betas.index:
        contributions = {factor: float(betas.loc[portfolio, factor]) * shock for factor, shock in STRESS_SHOCKS.items()}
        total = sum(contributions.values())
        abs_sum = sum(abs(value) for value in contributions.values())
        for factor, contribution in contributions.items():
            rows.append({
                "portfolio": portfolio,
                "factor": factor,
                "factor_label": FACTOR_LABELS[factor],
                "shock": STRESS_SHOCKS[factor],
                "beta": float(betas.loc[portfolio, factor]),
                "contribution_return_pp": contribution,
                "total_delta_return_pp": total,
                "sum_abs_contributions_pp": abs_sum,
            })
        rows.append({
            "portfolio": portfolio,
            "factor": "TOTAL",
            "factor_label": "Total stress scenario",
            "shock": np.nan,
            "beta": np.nan,
            "contribution_return_pp": total,
            "total_delta_return_pp": total,
            "sum_abs_contributions_pp": abs_sum,
        })
    return pd.DataFrame(rows)


def make_plots(reg_results: pd.DataFrame, fitted_values: pd.DataFrame, stress: pd.DataFrame, figures_dir: Path) -> None:
    """Create factor-analysis figures using matplotlib only."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    beta = reg_results[reg_results["term"].isin(FACTOR_COLS)].pivot(index="portfolio", columns="term", values="coef")[FACTOR_COLS]
    fig, ax = plt.subplots(figsize=(13, 5.7))
    matrix = beta.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(FACTOR_COLS)))
    ax.set_xticklabels([FACTOR_LABELS[c] for c in FACTOR_COLS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(beta.index)))
    ax.set_yticklabels(beta.index.tolist())
    ax.set_xlabel("Macro factor")
    ax.set_ylabel("Portfolio")
    ax.set_title("Portfolio beta coefficients to macro factors")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="β: return p.p. per 1 unit of factor")
    fig.tight_layout()
    fig.savefig(figures_dir / "betas_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    portfolios = fitted_values["portfolio"].unique().tolist()
    fig_height = max(4, 3.6 * len(portfolios))
    fig, axes = plt.subplots(len(portfolios), 1, figsize=(12, fig_height), sharex=True)
    if len(portfolios) == 1:
        axes = [axes]
    for ax, portfolio in zip(axes, portfolios):
        sub = fitted_values[fitted_values["portfolio"] == portfolio]
        r2 = reg_results[reg_results["portfolio"] == portfolio]["r_squared"].iloc[0]
        ax.plot(sub["date"], sub["actual_return_pct"], marker="o", label="Actual")
        ax.plot(sub["date"], sub["fitted_return_pct"], marker="s", linestyle="--", label="Fitted")
        ax.axhline(0, linewidth=0.8)
        ax.set_title(f"{portfolio}: actual vs fitted return, R²={r2:.3f}")
        ax.set_ylabel("Monthly return, %")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(figures_dir / "fitted_vs_actual.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    contrib = stress[stress["factor"].isin(FACTOR_COLS)].copy()
    portfolios = contrib["portfolio"].drop_duplicates().tolist()
    x = np.arange(len(portfolios))
    width = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, factor in enumerate(FACTOR_COLS):
        values = []
        for portfolio in portfolios:
            value = contrib[(contrib["portfolio"] == portfolio) & (contrib["factor"] == factor)]["contribution_return_pp"].iloc[0]
            values.append(value)
        ax.bar(x + (idx - 1.5) * width, values, width, label=FACTOR_LABELS[factor])
    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(portfolios, rotation=0)
    ax.set_xlabel("Portfolio")
    ax.set_ylabel("Return change, p.p.")
    ax.set_title("Macro factor contributions in stress scenario")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Factor", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(figures_dir / "sensitivity_barchart.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def make_interpretation(stress: pd.DataFrame, vif: pd.DataFrame) -> str:
    total = stress[stress["factor"] == "TOTAL"].copy()
    total["abs_total_delta_pp"] = total["total_delta_return_pp"].abs()
    least_abs_total = total.sort_values("abs_total_delta_pp").iloc[0]
    least_abs_contrib = total.sort_values("sum_abs_contributions_pp").iloc[0]

    lines = [
        "Macro factor stress-testing interpretation",
        "=" * 52,
        "Model: R_portfolio,t (%) = α + β1·inflation + β2·FX_depreciation + β3·rate_change + β4·industrial_growth_change + ε.",
        "Portfolio returns are converted to percentages, so beta shows the change in monthly return in percentage points per one unit of the factor.",
        "",
        "VIF diagnostics:",
    ]
    for _, row in vif.iterrows():
        flag = "OK" if row["vif"] < 5 else ("check" if row["vif"] < 10 else "high multicollinearity")
        lines.append(f"- {row['factor_label']}: VIF={row['vif']:.2f} ({flag})")
    lines.append("")
    lines.append("Stress scenario (+1 p.p. inflation, +5% USD/BYN, +1 p.p. refinancing rate, -2 p.p. industrial growth):")
    for _, row in total.iterrows():
        lines.append(f"- {row['portfolio']}: total delta={row['total_delta_return_pp']:.2f} p.p.; sum of absolute factor contributions={row['sum_abs_contributions_pp']:.2f} p.p.")
    lines.append("")
    lines.append(f"Lowest absolute total delta: {least_abs_total['portfolio']} ({least_abs_total['total_delta_return_pp']:.2f} p.p.).")
    lines.append(f"Lowest sum of absolute factor contributions: {least_abs_contrib['portfolio']} ({least_abs_contrib['sum_abs_contributions_pp']:.2f} p.p.).")
    lines.append("")
    lines.append("This complements the CVaR case: CVaR measures the historical adverse tail, while the factor model measures sensitivity to specific macro shocks. The sample is short, so the estimates should be interpreted as indicative rather than causal.")
    return "\n".join(lines)
