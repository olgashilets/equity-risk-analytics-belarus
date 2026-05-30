# -*- coding: utf-8 -*-
"""Visualization helpers for the CVaR module."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from portfolio_metrics import historical_cvar, historical_var


def _save(fig: plt.Figure, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_returns(portfolio_returns: pd.DataFrame, output_path: str | Path, log_returns: bool = True) -> None:
    """Plot cumulative wealth for portfolio return columns."""
    wealth = np.exp(portfolio_returns.cumsum()) if log_returns else (1.0 + portfolio_returns).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    wealth.plot(ax=ax, linewidth=2)
    ax.set_title("Cumulative portfolio returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value index")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Portfolio")
    _save(fig, output_path)


def plot_cvar_histogram(portfolio_returns: pd.DataFrame, output_path: str | Path, confidence: float = 0.95) -> None:
    """Plot monthly return distributions with VaR and CVaR markers."""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    cvar_portfolio_name = "CVaR 95%"

    for i, col in enumerate(portfolio_returns.columns):
        color = colors[i % len(colors)] if colors else None
        r = portfolio_returns[col].dropna()
        var = historical_var(r, confidence=confidence)
        cvar = historical_cvar(r, confidence=confidence)
        ax.hist(r * 100.0, bins=12, alpha=0.30, label=col, color=color)
        ax.axvline(var * 100.0, linestyle="--", linewidth=1.5, color=color, label=f"VaR {col}")
        if col == cvar_portfolio_name:
            ax.axvspan(float(r.min() * 100.0 - 0.5), float(var * 100.0), alpha=0.12, color=color, label="CVaR tail")
            ax.axvline(cvar * 100.0, linestyle=":", linewidth=2.0, color=color, label="CVaR 95%")

    ax.set_title("Monthly return distributions, VaR and CVaR")
    ax.set_xlabel("Monthly return, %")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, output_path)


def plot_weights_comparison(weights: pd.DataFrame, output_path: str | Path) -> None:
    """Compare portfolio weights."""
    fig, ax = plt.subplots(figsize=(12, 6))
    weights.plot(kind="bar", ax=ax)
    ax.set_title("Portfolio weights comparison")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Weight")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Portfolio")
    _save(fig, output_path)
