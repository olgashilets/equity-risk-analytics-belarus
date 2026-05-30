# -*- coding: utf-8 -*-
"""Portfolio risk and return metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | np.ndarray) -> pd.Series:
    """Calculate portfolio returns as R @ w."""
    if isinstance(weights, pd.Series):
        w = weights.reindex(returns.columns).fillna(0.0).to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    return pd.Series(returns.to_numpy(dtype=float) @ w, index=returns.index, name="portfolio_return")


def historical_var(return_series: pd.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR as the lower percentile of returns."""
    r = np.asarray(return_series, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return np.nan
    return float(np.percentile(r, (1.0 - confidence) * 100.0))


def historical_cvar(return_series: pd.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Historical CVaR / Expected Shortfall as mean return below VaR."""
    r = np.asarray(return_series, dtype=float)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return np.nan
    var = historical_var(r, confidence=confidence)
    tail = r[r <= var]
    if tail.size == 0:
        return var
    return float(tail.mean())


def max_drawdown(return_series: pd.Series | np.ndarray, log_returns: bool = True) -> float:
    """Calculate maximum drawdown from monthly returns."""
    r = pd.Series(return_series).dropna()
    if r.empty:
        return np.nan
    if log_returns:
        wealth = np.exp(r.cumsum())
    else:
        wealth = (1.0 + r).cumprod()
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1.0
    return float(drawdowns.min())


def summarize_portfolio(
    return_series: pd.Series,
    name: str,
    confidence: float = 0.95,
    risk_free_rate_annual: float = 0.0,
    log_returns: bool = True,
) -> dict[str, float | str]:
    """Calculate annualized and tail-risk metrics for one portfolio."""
    r = pd.Series(return_series).dropna()
    mean_annual = float(r.mean() * 12.0)
    vol_annual = float(r.std(ddof=1) * np.sqrt(12.0))
    var = historical_var(r, confidence=confidence)
    cvar = historical_cvar(r, confidence=confidence)
    mdd = max_drawdown(r, log_returns=log_returns)
    sharpe = np.nan if vol_annual == 0 else (mean_annual - risk_free_rate_annual) / vol_annual
    return {
        "portfolio": name,
        "annual_return_pct": mean_annual * 100.0,
        "annual_volatility_pct": vol_annual * 100.0,
        "var_95_monthly_pct": var * 100.0,
        "cvar_95_monthly_pct": cvar * 100.0,
        "max_drawdown_pct": mdd * 100.0,
        "sharpe_annual": sharpe,
    }


def summarize_portfolios(
    portfolio_returns_df: pd.DataFrame,
    confidence: float = 0.95,
    risk_free_rate_annual: float = 0.0,
    log_returns: bool = True,
) -> pd.DataFrame:
    """Summarize all portfolio return columns in one table."""
    rows = [
        summarize_portfolio(
            portfolio_returns_df[col],
            name=col,
            confidence=confidence,
            risk_free_rate_annual=risk_free_rate_annual,
            log_returns=log_returns,
        )
        for col in portfolio_returns_df.columns
    ]
    return pd.DataFrame(rows)
