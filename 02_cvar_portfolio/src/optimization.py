# -*- coding: utf-8 -*-
"""Long-only portfolio optimization routines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


def _as_series(weights: np.ndarray, columns: list[str] | pd.Index, name: str) -> pd.Series:
    weights = np.asarray(weights, dtype=float)
    weights[np.abs(weights) < 1e-10] = 0.0
    total = weights.sum()
    if total <= 0:
        raise ValueError("Portfolio weights sum to a non-positive value.")
    weights = weights / total
    return pd.Series(weights, index=list(columns), name=name)


def equal_weight(returns: pd.DataFrame, name: str = "Equal Weight") -> pd.Series:
    """Equal-weight long-only benchmark."""
    n = returns.shape[1]
    if n == 0:
        raise ValueError("No assets available for portfolio construction.")
    return pd.Series(np.ones(n) / n, index=returns.columns, name=name)


def min_volatility_weights(returns: pd.DataFrame) -> pd.Series:
    """Minimum-volatility long-only portfolio with sum(w)=1."""
    returns = returns.dropna(how="all").fillna(0.0)
    n = returns.shape[1]
    cov = np.nan_to_num(returns.cov().to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n)]
    result = minimize(
        objective,
        np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-volatility optimization failed: {result.message}")
    return _as_series(result.x, returns.columns, "Minimum Volatility")


def min_cvar_weights(returns: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """Minimum historical CVaR / Expected Shortfall portfolio.

    The optimization follows the Rockafellar-Uryasev linear-programming form.
    Returns are monthly log returns. Loss is defined as `-portfolio_return`.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    returns = returns.dropna(how="all").fillna(0.0)
    R = returns.to_numpy(dtype=float)
    T, n = R.shape
    if T == 0 or n == 0:
        raise ValueError("Not enough observations for CVaR optimization.")

    # Variables: w_0..w_{n-1}, eta, u_0..u_{T-1}
    c = np.r_[np.zeros(n), 1.0, np.ones(T) / (alpha * T)]

    A_eq = np.zeros((1, n + 1 + T))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    A_ub = np.zeros((T, n + 1 + T))
    b_ub = np.zeros(T)
    for t in range(T):
        A_ub[t, :n] = -R[t]
        A_ub[t, n] = -1.0
        A_ub[t, n + 1 + t] = -1.0

    bounds = [(0.0, 1.0) for _ in range(n)] + [(None, None)] + [(0.0, None) for _ in range(T)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"CVaR optimization failed: {result.message}")
    return _as_series(result.x[:n], returns.columns, "CVaR 95%")


def combine_weights(*weight_series: pd.Series) -> pd.DataFrame:
    """Combine several weight vectors into one table."""
    return pd.concat(weight_series, axis=1).fillna(0.0)
