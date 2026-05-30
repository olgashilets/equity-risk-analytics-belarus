# -*- coding: utf-8 -*-
"""Input data loading for the CVaR portfolio module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_selected_prices(path: str | Path) -> pd.DataFrame:
    """Load the predefined selected-asset price matrix.

    This function expects a wide CSV/XLSX table with a date column and one column
    per asset. The asset universe is treated as a fixed input; no clustering is
    performed in this module.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    date_col = "date" if "date" in df.columns else ("Дата" if "Дата" in df.columns else None)
    if date_col is None:
        raise ValueError("Price table must contain a 'date' or 'Дата' column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).sort_values("date").set_index("date")
    df = df.replace(0, np.nan).ffill().interpolate(method="linear", limit_direction="both").bfill()
    df = df.dropna(axis=1, how="any")
    if df.shape[1] < 2:
        raise ValueError("At least two assets are required for portfolio optimization.")
    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly log returns from a price matrix."""
    returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return returns.fillna(0.0)
