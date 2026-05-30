from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_loading import compute_log_returns, load_selected_prices
from optimization import combine_weights, equal_weight, min_cvar_weights, min_volatility_weights


def test_portfolio_weights_are_long_only_and_sum_to_one():
    prices = load_selected_prices(PROJECT_DIR / "data" / "input" / "selected_asset_prices.csv")
    returns = compute_log_returns(prices)
    weights = combine_weights(equal_weight(returns), min_volatility_weights(returns), min_cvar_weights(returns))
    assert np.allclose(weights.sum(axis=0).values, 1.0, atol=1e-6)
    assert (weights >= -1e-8).all().all()
