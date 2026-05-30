# 02. CVaR Portfolio Optimization

[Русская версия](README.ru.md)

## Purpose

This case study builds a defensive long-only portfolio using historical CVaR / Expected Shortfall at the 95% confidence level.

The project is intentionally focused on portfolio optimization. The asset universe is **predefined** and loaded from `data/input/selected_asset_prices.csv`. Clustering is not repeated inside this module.

## Business question

Can a CVaR-optimized portfolio reduce expected losses in the worst historical monthly scenarios compared with an equal-weight benchmark and a minimum-volatility portfolio?

## Inputs

```text
data/input/selected_assets.csv
data/input/selected_asset_prices.csv
```

The selected universe comes from the previous clustering case. This separation makes the module cleaner and easier to review: the input universe is fixed, and the module only performs portfolio analytics.

## Methodology

- Clean selected monthly prices.
- Calculate monthly log returns.
- Build three portfolios:
  - Equal Weight;
  - Minimum Volatility;
  - CVaR 95% / Expected Shortfall.
- Compare annualized return, annualized volatility, VaR 95%, CVaR 95%, maximum drawdown and Sharpe ratio.

## How to run

From this folder:

```bash
python run_cvar_portfolio.py
```

## Outputs

```text
outputs/
├── tables/
│   ├── log_returns_selected_assets.csv
│   ├── portfolio_weights_all.csv
│   ├── optimal_weights_cvar.csv
│   ├── optimal_weights_min_vol.csv
│   ├── portfolio_returns.csv
│   └── portfolio_comparison.csv
├── figures/
│   ├── cumulative_returns.png
│   ├── cvar_histogram.png
│   └── weights_comparison.png
└── interpretation.txt
```

## Limitations

The sample is short and the market is illiquid. Historical CVaR is sensitive to the observed scenarios and should be validated out of sample before practical use. This project is an analytical case study and not investment advice.
