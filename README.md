# Equity Risk Analytics for an Illiquid Market

[Русская версия](README.ru.md)

## Overview

This repository contains a reproducible market risk analytics pipeline for Belarusian equities. It is organized as three connected case studies:

1. **Asset universe clustering** — reduce a set of equities by grouping assets with similar return behavior.
2. **CVaR portfolio optimization** — build and compare long-only portfolios with a focus on historical tail risk / Expected Shortfall.
3. **Macro factor stress testing** — estimate portfolio sensitivity to inflation, FX, refinancing-rate and industrial-production shocks.

The repository is designed as a portfolio project for Data Analyst, Risk Analyst and Market Risk Analyst roles in finance and fintech.

## Business problem

Illiquid equity markets often contain assets with irregular trading activity, short price histories and unstable relationships. A risk analyst needs to:

- reduce a redundant asset universe before portfolio construction;
- compare benchmark and optimized portfolios using interpretable risk metrics;
- understand that tail-risk robustness and macro-factor robustness are different dimensions of risk.

## Repository structure

```text
equity-risk-analytics-belarus/
├── 01_clustering/
├── 02_cvar_portfolio/
├── 03_factor_stress_testing/
├── docs/
├── requirements.txt
├── .gitignore
├── README.md
└── README.ru.md
```

## Case studies

### 01. Asset clustering

**Goal:** group equities with similar monthly simple-return behavior and select one representative asset from each cluster.

**Methods:** monthly simple returns, correlation distance `1 - |corr|`, hierarchical clustering, representative selection by annualized return within each cluster.

**Output:** cluster labels, selected asset universe, selected price matrix and clustering visualizations.

### 02. CVaR portfolio optimization

**Goal:** construct a long-only defensive portfolio by minimizing historical CVaR / Expected Shortfall at the 95% confidence level.

**Important:** this module no longer repeats the clustering stage. It receives a predefined universe of 7 selected assets as input and focuses only on portfolio optimization and tail-risk analysis.

**Methods:** monthly log returns, equal-weight benchmark, minimum-volatility optimization, CVaR optimization, VaR, CVaR, maximum drawdown and Sharpe comparison.

### 03. Macro factor stress testing

**Goal:** estimate the sensitivity of portfolio returns to macroeconomic factors and run a scenario stress test.

**Methods:** multiple linear regression, VIF diagnostics, robust p-values, factor beta interpretation and stress-scenario decomposition.

## Key results

- The clustering stage reduces the asset universe to 7 representative equities.
- The CVaR portfolio has a lower historical tail loss than the equal-weight and minimum-volatility portfolios in the analyzed sample.
- The factor model shows that a portfolio can be strong by CVaR while still being sensitive to specific macro shocks.

## Tech stack

Python, pandas, NumPy, SciPy, statsmodels, matplotlib, seaborn, pytest.

## How to run

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run each case study separately:

```bash
cd 01_clustering
python run_clustering.py

cd ../02_cvar_portfolio
python run_cvar_portfolio.py

cd ../03_factor_stress_testing
python run_factor_analysis.py
```

Each module writes its results to its own `outputs/` folder.

## Data and limitations

This project is a case study and is not an investment recommendation. See [`docs/limitations.md`](docs/limitations.md) for details.
