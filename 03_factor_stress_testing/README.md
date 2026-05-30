# 03. Macro Factor Stress Testing

[Русская версия](README.ru.md)

## Purpose

This case study estimates how three portfolio strategies react to macroeconomic shocks. It complements the CVaR project: CVaR measures the adverse historical tail, while factor stress testing measures sensitivity to specific macro factors.

## Inputs

```text
data/input/portfolio_returns.csv
data/input/macro_factors.csv
data/input/macro_factors_source_levels.csv
```

`portfolio_returns.csv` comes from the CVaR module. Macro factors are based on official/public Belarusian macroeconomic time series and are transformed into monthly model variables.

## Macro factors

- `inflation_mom_pct` — monthly CPI change, %.
- `usd_byn_depreciation_pct` — monthly BYN depreciation against USD, %.
- `refinancing_rate_change_pp` — monthly refinancing-rate change, percentage points.
- `industrial_prod_yoy_change_pp` — monthly change in industrial-production growth, percentage points.

## Methodology

For each portfolio, the following model is estimated:

```text
R_portfolio,t (%) = α + β1·inflation + β2·FX_depreciation + β3·rate_change + β4·industrial_growth_change + ε
```

Diagnostics and outputs:

- VIF for multicollinearity;
- OLS factor betas;
- robust p-values with covariance-type note;
- actual vs fitted returns;
- factor contribution in a stress scenario.

Stress scenario:

```text
+1 p.p. inflation
+5% BYN depreciation vs USD
+1 p.p. refinancing rate
-2 p.p. industrial-production growth
```

## How to run

From this folder:

```bash
python run_factor_analysis.py
```

## Outputs

```text
outputs/
├── tables/
│   ├── model_dataset.csv
│   ├── vif_results.csv
│   ├── regression_results.csv
│   ├── fitted_values.csv
│   ├── factor_equations.txt
│   └── stress_test_results.csv
├── figures/
│   ├── betas_heatmap.png
│   ├── fitted_vs_actual.png
│   └── sensitivity_barchart.png
└── interpretation.txt
```

## Limitations

The factor estimates are indicative. The script attempts HC3 robust covariance, but falls back to HC1 if HC3 becomes non-finite due to high-leverage observations. The sample has only 24 monthly portfolio-return observations, the market is illiquid, refinancing-rate changes are rare, and most coefficients are not statistically significant. This is a sensitivity-analysis and stress-testing framework, not a forecasting model or investment recommendation.
