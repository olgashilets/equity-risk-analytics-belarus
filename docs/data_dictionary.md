# Data dictionary

## Asset data

| Field | Meaning |
|---|---|
| `date` | Month-end observation date |
| asset columns | Monthly price levels for selected equities |
| `simple_return` | Percentage change between adjacent monthly prices |
| `log_return` | Natural log of adjacent price ratio |

## Portfolio outputs

| File | Meaning |
|---|---|
| `portfolio_weights_all.csv` | Weights of equal-weight, minimum-volatility and CVaR portfolios |
| `portfolio_returns.csv` | Monthly returns of the three portfolios |
| `portfolio_comparison.csv` | Annual return, annual volatility, VaR, CVaR, max drawdown and Sharpe |

## Macro factor data

| Field | Meaning |
|---|---|
| `inflation_mom_pct` | Monthly CPI change, percent |
| `usd_byn_avg_geometric` | Average official USD/BYN exchange rate |
| `usd_byn_depreciation_pct` | Monthly percent change in USD/BYN; positive value means BYN depreciation |
| `refinancing_rate_pct` | Refinancing rate, percent |
| `refinancing_rate_change_pp` | Monthly change in refinancing rate, percentage points |
| `industrial_prod_yoy_index_pct` | Industrial production index, percent to the corresponding month of the previous year |
| `industrial_prod_yoy_change_pp` | Monthly change in industrial-production growth, percentage points |
