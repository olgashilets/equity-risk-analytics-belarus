# 01. Asset Universe Clustering

[Русская версия](README.ru.md)

## Purpose

This case study reduces an equity universe by grouping assets with similar monthly return behavior. It is intended as the first step of the risk analytics pipeline: it selects a compact universe for later portfolio optimization.

## Methodology

- Missing and zero prices are treated as missing values.
- Assets with fewer than 10 monthly observations are excluded.
- Returns are calculated as **monthly simple returns** using `pct_change()`.
- The distance between assets is calculated as `1 - |correlation|`.
- Hierarchical clustering is applied with Ward linkage.
- One representative asset is selected from each cluster using the highest annualized simple return within the cluster.

Using simple returns here is intentional. It reproduces the selected universe used in the downstream portfolio project. Changing this stage to log returns changes the clustering output and would require recalculating the CVaR and factor-analysis modules.

## How to run

From this folder:

```bash
python run_clustering.py
```

## Outputs

```text
outputs/
├── tables/
│   ├── clean_prices.csv
│   ├── simple_returns.csv
│   ├── correlation_matrix.csv
│   ├── cluster_results.csv
│   ├── selected_assets.csv
│   ├── selected_asset_prices.csv
│   └── selected_asset_prices_long.csv
├── figures/
│   └── clustering_summary.png
└── interpretation.txt
```

The file `outputs/tables/selected_asset_prices.csv` is the intended input for the next module, `02_cvar_portfolio`.

## Portfolio value

This project demonstrates exploratory financial analysis, return preprocessing, correlation analysis, hierarchical clustering and business interpretation of a reduced asset universe.
