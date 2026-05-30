# Methodology

## 1. Asset clustering

The clustering module calculates monthly **simple returns** using `pct_change()`. This is intentional: the published cluster structure and selected asset universe were produced under this setup. Changing the return definition to logarithmic returns changes the clustering output and would require recalculating the downstream portfolio and stress-test modules.

Distance between assets is based on the absolute correlation of returns:

```text
distance = 1 - |correlation|
```

The absolute value focuses on strength of co-movement. Hierarchical clustering is then used to form the asset groups.

## 2. CVaR portfolio optimization

The CVaR module treats the selected asset universe as a fixed input. It does not repeat clustering. Prices are transformed into monthly **log returns** before portfolio construction.

Three portfolios are compared:

- equal-weight benchmark;
- minimum-volatility long-only portfolio;
- CVaR 95% / Expected Shortfall long-only portfolio.

The CVaR objective minimizes expected loss in the adverse 5% tail of the empirical monthly return distribution.

## 3. Macro factor stress testing

The factor model estimates portfolio return sensitivity to four macro factors:

- monthly inflation;
- monthly BYN depreciation against USD;
- refinancing-rate change;
- change in industrial-production growth.

The model is estimated separately for each portfolio. Robust p-values are used because the sample is short. The script attempts HC3 first and falls back to HC1 when HC3 becomes non-finite because of high-leverage observations. VIF is used as a multicollinearity diagnostic.
