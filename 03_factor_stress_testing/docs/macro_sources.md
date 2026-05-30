# Macro data sources and transformations

The macro factor dataset is based on official/public Belarusian macroeconomic time series:

- monthly CPI change from Belstat;
- average official USD/BYN exchange rate from the National Bank of the Republic of Belarus;
- refinancing rate from the National Bank of the Republic of Belarus;
- industrial production index from Belstat.

The file `data/input/macro_factors_source_levels.csv` contains source-level monthly data and derived factor columns. The model uses transformed monthly variables:

| Model variable | Transformation |
|---|---|
| `inflation_mom_pct` | Monthly CPI change, percent |
| `usd_byn_depreciation_pct` | Percent change in average USD/BYN rate; positive value means BYN depreciation |
| `refinancing_rate_change_pp` | Monthly change in refinancing rate, percentage points |
| `industrial_prod_yoy_change_pp` | Monthly change in industrial-production YoY growth, percentage points |

Notes:

- Monthly inflation values are rounded to one decimal place in the dataset; accumulated annual values can slightly differ from official annual indexes due to rounding.
- The regression uses transformed monthly factors, not the raw macro levels.
- The data should be re-verified and extended before any production use.

Source URLs for manual verification:

- https://www.belstat.gov.by/
- https://www.nbrb.by/statistics/rates/avgrate
- https://www.nbrb.by/statistics/monetarypolicyinstruments/refinancingrate
