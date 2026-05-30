# Limitations

This repository is a portfolio case study and not an investment recommendation.

Main limitations:

- the sample contains only monthly observations for a short period;
- the Belarusian equity market is illiquid and has irregular trading activity;
- historical CVaR is sample-dependent and should be validated out of sample before practical use;
- factor regressions have low statistical power because the sample is short;
- refinancing-rate changes are rare, so its coefficient has higher uncertainty;
- most factor-model p-values are not statistically significant, so coefficients should be read as sensitivity indicators rather than causal estimates.

Recommended future improvements:

- add a longer history;
- run out-of-sample tests;
- add liquidity constraints;
- include transaction costs;
- test robustness to alternative clustering methods and return definitions;
- compare historical CVaR with parametric or bootstrap CVaR estimates.
