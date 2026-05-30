# Case study: CVaR defensive portfolio optimization

This module focuses on tail-risk portfolio construction. Unlike the original academic draft, the clustering step is no longer repeated here: the asset universe is treated as a predefined input.

Professional workflow:

```text
selected asset prices → log returns → portfolio optimization → risk metrics → interpretation
```

This makes the case easier to review: it demonstrates Expected Shortfall / CVaR, long-only constraints, portfolio strategy comparison and tail-risk interpretation.
