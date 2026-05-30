# Case study: Belarusian equity clustering

The purpose of this case is to demonstrate how a risk/data analyst can reduce a redundant asset universe before portfolio optimization. Instead of manually selecting equities, the module uses the correlation structure of monthly returns.

The clustering stage uses monthly simple returns. This is kept intentionally to reproduce the selected universe and maintain consistency with the downstream modules.

The main deliverable is a selected 7-asset price matrix used as the fixed input for the CVaR portfolio optimization module.
