# Intraday cross-asset dependence

This report measures how much of the observed 15-minute evidence is shared
between assets. It is descriptive only: it does not calculate a directional
label, prediction, future outcome, operation, size, stop, leverage, or P&L.

## Contemporaneous contract

The price statistic is the log change between two consecutive completed
15-minute closes in the same New York session. Overnight changes and gaps are
excluded. Pairwise tables also compare contemporaneous CHOP, Squeeze,
dimensionless Squeeze Momentum, and CMF when real volume exists.

For each pair the report includes:

- Pearson and Spearman close-change correlation;
- absolute-change correlation;
- fixed-block correlation range and adjacent-block change;
- feature rank correlations;
- CHOP, Squeeze, and joint-state agreement;
- normalized mutual information for the joint context;
- a pre-declared dependence label.

The labels `HIGH_DEPENDENCE`, `MODERATE_DEPENDENCE`, and `LOW_DEPENDENCE`
describe shared contemporaneous evidence. They are not model-quality or
strategy labels.

## Effective breadth

The nominal ticker count can substantially overstate the number of distinct
observations when market proxies and their constituents move together. The
report therefore eigendecomposes the complete-case correlation matrix and
calculates the participation ratio:

```text
effective assets = (sum eigenvalues)^2 / sum(eigenvalues^2)
```

It also reports the largest component share, the median and p90 absolute
pairwise correlations, and the same metrics inside fixed 30-session blocks.
Completed historical block boundaries never move when later sessions are
added.

`ORDINARY_ASSETS` excludes VXX because its instrument structure is different.
`ALL_ASSETS_DESCRIPTIVE` retains it only as a clearly marked sensitivity
view.

## Interpretation limits

- Correlated tickers are not independent replications.
- A negative correlation can still imply strong dependence.
- Context agreement can reflect a shared market state rather than useful
  information.
- Spectral effective breadth is sample-dependent and must be monitored across
  fixed blocks.
- None of these metrics measures accuracy or profitability.

