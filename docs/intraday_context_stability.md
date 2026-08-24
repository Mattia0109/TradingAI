# Intraday context stability across historical blocks

This report checks whether the descriptive intraday context atlas remains
similar across fixed, complete historical blocks. It is separate from every
strategy, forecast, risk, allocation, and execution component.

No future outcome, directional label, signal, order, size, leverage, or P&L is
used.

## Compared distributions

For every asset, consecutive complete blocks are compared in two ways:

1. context occupancy, separately for `OPEN`, `MID_SESSION`, and `CLOSE`;
2. the whole distribution of valid within-session context transitions.

Context means session phase x CHOP segment x Squeeze state. Transitions still
require two valid bars exactly 15 minutes apart in the same session.

## Metrics

- Total variation distance (`TVD`) measures the absolute redistribution of
  probability mass and ranges from 0 to 1.
- Jensen-Shannon divergence (`JS`, bits) measures distributional separation
  symmetrically and remains finite when a category is absent in one block.
- Entropy and effective context count describe block diversity without adding
  any directional interpretation.

Default descriptive shift thresholds are:

| Label | TVD | JS bits |
|---|---:|---:|
| `LOW_SHIFT` | below 0.10 | below 0.02 |
| `MODERATE_SHIFT` | at least 0.10 | at least 0.02 |
| `HIGH_SHIFT` | at least 0.25 | at least 0.08 |

The higher of the two metric levels is retained. A comparison is
`INSUFFICIENT` when the declared block or row minimum is not met.

An absolute threshold is not sufficient by itself. For every block pair, 96
deterministic label permutations preserve the two sample sizes and the pooled
category frequencies. A shift is elevated only when the observed metric also
exceeds the 95th percentile of this null distribution. The CSV retains the
null quantiles and the observed excess for auditability.

## Persistence patterns

Consecutive block comparisons are summarized as:

- `LOW_OR_NONE`;
- `ISOLATED_SHIFT`;
- `PERSISTENT_ELEVATED`;
- `PERSISTENT_HIGH`;
- `INSUFFICIENT`.

These labels describe distribution drift only and are not model approvals.

## Export

Add this option to the local research command:

```powershell
--context-stability-output reports\intraday_context_stability.csv
```

The CSV has one row per asset, component, and scope. Fixed blocks mean that
appending later sessions does not change earlier block comparisons.
