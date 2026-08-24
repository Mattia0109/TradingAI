# Intraday descriptive context atlas

The context atlas describes which combinations of observable intraday states
are represented in the local 15-minute sample. It is deliberately separate
from `adaptive/regime_detector.py`, the strategy modules, portfolio
construction, and risk controls.

The atlas does not calculate a future return, directional label, signal,
entry, exit, position, order, leverage, or P&L.

## State definition

Each valid 15-minute bar is assigned one context composed of:

- session phase: `OPEN`, `MID_SESSION`, or `CLOSE`;
- CHOP segment: `TRENDING`, `NEUTRAL`, or `CHOPPY`;
- Squeeze state: `SQUEEZE_ON`, `SQUEEZE_OFF`, or `NEUTRAL`.

CMF is not converted into a positive/negative state. The atlas records only
whether real volume makes CMF `AVAILABLE` or whether a price-only source makes
it `NOT_AVAILABLE`. This keeps SPX/NDX explicit and prevents fabricated
volume.

Warm-up rows marked `INSUFFICIENT` by CHOP or Squeeze are excluded. Source
status (`READY_DESCRIPTIVE`, `PRICE_ONLY`, or `LIMITED`) is retained in every
output row.

## Coverage labels

For every asset and context cell, the report counts observations, sessions,
historical blocks, and complete historical blocks. Default coverage requires:

- at least 60 observations;
- at least 10 distinct sessions;
- presence in at least 3 complete 30-session blocks.

The labels are:

- `ADEQUATE_DESCRIPTIVE_COVERAGE`;
- `SPARSE_CONTEXT`;
- `INSUFFICIENT`.

They describe sample coverage only. They are not model approvals.

## Transition contract

A transition is counted only when both states are valid and their timestamps
are exactly 15 minutes apart within the same New York session. The chain is
broken by:

- the overnight boundary;
- a missing 15-minute bucket;
- an invalid warm-up state.

This prevents artificial transitions across unavailable observations. Fixed
session blocks preserve earlier assignments when later sessions are appended.

## Run and export

```powershell
python -m adaptive.run_local_intraday_research `
  --data-dir C:\TradingAI\data\firstrate `
  --tickers SPY QQQ DIA EEM AAPL AMZN META TSLA SPX NDX VXX `
  --sessions-per-block 30 `
  --minimum-complete-blocks 4 `
  --matrix-output reports\intraday_feature_stability_matrix.csv `
  --regime-atlas-output reports\intraday_regime_atlas.csv `
  --regime-transition-output reports\intraday_regime_transitions.csv `
  --context-stability-output reports\intraday_context_stability.csv
```

The first CSV has one row per asset, phase, CHOP segment, and Squeeze state.
The second has one row per observed within-session state transition. Cross-
asset counts in the console report measure presence only; correlated assets
are not treated as independent statistical evidence.

The additional context-stability report compares occupancy and transition
distributions across fixed complete blocks using total variation and
Jensen-Shannon divergence. See `docs/intraday_context_stability.md`.
