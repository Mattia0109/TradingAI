# Cross-asset dependence by session phase

This report separates contemporaneous 15-minute cross-asset structure into
`OPEN`, `MID_SESSION`, and `CLOSE`. It is descriptive only: it does not create
directions, predictions, future outcomes, operations, size, stops, leverage,
or P&L.

## Return contract

- New York regular-session timestamps define the phases.
- `OPEN` ends before 10:30, `CLOSE` starts at 15:00, and the remaining bars
  belong to `MID_SESSION`.
- A close change is retained only when both its starting bar and ending bar
  belong to the same phase and the bars are consecutive in the same session.
- The 10:15 -> 10:30 and 14:45 -> 15:00 changes are therefore excluded.
- Overnight changes, missing buckets, and extended hours remain excluded.

## Phase decomposition

For each declared asset scope and phase, the report recomputes independently:

- first-component share;
- raw spectral effective asset count;
- residual effective asset count after removing the first component;
- raw and residual median absolute correlation;
- deterministic per-asset loadings.

The same decomposition is repeated independently inside every fixed complete
30-session block. A phase block needs at least 60 aligned within-phase changes.
Adding future sessions cannot change a previously complete block.

## Phase contrast

The contrast table reports the minimum, median, maximum, and range across all
three phases. Declared thresholds classify the observed dispersion as:

- `LOW_PHASE_HETEROGENEITY`;
- `MODERATE_PHASE_HETEROGENEITY`;
- `HIGH_PHASE_HETEROGENEITY`;
- `INSUFFICIENT_PHASES`.

The labels describe contemporaneous structural variation only. They do not
measure accuracy, profitability, or operational suitability.

## CSV outputs

- `intraday_phase_factor_loadings.csv`;
- `intraday_phase_factor_blocks.csv`;
- `intraday_phase_factor_summary.csv`;
- `intraday_phase_factor_contrast.csv`.
