# Intraday residual dependence by session phase

This research-only block measures whether contemporaneous cross-asset
dependence left after removing the first common component changes between
`OPEN`, `MID_SESSION`, and `CLOSE`.

It does not calculate directions, predictions, orders, positions, future
outcomes, or P&L.

## Same-phase contract

A close change is admitted only when both its starting bar and ending bar are
inside the same declared session phase. The two boundary changes
`10:15 -> 10:30` and `14:45 -> 15:00` are excluded, as are overnight and
missing-bucket transitions.

The first principal component is estimated independently:

- for each phase over the complete sample;
- for each phase inside every fixed complete session block.

This prevents the phase comparison from reusing a full-session component that
may not describe the phase-specific covariance structure.

## Outputs

- `intraday_phase_residual_pair_blocks.csv` contains raw and residual
  correlations for each pair, phase, and chronological block;
- `intraday_phase_residual_pairs.csv` summarizes each pair within each phase;
- `intraday_phase_residual_contrast.csv` compares the three phase summaries
  for each pair.

The contrast records correlation ranges, the phase with the largest absolute
residual dependence, sign reversals, the number of phases in which a stable
proxy link is present, and a descriptive state.

## Declared states

The default magnitude-range thresholds are declared before observing a new
sample:

- below `0.15`: `LOW_PHASE_RESIDUAL_VARIATION`;
- from `0.15` to below `0.30`: `MODERATE_PHASE_RESIDUAL_VARIATION`;
- at least `0.30`: `HIGH_PHASE_RESIDUAL_VARIATION`.

A sign reversal with material magnitude is labelled
`PHASE_SIGN_REVERSAL`. A raw proxy relation present in only part of the phase
grid is `PHASE_SPECIFIC_PROXY_LINK`; one present in all three phases is
`STABLE_PROXY_ACROSS_PHASES`. Missing phases or too few complete blocks remain
`INSUFFICIENT_PHASES`.

These labels describe contemporaneous covariance geometry only. They are not
directional or prospective evidence.

## One-directory export

`run_local_intraday_research` accepts `--output-dir`. When provided, all
standard descriptive CSV paths are populated automatically. An explicit
per-report path still takes precedence. Existing files outside that directory
are never removed.
