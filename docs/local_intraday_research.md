# Local 1m -> 15m descriptive research

This research path uses local FirstRate Data ZIP samples instead of the
60-session Yahoo intraday window. It validates and aggregates data, then runs
the existing descriptive feature, stability, and Lorentzian geometry reports.
It does not calculate a directional label, entry, exit, position, order,
leverage, future outcome, or P&L.

## Supported input

Archive names must follow:

```text
TICKER_1min_*.zip
```

Each ZIP must contain exactly one UTF-8 CSV/TXT file. Supported schemas are:

```text
timestamp,open,high,low,close,volume
timestamp,open,high,low,close
```

The second schema is classified as `PRICE_ONLY`. Squeeze Momentum and CHOP
remain available, CMF is marked `UNAVAILABLE_NO_VOLUME`, and the Lorentzian
descriptor uses three features instead of four. Volume is never fabricated.

The loader rejects encrypted archives, unsafe internal paths, invalid CRC,
duplicate or off-grid timestamps, non-numeric values, and incoherent OHLC
relations.

## Time and calendar contract

- Source timestamps are interpreted as `America/New_York`.
- Only the regular USA session is retained.
- Early closes are declared explicitly in `LocalIntradayDataConfig`; the
  loader never infers an exchange calendar from price movement.
- A historical period outside the configured calendar must provide its
  applicable early-close dates before it is treated as complete.

The 15-minute aggregation is session-local and left-labelled:

- open: first 1-minute open;
- high: maximum 1-minute high;
- low: minimum 1-minute low;
- close: last 1-minute close;
- volume: sum of 1-minute volume.

A bucket is emitted only when all 15 expected minute timestamps are present.
Missing buckets are not filled, interpolated, or carried forward. Tests verify
that adding future minutes cannot change any already-closed 15-minute bucket.

## Run

Activate the project virtual environment, then run:

```powershell
python -m adaptive.run_local_intraday_research `
  --data-dir C:\path\to\firstrate-zips `
  --tickers SPY QQQ DIA EEM AAPL AMZN META TSLA SPX NDX VXX `
  --sessions-per-block 30 `
  --minimum-complete-blocks 4 `
  --output-dir reports\v15
```

`--output-dir` enables all 28 standard descriptive CSV files and a deterministic
integrity manifest. Individual output arguments remain available and override
only their matching standard path.

No Docker, LEAN, QuantConnect Cloud, or network download is used by this
command.

## Report sections

1. Source and resampling quality: sessions, 1m rows, 15m bars, coverage,
   complete buckets, early closes, and missing minutes.
2. Feature distributions: exact available reference formulas for Squeeze,
   CHOP, and CMF when real volume exists.
3. Historical block stability: persistent distribution changes in numeric
   features, categorical states, and session phases.
4. Causal Lorentzian geometry: trailing robust normalization, historical-only
   neighbors, embargo, and whole-sample density/distance summaries.
5. Descriptive stability matrix: one tidy row for each asset, feature, and
   scope (`ALL_SESSION`, `OPEN`, `MID_SESSION`, or `CLOSE`).
6. Descriptive context atlas: observed frequencies and historical coverage for
   each session-phase x CHOP x Squeeze combination.
7. Intraday context transitions: counts only between adjacent valid 15-minute
   bars in the same session; overnight and missing buckets break the chain.
8. Context stability across fixed blocks: TVD and Jensen-Shannon comparisons
   for phase occupancy and within-session transition distributions.
9. Lorentzian neighborhood memory stability: coverage, distance, neighbor age,
   and overlap across 520/1,040/2,080/4,000-bar historical windows, separately
   for phase-only and joint phase x CHOP x Squeeze filters.
10. Soft Lorentzian surface: a pre-declared recency-half-life x context-penalty
    grid that measures context agreement, session diversity, age, distance,
    and overlap without hard-filtering the full context.
11. Soft-profile block stability: the frozen `HL_2080_CTX_0.5` profile is
    compared across fixed 30-session blocks using geometry and neighbor
    diversity only.
12. Cross-asset dependence: pairwise contemporaneous dependence and spectral
    effective breadth show how many nominal tickers contribute distinct
    evidence, including fixed-block stability.
13. Common-factor decomposition: the first contemporaneous component is
    separated from residual variation and compared across fixed blocks.
14. Residual pair dependence: the common component is removed independently
    in each fixed complete block before contemporaneous residual correlations
    are compared.
15. Stable proxy groups: only high positive raw correlations that remain high
    across eligible blocks form a group; residual correlation is reported but
    does not define membership.
16. Proxy sensitivity: every possible retained member of each stable group is
    enumerated on the same complete timestamps. No representative is selected.
17. Proxy block sensitivity: the identical scenario grid is recomputed inside
    each fixed historical block with no estimates borrowed across blocks.
18. Proxy temporal stability: per-scenario minimum, median, maximum, and range
    are summarized using complete eligible blocks only.
19. Representative invariance: within each block, the dispersion across the
    complete set of proxy-member choices is reported without naming a winner.
20. Phase-specific cross-asset structure: common-factor share, raw and
    residual effective breadth, loadings, and fixed-block stability are
    recomputed separately for `OPEN`, `MID_SESSION`, and `CLOSE`. Returns that
    cross a phase boundary are excluded.
21. Phase-specific residual dependence: the common component is re-estimated
    independently within every phase and complete block. Pairwise raw and
    residual correlations are compared across the full phase grid, including
    sign reversals and phase-specific proxy consistency.
22. Artifact audit: the complete output set is checked for missing or stale
    CSVs, hashes, duplicate rows, non-finite values, all-missing columns, and
    operational field names. The result is stored in
    `intraday_report_manifest.csv`.

The matrix uses only descriptive labels:

- `STABLE_DESCRIPTIVE`;
- `MONITOR_ISOLATED_SHIFT`;
- `CONTEXT_REQUIRED`;
- `INSUFFICIENT`;
- `NOT_AVAILABLE`.

`SOURCE_LIMITED` is retained at asset level for sources such as VXX. These
labels are data-quality and drift states, not model approvals or trading
decisions.

Lorentzian normalization is estimated independently for each 15-minute slot
using a 60-session trailing window and a 30-session warm-up. Candidate
neighbors remain restricted to the same broad session phase. This removes
known intraday seasonality without using future information.

## Local result handling

Source periods, coverage values, missing-row counts, feature distributions,
and derived dependence statistics remain in the generated `--output-dir` and
its integrity manifest. They are deliberately not copied into public source
documentation. Price-only indices remain separated from OHLCV assets, while
special-context series remain in their dedicated sensitivity scope.

See [intraday_residual_redundancy.md](intraday_residual_redundancy.md) for the
residual-dependence and stable-proxy contracts.

See [intraday_phase_dependence.md](intraday_phase_dependence.md) for the
same-phase return contract, block stability, and phase-contrast definitions.

See
[intraday_phase_residual_dependence.md](intraday_phase_residual_dependence.md)
for the phase-specific residual and proxy-consistency contract.

See [intraday_report_audit.md](intraday_report_audit.md) for the reproducible
artifact-integrity contract and standalone audit command.

These findings improve the descriptive research base. They do not establish
profitability or approve an operational strategy.

The context atlas is documented separately in
`docs/intraday_regime_atlas.md`. It is not connected to the existing
operational regime detector or any strategy module.

The memory stress test is documented in
`docs/lorentzian_neighborhood_stability.md`. Its overlap labels describe only
geometric sensitivity and do not select a model or historical window.

The soft surface is documented in `docs/lorentzian_soft_surface.md`. It does
not select a profile and is not connected to any strategy or execution code.

The block comparison is documented in
`docs/lorentzian_soft_block_stability.md`. Its labels describe only geometric
variation across time and do not measure predictive quality.

Cross-asset dependence is documented in
`docs/intraday_cross_asset_dependence.md`. It uses only already-completed
same-session changes and contemporaneous feature states; it does not use a
future label or approve a model.

The common-factor decomposition is documented in
`docs/intraday_common_factor.md`. Its loadings and residual breadth are
descriptive statistics and are not connected to a strategy or execution path.
