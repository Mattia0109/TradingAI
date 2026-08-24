# Residual dependence and stable proxy groups

This module audits redundancy in the local 15-minute descriptive sample. It
does not calculate a future label, directional forecast, entry, exit, order,
position, size, stop, leverage, economic result, or P&L.

## Input contract

Only consecutive, completed 15-minute close changes from the same regular
session are used. All assets in a scope must be present at the same timestamp;
missing values are not filled. `VXX` remains outside `ORDINARY_ASSETS` and is
included only in `ALL_ASSETS_DESCRIPTIVE`.

Session blocks are fixed by chronological position and contain the declared
number of sessions. An incomplete final block is retained in the output but
cannot qualify as evidence. Adding future sessions therefore cannot reassign
or change an already completed historical block.

## Residual pair dependence

For each scope, the input matrix is standardized and its first principal
component is estimated. The contemporaneous reconstruction from that component
is subtracted and the pairwise correlations of the residual matrix are
reported.

The same decomposition is re-estimated independently inside each fixed,
complete block. The block calculation never borrows loadings, means, standard
deviations, or observations from a later block. This makes the chronological
stability audit causal and keeps future extensions from altering completed
blocks.

Residual correlation is a subspace diagnostic. In particular, removing one
dominant component from two nearly identical proxies can leave small residuals
with an unstable or negative correlation. For that reason, residual
correlation does not decide proxy membership.

## Stable proxy contract

A pair forms a stable proxy link only when all declared raw conditions hold:

- at least the configured minimum number of eligible complete blocks;
- overall raw close-change correlation of at least `0.95`;
- minimum eligible-block raw correlation of at least `0.90`;
- maximum adjacent eligible-block raw-correlation change no greater than
  `0.05`.

Connected stable links form deterministic groups. A group identifies
redundancy only; it does not rank its members or name a preferred
representative. Singletons remain explicit in the membership table.

## Sensitivity without representative selection

The sensitivity table contains:

- the full-scope baseline;
- every combination that retains exactly one member from each stable proxy
  group while retaining all singletons;
- one scenario that removes each complete stable proxy group.

Every scenario uses the baseline's same complete timestamp set, so changes
come from asset composition rather than changing sample coverage. If the full
combination grid would exceed the declared cap, the table records
`COMBINATION_LIMIT_EXCEEDED` and does not emit a partial, potentially biased
grid.

The output compares only descriptive quantities: spectral effective breadth,
first-component share, raw pair dependence, residual effective breadth, and
residual pair dependence. It never promotes a scenario or feeds an operational
component.

## Fixed-block proxy sensitivity

The complete non-selective scenario grid is repeated independently in every
fixed historical block. Means, correlations, factor loadings, and residuals
are therefore re-estimated from that block alone. An incomplete final block is
retained for auditability, but all scenario metrics are unavailable and it
cannot enter the temporal summary.

The block-stability table reports raw minimum, median, maximum, and range for
each scenario. These are descriptive dispersion statistics, not pass/fail
criteria and not a selection rule.

For every complete block, the representative-invariance table also measures
the maximum-minus-minimum dispersion across every combination that retains one
member from each stable proxy group. It emits values only when the entire
combination grid is present. A combination cap, missing scenario, insufficient
asset count, or incomplete block remains an explicit state rather than being
silently ignored.

## CSV outputs

- `intraday_residual_pair_blocks.csv`: residual correlations per fixed block;
- `intraday_residual_pairs.csv`: overall and block-stability pair summary;
- `intraday_stable_proxy_clusters.csv`: deterministic group membership;
- `intraday_proxy_sensitivity.csv`: complete non-selective sensitivity grid;
- `intraday_proxy_block_sensitivity.csv`: the same scenarios in every fixed
  block;
- `intraday_proxy_block_stability.csv`: per-scenario ranges across eligible
  blocks;
- `intraday_proxy_representative_invariance.csv`: within-block dispersion
  across the complete representative grid.

These files describe redundancy in the observed sample. They do not establish
predictive value or profitability.
