# Intraday common-factor decomposition

This report decomposes contemporaneous 15-minute close changes into a first
cross-asset principal component and an orthogonal residual. It is descriptive
only and does not calculate a directional label, prediction, future outcome,
operation, size, stop, leverage, or P&L.

## Input contract

- Only consecutive completed 15-minute closes from the same New York session
  are used.
- Overnight changes and gaps are excluded.
- Every scope uses complete aligned rows across its declared asset list.
- `ORDINARY_ASSETS` excludes VXX; `ALL_ASSETS_DESCRIPTIVE` retains it only as
  a sensitivity view.
- The sign of each loading vector is oriented deterministically from its
  largest absolute element, without choosing a preferred asset.

## Whole-sample outputs

For each scope the report writes:

- fraction of cross-asset variance explained by the first component;
- raw spectral effective asset count;
- residual effective asset count after removing the first component;
- raw and residual median absolute correlations;
- per-asset loading, squared loading share, and explained variance share.

The residual is an analytical decomposition, not a tradeable series and not a
model input selected by this report.

## Fixed-block stability

The decomposition is recomputed independently in each fixed 30-session block.
The block table reports component share, raw and residual breadth, and absolute
cosine similarity of adjacent loading vectors. The absolute value makes the
comparison invariant to the arbitrary sign of a principal-component vector.
Adding future sessions cannot alter an already-complete block.

The labels are descriptive:

- `STABLE_COMMON_MODE`;
- `VARIABLE_COMMON_MODE`;
- `UNSTABLE_COMMON_MODE`;
- `INSUFFICIENT_BLOCKS`.

They describe persistence of the common cross-asset structure. They do not
measure accuracy, profitability, or operational suitability.
