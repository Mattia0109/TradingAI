# Intraday research freeze and forward-readiness protocol

This protocol closes the gap between a structurally valid descriptive report
and a genuinely unseen future checkpoint. It remains research-only: it never
calculates a direction, signal, order, position size, stop, leverage, future
outcome, or P&L.

## What is frozen

`intraday_research_freeze.json` contains deterministic fingerprints for:

- all parameters that define the 1m-to-15m descriptive pipeline;
- the implementation files used by features, context analysis, cross-asset
  dependence, and Lorentzian geometry;
- every source archive;
- every completed 15-minute session, separately;
- the exact 28-report set and its structural audit;
- the ticker universe represented by both reports and source archives.

The freeze file is self-verifying. An existing freeze can be written again
only when its content is identical. A different checkpoint must use a new
directory, preventing accidental overwrite of the baseline.

With `--output-dir`, the complete local research runner creates the freeze
automatically after the 28 CSV files pass their audit:

```powershell
python -m adaptive.run_local_intraday_research `
  --data-dir C:\TradingAI\data\firstrate `
  --tickers SPY QQQ DIA EEM AAPL AMZN META TSLA SPX NDX VXX `
  --sessions-per-block 30 `
  --minimum-complete-blocks 4 `
  --output-dir reports\baseline_v1
```

An already completed report directory can be frozen without recomputing all
analyses:

```powershell
python -m adaptive.run_research_freeze `
  --data-dir C:\TradingAI\data\firstrate `
  --report-dir reports\v15_complete `
  --tickers SPY QQQ DIA EEM AAPL AMZN META TSLA SPX NDX VXX
```

All source archives must still be intact. The command refuses missing,
truncated, substituted, or mismatched archives.

## Forward checkpoint

After new data has been collected, the same frozen code and parameters create
a new report directory and a new freeze. Compare it with the baseline:

```powershell
python -m adaptive.run_forward_research_readiness `
  --baseline-freeze reports\baseline_v1\intraday_research_freeze.json `
  --candidate-freeze reports\checkpoint_01\intraday_research_freeze.json
```

The readiness summary and detail CSV are written below the candidate's
`forward_readiness` subdirectory. This keeps evidence outputs separate from
the frozen 28-report directory and prevents them from becoming unexpected
report inputs during a later integrity audit.

The audit supports either a cumulative archive containing the unchanged
baseline plus new sessions, or a separate forward-only archive beginning
strictly after the baseline cutoff. It rejects:

- changed historical 15-minute sessions;
- inserted pre-cutoff sessions;
- missing baseline sessions in cumulative data;
- a changed feature/Lorentzian specification;
- a changed descriptive implementation;
- a changed source universe;
- a report set that no longer passes the structural audit.

The default readiness target is 60 common new sessions, divided into three
20-session checkpoints, with at least five eligible sources. These are sample
coverage requirements, not performance thresholds.

Possible states are:

- `FAIL`: chronology or integrity was violated;
- `INCONCLUSIVE`: the checkpoint is clean but still too short;
- `PASS`: enough untouched data exists for a later blinded scientific review.

`PASS` does not mean that an indicator is accurate or that a strategy works.
It only establishes that the next review can be performed without silently
reusing or rewriting the baseline.
