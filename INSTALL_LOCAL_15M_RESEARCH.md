# Install the local 15m research pack

Use this bundle only on `feat/v1-signal-engine` after pulling the current
remote branch.

```powershell
cd C:\TradingAI
& .\venv\Scripts\Activate.ps1
git switch feat/v1-signal-engine
git pull --ff-only origin feat/v1-signal-engine
```

Extract the bundle into the repository root, preserving the `adaptive` and
`docs` directories. If the ZIP is in `Downloads`:

```powershell
Expand-Archive `
  -Path "$env:USERPROFILE\Downloads\TradingAI_local_15m_research_pack.zip" `
  -DestinationPath . `
  -Force
```

Then verify:

```powershell
python -m pytest -q
```

The complete suite for this bundle is expected to report `404 passed` on the
current branch state.

Place the FirstRate Data ZIP files together in a separate local directory and
run, for example:

```powershell
python -m adaptive.run_local_intraday_research `
  --data-dir C:\TradingAI\data\firstrate `
  --tickers SPY QQQ DIA EEM AAPL AMZN META TSLA SPX NDX VXX `
  --sessions-per-block 30 `
  --minimum-complete-blocks 4 `
  --output-dir reports\v15
```

`--output-dir` creates all 28 standard descriptive CSV paths plus
`intraday_report_manifest.csv` and `intraday_research_freeze.json`. The
manifest records hashes, schemas, row counts and integrity checks, while the
freeze binds them to parameters, implementation, source archives and each
historical session. Both reject silent contamination or stale outputs. Explicit
per-report paths remain available and take precedence when supplied; when one
is outside the directory, run the standalone audit on the completed set.

Price-only index archives such as SPX and NDX are accepted too. They use
Squeeze Momentum, CHOP, and a three-feature Lorentzian descriptor; CMF remains
explicitly unavailable because real volume is absent.

The two additional CSV files contain only observed context frequencies and
within-session transitions. A transition is recorded only between consecutive
valid 15-minute bars in the same session; overnight boundaries, missing
buckets, and warm-up rows break the chain.

The context-stability CSV then compares the observed distributions across
fixed complete blocks using total variation and Jensen-Shannon divergence. It
also requires the observed drift to exceed a deterministic permutation null.
It does not use future outcomes or connect to the operational regime detector.

The Lorentzian stability CSV compares 520, 1,040, 2,080, and 4,000-bar
historical memories. It reports coverage, neighbor age, distance, and neighbor
set overlap for phase-only and phase + CHOP + Squeeze contexts. It remains a
geometric stress test and contains no future label or performance metric.

The soft-surface CSV retains phase-only historical candidates and evaluates a
pre-declared grid of recency half-lives and CHOP/Squeeze disagreement
penalties. It reports only neighbor geometry, context agreement, and session
diversity; no profile is selected automatically.

The two soft-block CSV files then freeze `HL_2080_CTX_0.5` and compare its
geometry across fixed 30-session blocks. They report coverage, distance, age,
context agreement, overlap, distinct-session count, effective-session count,
and maximum concentration in one session. The labels describe only block
variation and never use future outcomes.

The three cross-asset CSV files measure contemporaneous dependence between
asset pairs and the spectral effective breadth of the nominal ticker set.
Only consecutive completed 15-minute close changes from the same session are
used. VXX is excluded from the ordinary-asset scope and retained only in the
separate all-asset sensitivity view. These statistics prevent correlated
proxies from being counted as independent evidence.

The three common-factor CSV files separate the first contemporaneous
cross-asset component from the residual variation. They report component
share, deterministic per-asset loadings, residual effective breadth, and
adjacent-block loading stability. The decomposition is analytical only and is
not connected to a model or operational decision.

The seven residual/proxy CSV files extend that audit without selecting assets.
The residual component is re-estimated independently inside every fixed
complete block. Stable proxy groups instead require a high positive raw
correlation overall and in every eligible block. Every possible retained
member of each group is enumerated in the sensitivity table; no representative
is chosen automatically. The same scenario grid is recomputed independently
inside every fixed block, while separate outputs summarize temporal ranges and
the within-block dispersion across all representative choices. Incomplete
blocks and incomplete combination grids cannot contribute to those summaries.

The four phase-factor CSV files then repeat the cross-asset decomposition
separately for `OPEN`, `MID_SESSION`, and `CLOSE`. A return is retained only
when both bars belong to the same phase, so phase-boundary changes are
excluded. Fixed-block tables and the phase-contrast summary describe whether
the common component and effective breadth differ across the trading day;
they remain contemporaneous and descriptive.

The three phase-residual CSV files re-estimate the common component in every
phase and every complete block, then compare pairwise residual dependence
across the declared phase grid. Sign reversals and phase-specific proxy links
are reported explicitly, without creating a directional relationship or using
a future outcome.

The artifact audit runs automatically at the end of the command. It must show
`28/28`, zero unexpected CSVs, and `PASS`. It verifies report integrity only;
it does not measure accuracy or profitability. It can also be repeated with:

```powershell
python -m adaptive.run_intraday_report_audit `
  --report-dir reports\v15
```

This command uses local files only. Docker, LEAN, QuantConnect Cloud, and a
TradingView subscription are not required.

For the separate Kibot IVE tick sample:

```powershell
python -m adaptive.run_kibot_tick_audit `
  --file "C:\path\to\IVE (1).txt" `
  --ticker IVE
```

After the audit, run the integrated descriptive pipeline:

```powershell
python -m adaptive.run_kibot_intraday_research `
  --file "C:\path\to\IVE (1).txt" `
  --ticker IVE
```

It calculates 15m feature distributions, block stability, microstructure by
session phase, and causal Lorentzian neighborhood geometry.

IVE is handled separately because it is tick data with bid/ask. Its concrete
coverage and microstructure statistics remain in the locally generated report.

The report is descriptive: it contains no direction, order, size, stop,
leverage, future outcome, or P&L.

If the v15 reports already exist, validate every source ZIP, create the
immutable baseline freeze, and run its zero-forward-data self-check with one
command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\freeze_intraday_baseline.ps1
```

The script uses `venv\Scripts\python.exe` directly, so activation is optional.
Docker and LEAN are not used. The expected initial readiness state is
`INCONCLUSIVE` with zero new sessions; this confirms the chronology guard and
does not evaluate a strategy.

To compare a later untouched checkpoint with the frozen baseline, follow
`docs/intraday_research_freeze.md`. The forward-readiness result describes
sample integrity and coverage only; it does not approve a strategy.
