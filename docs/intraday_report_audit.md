# Intraday report artifact audit

The v15 research runner creates 28 descriptive CSV files inside a dedicated
`--output-dir`. After the last file is written, it also creates
`intraday_report_manifest.csv`.

The manifest verifies, for every expected report:

- presence and byte size;
- SHA-256 content hash;
- row and column counts;
- duplicate rows;
- non-finite numeric values;
- columns containing only missing values;
- column names associated with operational output such as signals, orders,
  positions, stops, forecasts, trades, or P&L.

Unexpected CSV files are reported too. This prevents a new run from being
silently mixed with stale outputs from an older version. The manifest itself
is ignored when the directory is audited again, so repeated checks are
deterministic.

The integrated audit runs automatically when every standard output remains
inside `--output-dir`. If explicit output paths are used outside that
directory, run the standalone checker on the completed directory:

```powershell
python -m adaptive.run_intraday_report_audit `
  --report-dir reports\v15
```

`PASS` means that the report set is complete and structurally clean. It does
not mean that an indicator is accurate, profitable, or suitable for live use.
The audit reads artifacts only and does not calculate signals, future
outcomes, orders, positions, or P&L.

After a complete integrated run, the companion research freeze binds this
manifest to the exact source archives, per-session bar fingerprints, frozen
parameters, and descriptive implementation. See
`docs/intraday_research_freeze.md`.
