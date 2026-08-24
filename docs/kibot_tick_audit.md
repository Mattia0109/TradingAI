# Kibot tick with bid/ask audit

The free Kibot IVE sample uses the documented six-field layout:

```text
Date,Time,Price,Bid,Ask,Size
```

Each row is one executed trade with the prevailing NBBO bid and ask. Identical
rows are therefore retained: two trades can share the same second, price,
quote, and size.

The loader validates timestamps and numeric fields, limits rows to the regular
USA session, and aggregates trades into left-labelled 15-minute bars:

- open/high/low/close from `Price`;
- volume from the sum of `Size`;
- tick count;
- median and 90th-percentile quoted spread in basis points;
- fraction of recorded trades outside the accompanying NBBO.

It does not create missing ticks or deduplicate executions.

## Run

```powershell
python -m adaptive.run_kibot_tick_audit `
  --file "C:\path\to\IVE (1).txt" `
  --ticker IVE
```

## Local result handling

Concrete coverage, spread, period, and row-count statistics remain in the
locally generated report and are not committed to the public repository.
Source-specific observations must be reviewed together with the audit status
and must not be generalized to an index or a longer historical sample.

No direction, prediction, order, operational size, leverage, future outcome,
or P&L is calculated.
