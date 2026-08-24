"""CLI descrittiva per campioni Kibot tick con bid/ask."""

from __future__ import annotations

import argparse
import sys

from adaptive.kibot_tick_data import KibotTickLoader


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Valida tick Kibot Date,Time,Price,Bid,Ask,Size e li aggrega a "
            "15m senza generare segnali o P&L."
        )
    )
    parser.add_argument("--file", required=True)
    parser.add_argument("--ticker", default="IVE")
    parser.add_argument("--interval", default="15m", choices=("15m",))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    result = KibotTickLoader().load(arguments.file, arguments.ticker)
    audit = result.audit

    print("\nTRADINGAI KIBOT TICK DATA AUDIT — DESCRIPTIVE ONLY")
    print("Schema: Date, Time, Price, Bid, Ask, Size; timezone New York.")
    print(
        "Record identici non deduplicati: possono rappresentare esecuzioni "
        "distinte nello stesso secondo."
    )
    print("Nessun segnale, ordine, size operativa, outcome futuro o P&L.")
    print(f"Ticker:                         {audit.ticker}")
    print(f"Periodo:                        {audit.first_timestamp} -> {audit.last_timestamp}")
    print(f"Tick RTH:                       {audit.regular_ticks}")
    print(f"Sessioni:                       {audit.observed_sessions}")
    print(f"Barre 15m:                      {audit.output_bars}")
    print(f"Copertura bucket 15m:           {audit.bucket_coverage:.1%}")
    print(f"Righe con timestamp condiviso:  {audit.duplicate_timestamp_rows}")
    print(f"Righe esattamente ripetute:     {audit.exact_duplicate_rows}")
    print(f"Spread mediano:                 {audit.median_spread_bps:.3f} bps")
    print(f"Spread p90:                     {audit.spread_p90_bps:.3f} bps")
    print(f"Quote crossed:                  {audit.crossed_quote_fraction:.3%}")
    print(f"Quote locked:                   {audit.locked_quote_fraction:.3%}")
    print(f"Trade fuori NBBO registrato:    {audit.outside_nbbo_fraction:.3%}")
    print(f"Tick mediani per barra:         {audit.median_ticks_per_bar:.1f}")
    print(f"Tick minimi per barra:          {audit.minimum_ticks_per_bar}")
    print(f"Esito:                          {audit.status.value}")
    for reason in audit.reasons:
        print(f"- {reason}")
    print(
        "Il campione descrive IVE (S&P 500 Value ETF), non l'intero indice "
        "S&P 500."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
