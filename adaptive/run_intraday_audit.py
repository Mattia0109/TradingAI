"""CLI per l'audit descrittivo dei dati equity USA a 15 minuti."""

from __future__ import annotations

import argparse
import sys

from adaptive.intraday_audit import IntradayReadiness, IntradayResearchAuditor
from adaptive.run_adaptive_scan import load_markets
from data_engine.pipeline import MarketDataPipeline


DEFAULT_INTRADAY_UNIVERSE = ("SPY", "QQQ", "IWM", "GLD", "TLT")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Verifica qualità e sufficienza dei dati equity USA a 15 minuti. "
            "Non genera segnali, ordini o performance di trading."
        )
    )
    parser.add_argument(
        "--period",
        default="30d",
        help="Finestra dati Yahoo; per l'intraday il provider limita lo storico.",
    )
    parser.add_argument(
        "--interval",
        default="15m",
        choices=("15m",),
        help="Il profilo V1 supporta soltanto barre da 15 minuti.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_INTRADAY_UNIVERSE),
    )
    return parser.parse_args(argv)


def _unique_tickers(values) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).upper().strip()
            for value in values
            if str(value).strip()
        )
    )


def _print_report(report, download_errors) -> None:
    print("\nTRADINGAI 15M DATA READINESS — RESEARCH-ONLY")
    print("Profilo: ETF/azioni USA, sessione regolare 09:30–16:00 New York.")
    print("Nessun segnale, ordine, size, stop, leva o P&L viene calcolato.")
    print(
        "READY significa soltanto dati idonei allo studio descrittivo delle "
        "feature."
    )
    print(
        "Nota provider: lo storico intraday Yahoo è limitato e non basta, da "
        "solo, per una validazione multi-regime."
    )
    print()
    print(
        f"{'TICKER':10} {'STATUS':10} {'RTH':>7} {'SESS':>5} "
        f"{'COVER':>8} {'FULL':>8} {'MISS':>6} {'GRID':>6} "
        f"{'ZERO-V':>8} {'QUAL':>7}"
    )
    print("-" * 91)
    for ticker, audit in sorted(report.assets.items()):
        print(
            f"{ticker:10} {audit.status.value:10} "
            f"{audit.regular_session_rows:7d} {audit.observed_sessions:5d} "
            f"{audit.median_session_coverage:8.1%} "
            f"{audit.full_session_fraction:8.1%} "
            f"{audit.estimated_missing_bars:6d} {audit.off_grid_rows:6d} "
            f"{audit.zero_volume_fraction:8.1%} "
            f"{audit.data_quality_score:7.1%}"
        )

    print("\nESITO PER ASSET")
    for ticker, audit in sorted(report.assets.items()):
        print(f"{ticker} — {audit.status.value}")
        for reason in audit.reasons:
            print(f"- {reason}")
        if audit.extended_hours_rows:
            print(
                f"- Barre extended-hours escluse dall'audit RTH: "
                f"{audit.extended_hours_rows}."
            )

    all_errors = {**download_errors, **report.errors}
    if all_errors:
        print("\nERRORI ISOLATI")
        for ticker, error in sorted(all_errors.items()):
            print(f"- {ticker}: {error}")

    print(f"\nESITO AGGREGATO DATI: {report.status.value}")
    if report.status is IntradayReadiness.LIMITED:
        print(
            "Lo storico può supportare controlli tecnici o feature "
            "descrittive, non conclusioni robuste tra regimi."
        )
    elif report.status is IntradayReadiness.REJECTED:
        print("Correggere i problemi dati prima di analizzare le feature 15m.")


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    tickers = _unique_tickers(arguments.tickers)
    if not tickers:
        raise ValueError("Serve almeno un ticker.")

    markets, download_errors = load_markets(
        MarketDataPipeline(),
        tickers,
        arguments.period,
        arguments.interval,
    )
    if not markets:
        raise ValueError("Nessun mercato disponibile per l'audit 15m.")

    report = IntradayResearchAuditor().audit_markets(markets)
    _print_report(report, download_errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
