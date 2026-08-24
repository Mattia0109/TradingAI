"""Report distributivo delle feature 15m, privo di output operativo."""

from __future__ import annotations

import argparse
import sys

from adaptive.intraday_audit import IntradayReadiness, IntradayResearchAuditor
from adaptive.intraday_features import IntradayReferenceFeatureEngine
from adaptive.run_adaptive_scan import load_markets
from data_engine.pipeline import MarketDataPipeline


DEFAULT_FEATURE_UNIVERSE = ("SPY", "QQQ", "IWM", "GLD", "TLT")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Riassume distribuzioni storiche di CHOP, Squeeze e CMF su barre "
            "15m. Non genera segnali o performance."
        )
    )
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="15m", choices=("15m",))
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_FEATURE_UNIVERSE),
    )
    return parser.parse_args(argv)


def _tickers(values) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).upper().strip()
            for value in values
            if str(value).strip()
        )
    )


def _percentage(series, value: str) -> float:
    valid = series.loc[series != "INSUFFICIENT"]
    if valid.empty:
        return 0.0
    return float((valid == value).mean())


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    tickers = _tickers(arguments.tickers)
    if not tickers:
        raise ValueError("Serve almeno un ticker.")
    markets, download_errors = load_markets(
        MarketDataPipeline(),
        tickers,
        arguments.period,
        arguments.interval,
    )
    if not markets:
        raise ValueError("Nessun mercato disponibile per il report feature.")

    auditor = IntradayResearchAuditor()
    engine = IntradayReferenceFeatureEngine()
    rows: list[tuple] = []
    errors = dict(download_errors)
    first_report = None
    for ticker, data in markets.items():
        try:
            audit = auditor.audit(ticker, data)
            if audit.status is IntradayReadiness.REJECTED:
                raise ValueError("Audit dati REJECTED.")
            report = engine.compute(data)
            first_report = first_report or report
            values = report.values
            rows.append(
                (
                    ticker,
                    len(values),
                    int(values["squeeze_momentum"].notna().sum()),
                    int(values["choppiness"].notna().sum()),
                    int(values["cmf"].notna().sum()),
                    _percentage(values["chop_segment"], "CHOPPY"),
                    _percentage(values["chop_segment"], "TRENDING"),
                    _percentage(values["squeeze_state"], "SQUEEZE_ON"),
                )
            )
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"

    if not rows:
        raise ValueError("Nessun asset ha superato il gate feature.")

    print("\nTRADINGAI 15M FEATURE DISTRIBUTIONS — RESEARCH-ONLY")
    print("Nessun segnale, ordine, size, stop, leva o P&L viene calcolato.")
    print("Le percentuali descrivono l'intero campione, non l'ultima barra.")
    print()
    print(
        f"{'TICKER':10} {'ROWS':>6} {'SQZ_N':>7} {'CHOP_N':>7} "
        f"{'CMF_N':>7} {'CHOPPY':>9} {'TREND':>9} {'SQZ_ON':>9}"
    )
    print("-" * 82)
    for row in sorted(rows):
        ticker, total, squeeze_n, chop_n, cmf_n, choppy, trend, squeeze_on = row
        print(
            f"{ticker:10} {total:6d} {squeeze_n:7d} {chop_n:7d} "
            f"{cmf_n:7d} {choppy:9.1%} {trend:9.1%} {squeeze_on:9.1%}"
        )

    print("\nCONTRATTO DI PARITÀ")
    for name, status in first_report.parity.items():
        print(f"- {name}: {status.value}")
    for caveat in first_report.caveats:
        print(f"- {caveat}")
    if errors:
        print("\nERRORI ISOLATI")
        for ticker, error in sorted(errors.items()):
            print(f"- {ticker}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
