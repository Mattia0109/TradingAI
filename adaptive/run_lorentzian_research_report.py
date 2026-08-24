"""CLI per il vicinato Lorentziano descrittivo su barre USA a 15 minuti."""

from __future__ import annotations

import argparse
import sys

from adaptive.intraday_audit import IntradayReadiness, IntradayResearchAuditor
from adaptive.intraday_features import IntradayReferenceFeatureEngine
from adaptive.intraday_stability import (
    add_dimensionless_squeeze_features,
    regular_session_frame,
)
from adaptive.lorentzian_research import (
    CausalLorentzianResearchEngine,
    LorentzianResearchConfig,
    session_phase_context,
    summarize_lorentzian_distribution,
)
from adaptive.run_adaptive_scan import load_markets
from data_engine.pipeline import MarketDataPipeline


DEFAULT_LORENTZIAN_UNIVERSE = ("SPY", "QQQ", "IWM", "GLD", "TLT")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Descrive il vicinato storico Lorentziano delle feature 15m. "
            "Non produce classificazioni direzionali o decisioni operative."
        )
    )
    parser.add_argument(
        "--period",
        default="60d",
        help="Finestra dati; lo storico intraday del provider resta limitato.",
    )
    parser.add_argument("--interval", default="15m", choices=("15m",))
    parser.add_argument(
        "--normalization-window",
        type=int,
        default=260,
        help="Barre trailing usate per mediana e IQR.",
    )
    parser.add_argument(
        "--normalization-min-periods",
        type=int,
        default=130,
        help="Warm-up minimo della normalizzazione causale.",
    )
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--minimum-candidates", type=int, default=16)
    parser.add_argument("--embargo-bars", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--history-limit", type=int, default=2_000)
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_LORENTZIAN_UNIVERSE),
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


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    tickers = _tickers(arguments.tickers)
    if not tickers:
        raise ValueError("Serve almeno un ticker.")

    config = LorentzianResearchConfig(
        normalization_window=arguments.normalization_window,
        normalization_min_periods=arguments.normalization_min_periods,
        neighbors=arguments.neighbors,
        minimum_candidates=arguments.minimum_candidates,
        embargo_bars=arguments.embargo_bars,
        sample_stride=arguments.sample_stride,
        history_limit=arguments.history_limit,
    )
    markets, download_errors = load_markets(
        MarketDataPipeline(),
        tickers,
        arguments.period,
        arguments.interval,
    )
    if not markets:
        raise ValueError("Nessun mercato disponibile per il report Lorentziano.")

    auditor = IntradayResearchAuditor()
    feature_engine = IntradayReferenceFeatureEngine()
    lorentzian_engine = CausalLorentzianResearchEngine(config)
    rows: list[tuple] = []
    errors = dict(download_errors)
    first_report = None

    for ticker, market in markets.items():
        try:
            audit = auditor.audit(ticker, market)
            if audit.status is IntradayReadiness.REJECTED:
                raise ValueError("Audit dati REJECTED.")
            regular, _ = regular_session_frame(market)
            feature_report = feature_engine.compute(regular)
            features = add_dimensionless_squeeze_features(
                feature_report.values,
                regular,
            )
            context = session_phase_context(features.index)
            report = lorentzian_engine.compute(features, context)
            first_report = first_report or report
            summary = summarize_lorentzian_distribution(report)
            rows.append(
                (
                    ticker,
                    summary.total_rows,
                    summary.complete_rows,
                    summary.coverage,
                    summary.median_distance,
                    summary.distance_p90,
                    summary.median_density,
                    summary.median_neighbor_age_bars,
                )
            )
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"

    if not rows:
        raise ValueError("Nessun asset ha prodotto descrittori Lorentziani.")

    print("\nTRADINGAI 15M LORENTZIAN NEIGHBORHOOD — RESEARCH-ONLY")
    print("Formula: sum(log1p(abs(delta))), senza radice quadrata.")
    print(
        "Normalizzazione robusta trailing; vicini solo storici, stessa fase "
        "di sessione ed embargo."
    )
    print(
        "Nessuna direzione, previsione, operazione, size, stop, leva, outcome "
        "futuro o P&L."
    )
    print("Le statistiche descrivono l'intero campione, non l'ultima barra.")
    print()
    print(
        f"{'TICKER':10} {'ROWS':>6} {'COMP':>6} {'COVER':>8} "
        f"{'D_MED':>9} {'D_P90':>9} {'DENS':>8} {'AGE_MED':>9}"
    )
    print("-" * 83)
    for row in sorted(rows):
        ticker, total, complete, coverage, median, p90, density, age = row
        print(
            f"{ticker:10} {total:6d} {complete:6d} {coverage:8.1%} "
            f"{median:9.4f} {p90:9.4f} {density:8.4f} {age:9.1f}"
        )

    print("\nCONTRATTO DI PARITA' E PROVENIENZA")
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
