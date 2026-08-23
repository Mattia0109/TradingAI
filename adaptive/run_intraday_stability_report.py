"""CLI per la stabilità descrittiva delle feature intraday a 15 minuti."""

from __future__ import annotations

import argparse
import math
import sys

from adaptive.intraday_audit import IntradayReadiness, IntradayResearchAuditor
from adaptive.intraday_features import IntradayReferenceFeatureEngine
from adaptive.intraday_stability import (
    IntradayFeatureStabilityAnalyzer,
    IntradayStabilityConfig,
    add_dimensionless_squeeze_features,
    regular_session_frame,
    summarize_cross_asset_phase_consensus,
)
from adaptive.run_adaptive_scan import load_markets
from data_engine.pipeline import MarketDataPipeline


DEFAULT_STABILITY_UNIVERSE = ("SPY", "QQQ", "IWM", "GLD", "TLT")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Misura drift, profili di sessione e ridondanza delle feature 15m. "
            "Non genera segnali o performance."
        )
    )
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="15m", choices=("15m",))
    parser.add_argument("--sessions-per-block", type=int, default=15)
    parser.add_argument("--minimum-complete-blocks", type=int, default=3)
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_STABILITY_UNIVERSE),
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
    config = IntradayStabilityConfig(
        sessions_per_block=arguments.sessions_per_block,
        minimum_complete_blocks=arguments.minimum_complete_blocks,
    )
    markets, download_errors = load_markets(
        MarketDataPipeline(),
        tickers,
        arguments.period,
        arguments.interval,
    )
    if not markets:
        raise ValueError("Nessun mercato disponibile per il report stabilità.")

    auditor = IntradayResearchAuditor()
    feature_engine = IntradayReferenceFeatureEngine()
    analyzer = IntradayFeatureStabilityAnalyzer(config)
    reports = []
    errors = dict(download_errors)
    for ticker, data in markets.items():
        try:
            audit = auditor.audit(ticker, data)
            if audit.status is IntradayReadiness.REJECTED:
                raise ValueError("Audit dati REJECTED.")
            regular_data, _ = regular_session_frame(data, config)
            features = feature_engine.compute(regular_data).values
            features = add_dimensionless_squeeze_features(
                features,
                regular_data,
                config,
            )
            reports.append(analyzer.analyze(ticker, features))
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
    if not reports:
        raise ValueError("Nessun asset ha superato il gate di stabilità.")

    print("\nTRADINGAI 15M FEATURE STABILITY — RESEARCH-ONLY")
    print("Nessun segnale, ordine, size, stop, leva, outcome futuro o P&L.")
    print("I blocchi sono fissi: nuovi dati non riassegnano le sessioni passate.")
    print("Le feature vengono calcolate dopo aver escluso extended-hours e weekend.")
    print("Le feature Squeeze del drift sono dimensionless, divise per il close.")

    print("\nCOPERTURA BLOCCHI")
    print(f"{'TICKER':10} {'SESSIONI':>9} {'BLOCCHI':>8} {'ESCLUSE':>8} {'STATO':>20}")
    print("-" * 61)
    for report in sorted(reports, key=lambda item: item.ticker):
        status = (
            "BLOCKS_SUFFICIENT"
            if report.complete_blocks >= config.minimum_complete_blocks
            else "BLOCKS_LIMITED"
        )
        print(
            f"{report.ticker:10} {report.observed_sessions:9d} "
            f"{report.complete_blocks:8d} {report.excluded_rows:8d} {status:>20}"
        )

    print("\nPERSISTENZA DRIFT FEATURE NUMERICHE")
    print(
        f"{'TICKER':10} {'FEATURE':36} {'N':>3} {'ELEV':>4} {'HIGH':>4} "
        f"{'RUN':>3} {'MAX_PSI':>8} {'MAX_MED':>8} "
        f"{'LATEST':>16} {'PATTERN':>20}"
    )
    print("-" * 128)
    for report in sorted(reports, key=lambda item: item.ticker):
        if report.numeric_persistence.empty:
            print(
                f"{report.ticker:10} {'-':36} {'0':>3} {'0':>4} {'0':>4} "
                f"{'0':>3} {'n/a':>8} {'n/a':>8} "
                f"{'INSUFFICIENT':>16} {'INSUFFICIENT':>20}"
            )
            continue
        for row in report.numeric_persistence.itertuples(index=False):
            print(
                f"{row.ticker:10} {row.feature:36} "
                f"{row.transitions:3d} {row.elevated_transitions:4d} "
                f"{row.high_transitions:4d} {row.longest_elevated_run:3d} "
                f"{row.max_psi:8.3f} {row.max_median_shift_iqr:8.3f} "
                f"{row.latest_shift:>16} {row.pattern:>20}"
            )

    print("\nPERSISTENZA DRIFT STATI CATEGORICI")
    print(
        f"{'TICKER':10} {'FEATURE':20} {'N':>3} {'ELEV':>4} {'HIGH':>4} "
        f"{'RUN':>3} {'MAX_TVD':>8} {'LATEST':>16} {'PATTERN':>20}"
    )
    print("-" * 96)
    for report in sorted(reports, key=lambda item: item.ticker):
        if report.state_persistence.empty:
            print(
                f"{report.ticker:10} {'-':20} {'0':>3} {'0':>4} {'0':>4} "
                f"{'0':>3} {'n/a':>8} {'INSUFFICIENT':>16} "
                f"{'INSUFFICIENT':>20}"
            )
            continue
        for row in report.state_persistence.itertuples(index=False):
            print(
                f"{row.ticker:10} {row.feature:20} "
                f"{row.transitions:3d} {row.elevated_transitions:4d} "
                f"{row.high_transitions:4d} {row.longest_elevated_run:3d} "
                f"{row.max_total_variation:8.3f} {row.latest_shift:>16} "
                f"{row.pattern:>20}"
            )

    print("\nMEDIANE PER FASE DI SESSIONE")
    print(f"{'TICKER':10} {'FEATURE':36} {'OPEN':>12} {'MID':>12} {'CLOSE':>12}")
    print("-" * 86)
    for report in sorted(reports, key=lambda item: item.ticker):
        if report.phase_summary.empty:
            continue
        pivot = report.phase_summary.pivot(
            index="feature",
            columns="phase",
            values="median",
        )
        for feature, row in pivot.sort_index().iterrows():
            print(
                f"{report.ticker:10} {feature:36} "
                f"{row.get('OPEN', math.nan):12.6f} "
                f"{row.get('MID_SESSION', math.nan):12.6f} "
                f"{row.get('CLOSE', math.nan):12.6f}"
            )

    print("\nCONSENSO DESCRITTIVO CROSS-ASSET PER FASE")
    print(
        f"{'FEATURE':36} {'PHASE':12} {'ASSET':>5} {'VALID':>5} "
        f"{'PERSIST':>7} {'P_HIGH':>7} {'LATEST':>6} {'L_HIGH':>6}"
    )
    print("-" * 94)
    consensus = summarize_cross_asset_phase_consensus(reports)
    if consensus.empty:
        print("- Evidenza insufficiente per il confronto cross-asset.")
    else:
        for row in consensus.itertuples(index=False):
            print(
                f"{row.feature:36} {row.phase:12} {row.assets:5d} "
                f"{row.sufficient_assets:5d} {row.persistent_assets:7d} "
                f"{row.persistent_high_assets:7d} "
                f"{row.latest_elevated_assets:6d} "
                f"{row.latest_high_assets:6d}"
            )
    print("Gli asset sono correlati: i conteggi non sono osservazioni indipendenti.")

    print("\nRIDONDANZA ELEVATA")
    found_redundancy = False
    for report in sorted(reports, key=lambda item: item.ticker):
        redundant = report.redundancy.loc[
            report.redundancy["relation"] == "HIGH_REDUNDANCY"
        ]
        for row in redundant.itertuples(index=False):
            found_redundancy = True
            print(
                f"- {row.ticker}: {row.left_feature} × {row.right_feature} "
                f"Spearman={row.spearman:.3f}"
            )
    if not found_redundancy:
        print("- Nessuna coppia supera la soglia dichiarata nel campione disponibile.")

    if errors:
        print("\nERRORI ISOLATI")
        for ticker, error in sorted(errors.items()):
            print(f"- {ticker}: {error}")
    print("\nQuesti esiti descrivono i dati e non approvano una strategia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
