"""CLI per la stabilità descrittiva delle feature intraday a 15 minuti."""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

from adaptive.intraday_audit import IntradayReadiness, IntradayResearchAuditor
from adaptive.intraday_features import IntradayReferenceFeatureEngine
from adaptive.intraday_stability import (
    IntradayFeatureStabilityAnalyzer,
    IntradayStabilityConfig,
    regular_session_frame,
)
from adaptive.run_adaptive_scan import load_markets
from data_engine.pipeline import MarketDataPipeline


DEFAULT_STABILITY_UNIVERSE = ("SPY", "QQQ", "IWM", "GLD", "TLT")
SHIFT_RANK = {
    "INSUFFICIENT": -1,
    "LOW_SHIFT": 0,
    "MODERATE_SHIFT": 1,
    "HIGH_SHIFT": 2,
}


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


def _worst_shift(values: pd.Series) -> str:
    labels = [str(value) for value in values if str(value) in SHIFT_RANK]
    return max(labels, key=SHIFT_RANK.get) if labels else "INSUFFICIENT"


def _maximum(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else math.nan


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
            reports.append(analyzer.analyze(ticker, features))
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
    if not reports:
        raise ValueError("Nessun asset ha superato il gate di stabilità.")

    print("\nTRADINGAI 15M FEATURE STABILITY — RESEARCH-ONLY")
    print("Nessun segnale, ordine, size, stop, leva, outcome futuro o P&L.")
    print("I blocchi sono fissi: nuovi dati non riassegnano le sessioni passate.")
    print("Le feature vengono calcolate dopo aver escluso extended-hours e weekend.")

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

    print("\nDRIFT FEATURE NUMERICHE")
    print(
        f"{'TICKER':10} {'FEATURE':27} {'MAX_PSI':>9} "
        f"{'MAX_SHIFT':>10} {'STATO':>16}"
    )
    print("-" * 78)
    for report in sorted(reports, key=lambda item: item.ticker):
        if report.numeric_drift.empty:
            print(
                f"{report.ticker:10} {'-':27} {'n/a':>9} "
                f"{'n/a':>10} {'INSUFFICIENT':>16}"
            )
            continue
        for feature, group in report.numeric_drift.groupby("feature", sort=True):
            psi = _maximum(group["psi"])
            median_shift = _maximum(group["median_shift_iqr"])
            print(
                f"{report.ticker:10} {feature:27} "
                f"{psi:9.3f} {median_shift:10.3f} {_worst_shift(group['shift']):>16}"
            )

    print("\nDRIFT STATI CATEGORICI")
    print(f"{'TICKER':10} {'FEATURE':20} {'MAX_TVD':>9} {'STATO':>16}")
    print("-" * 59)
    for report in sorted(reports, key=lambda item: item.ticker):
        if report.state_drift.empty:
            print(f"{report.ticker:10} {'-':20} {'n/a':>9} {'INSUFFICIENT':>16}")
            continue
        for feature, group in report.state_drift.groupby("feature", sort=True):
            tvd = _maximum(group["total_variation"])
            print(
                f"{report.ticker:10} {feature:20} {tvd:9.3f} "
                f"{_worst_shift(group['shift']):>16}"
            )

    print("\nMEDIANE PER FASE DI SESSIONE")
    print(f"{'TICKER':10} {'FEATURE':27} {'OPEN':>11} {'MID':>11} {'CLOSE':>11}")
    print("-" * 75)
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
                f"{report.ticker:10} {feature:27} "
                f"{row.get('OPEN', math.nan):11.4f} "
                f"{row.get('MID_SESSION', math.nan):11.4f} "
                f"{row.get('CLOSE', math.nan):11.4f}"
            )

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
