"""Scanner da riga di comando della fondazione adattiva paper-only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adaptive.journal import SQLiteDecisionJournal
from adaptive.models import CycleStatus, PortfolioState
from adaptive.orchestrator import AdaptiveTradingSystem
from data_engine.pipeline import MarketDataPipeline
from market.research_universe import RESEARCH_UNIVERSE


STATUS_ORDER = {
    CycleStatus.PAPER_APPROVED: 0,
    CycleStatus.EXECUTION_REJECTED: 1,
    CycleStatus.RISK_REJECTED: 2,
    CycleStatus.NO_TRADE: 3,
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Scanner multi-mercato adattivo. Analizza candidati e li registra, "
            "ma non invia ordini."
        )
    )
    parser.add_argument("--period", default="1y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(RESEARCH_UNIVERSE),
    )
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--peak-equity", type=float, default=None)
    parser.add_argument("--gross-exposure", type=float, default=0.0)
    parser.add_argument(
        "--journal",
        default=str(Path(".tradingai") / "adaptive_decisions.db"),
        help="Database SQLite dell'audit trail; usare :memory: per non salvarlo.",
    )
    return parser.parse_args(argv)


def infer_asset_class(ticker: str) -> str:
    normalized = str(ticker).upper().strip()
    if normalized in RESEARCH_UNIVERSE:
        return RESEARCH_UNIVERSE[normalized]
    if normalized.endswith("-USD"):
        return "CRYPTO"
    if normalized.endswith("=X"):
        return "FX"
    return "EQUITY"


def load_markets(pipeline, tickers, period, interval):
    markets = {}
    errors = {}
    total = len(tickers)
    for index, raw_ticker in enumerate(tickers, start=1):
        ticker = str(raw_ticker).upper().strip()
        print(f"[{index}/{total}] Scaricamento {ticker} {period} {interval}")
        try:
            data = pipeline.get_historical_data(
                ticker=ticker,
                period=period,
                interval=interval,
            )
            if data is None or data.empty:
                raise ValueError("Nessun dato disponibile.")
            markets[ticker] = data
            print(f"    {len(data)} candele")
        except Exception as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
            print(f"    ERRORE: {errors[ticker]}")
    return markets, errors


def print_results(result, download_errors) -> None:
    print()
    print("TRADINGAI ADAPTIVE SCAN — PAPER ONLY")
    print("Nessun ordine è stato inviato a un broker.")
    print()
    print(
        f"{'TICKER':12} {'STATUS':20} {'REGIME':20} {'DIR':7} "
        f"{'CONF':>8} {'TARGET':>9} {'ORIZZ':>6} {'EDGE NET/H':>11}"
    )
    print("-" * 104)

    ordered = sorted(
        result.cycles.values(),
        key=lambda cycle: (
            STATUS_ORDER[cycle.status],
            -cycle.execution_decision.net_edge_bps,
            cycle.snapshot.ticker,
        ),
    )
    for cycle in ordered:
        allocated_target = result.allocation.target_weights.get(
            cycle.snapshot.ticker
        )
        display_status = cycle.status.value
        if (
            cycle.status is CycleStatus.PAPER_APPROVED
            and allocated_target is None
        ):
            display_status = "ALLOCATION_REJECTED"
        print(
            f"{cycle.snapshot.ticker:12} "
            f"{display_status:20} "
            f"{cycle.regime.primary.value:20} "
            f"{cycle.meta_decision.direction.value:7} "
            f"{cycle.meta_decision.confidence:8.2%} "
            f"{(allocated_target or 0.0):9.2%} "
            f"{cycle.meta_decision.horizon_bars:5d}b "
            f"{cycle.execution_decision.net_edge_bps:8.2f}bp"
        )

    all_errors = {**download_errors, **result.errors}
    if all_errors:
        print()
        print("ERRORI ISOLATI")
        for ticker, error in sorted(all_errors.items()):
            print(f"- {ticker}: {error}")

    counts = {
        status: sum(cycle.status is status for cycle in result.cycles.values())
        for status in CycleStatus
    }
    print()
    print(
        "Totali: "
        + ", ".join(f"{status.value}={counts[status]}" for status in CycleStatus)
    )
    print(
        "Portafoglio proposto: "
        f"gross={result.allocation.projected_gross_exposure:.2%}, "
        f"net={result.allocation.projected_net_exposure:.2%}, "
        f"target={len(result.allocation.target_weights)}"
    )


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    if arguments.equity <= 0.0:
        raise ValueError("--equity deve essere positivo.")
    peak_equity = (
        arguments.equity
        if arguments.peak_equity is None
        else arguments.peak_equity
    )
    if peak_equity <= 0.0:
        raise ValueError("--peak-equity deve essere positivo.")

    tickers = tuple(
        dict.fromkeys(str(item).upper().strip() for item in arguments.tickers)
    )
    tickers = tuple(ticker for ticker in tickers if ticker)
    if not tickers:
        raise ValueError("La lista dei ticker è vuota.")

    markets, download_errors = load_markets(
        MarketDataPipeline(),
        tickers,
        arguments.period,
        arguments.interval,
    )
    portfolio = PortfolioState(
        equity=arguments.equity,
        peak_equity=peak_equity,
        gross_exposure=arguments.gross_exposure,
    )
    system = AdaptiveTradingSystem(
        journal=SQLiteDecisionJournal(arguments.journal),
    )
    result = system.analyze_universe(
        markets,
        portfolio,
        asset_classes={ticker: infer_asset_class(ticker) for ticker in markets},
    )
    print_results(result, download_errors)
    return 0 if result.cycles else 1


if __name__ == "__main__":
    sys.exit(main())
