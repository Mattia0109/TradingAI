"""CLI del Long Burst Momentum Challenger, esclusivamente ricerca locale."""

from __future__ import annotations

import argparse
import math
import sys

from adaptive.backtester import AdaptiveBacktestConfig, AdaptiveBacktester
from adaptive.long_burst import LongBurstConfig, LongBurstSignalEngine
from adaptive.long_burst_backtester import (
    LongBurstBacktestConfig,
    LongBurstBacktester,
)
from adaptive.run_adaptive_scan import load_markets
from adaptive.validation import evaluate_challenger
from data_engine.pipeline import MarketDataPipeline
from market.research_universe import RESEARCH_UNIVERSE


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Backtest causale del Long Burst Momentum Challenger. "
            "Produce soltanto ricerca e report; non invia ordini."
        )
    )
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--tickers", nargs="+", default=list(RESEARCH_UNIVERSE))
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--max-holding-bars", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--diagnostic-min-signals", type=int, default=20)
    parser.add_argument(
        "--skip-champion-comparison",
        action="store_true",
        help="Salta il confronto con il Champion statico esistente.",
    )
    return parser.parse_args(argv)


def _print_group_diagnostics(
    title: str,
    index_label: str,
    diagnostics,
    minimum_signals: int,
) -> None:
    if diagnostics.empty:
        return
    visible = diagnostics.loc[diagnostics["signals"] >= minimum_signals]
    if visible.empty:
        return
    print(f"\n{title}")
    print(
        f"{index_label:22} {'N_IND':>6} {'N_ALL':>6} {'HIT':>8} "
        f"{'NET':>10} {'CONF':>8} {'CORR':>8}"
    )
    for group_name, row in visible.iterrows():
        print(
            f"{str(group_name):22} {int(row['signals']):6d} "
            f"{int(row['all_signals']):6d} {row['hit_rate']:8.2%} "
            f"{row['mean_net_return']:10.3%} "
            f"{row['mean_confidence']:8.2%} "
            f"{row['forecast_correlation']:8.3f}"
        )


def _print_primary_result(result, diagnostic_min_signals: int) -> None:
    print("\nLONG BURST MOMENTUM V1 — CHALLENGER RESEARCH-ONLY")
    print("Segnali ammessi: LONG / NO_TRADE. Leva: assente. Broker: assente.")
    print(
        f"Periodo valutato:       {result.daily_returns.index[0]} "
        f"→ {result.daily_returns.index[-1]}"
    )
    print(f"Rendimento totale:      {result.total_return:10.2%}")
    print(f"CAGR:                   {result.cagr:10.2%}")
    print(f"Volatilità annuale:     {result.annualized_volatility:10.2%}")
    print(f"Sharpe:                 {result.sharpe:10.2f}")
    print(f"Max drawdown:           {result.max_drawdown:10.2%}")
    print(f"Segnali maturati totali:{result.signal_count:10d}")
    print(f"Segnali indipendenti:   {result.independent_signal_count:10d}")
    print(f"Operazioni simulate:    {result.trade_count:10d}")
    print(f"Win rate:               {result.win_rate:10.2%}")
    print(f"Expectancy per trade:   {result.expectancy:10.3%}")
    profit_factor = (
        "INF"
        if math.isinf(result.profit_factor)
        else f"{result.profit_factor:.2f}"
    )
    print(f"Profit factor:          {profit_factor:>10}")
    print(f"Durata media:           {result.average_duration_bars:9.2f}b")
    print(f"Durata massima:         {result.maximum_duration_bars:9d}b")
    print(f"Tempo investito:        {result.invested_fraction:10.2%}")
    print(f"Esposizione gross media:{result.average_gross_exposure:10.2%}")

    print("\nBENCHMARK EQUAL WEIGHT — STESSO PERIODO")
    print(f"Rendimento totale:      {result.benchmark_total_return:10.2%}")
    print(f"CAGR:                   {result.benchmark_cagr:10.2%}")
    print(f"Sharpe:                 {result.benchmark_sharpe:10.2f}")
    print(f"Max drawdown:           {result.benchmark_max_drawdown:10.2%}")

    if not result.trades.empty:
        print("\nMOTIVI DI USCITA")
        for reason, count in result.trades["exit_reason"].value_counts().items():
            print(f"{reason:20} {int(count):6d}")
        signal_decay = int(
            (result.trades["exit_reason"] == "SIGNAL_DECAY").sum()
        )
        if signal_decay / len(result.trades) >= 0.50:
            print("\nAUDIT USCITE")
            print(
                "- Oltre metà delle simulazioni termina perché il gate "
                "LONG torna NO_TRADE."
            )
            print(
                "- NO_TRADE non equivale a previsione ribassista: questa "
                "uscita va trattata come ipotesi da validare, non come alpha."
            )

    print(
        "\nGli outcome diagnostici partono dall'apertura successiva e le "
        "metriche usano N_IND, il sottocampione non sovrapposto."
    )
    _print_group_diagnostics(
        "DIAGNOSTICA SEGNALE × REGIME",
        "REGIME",
        result.regime_diagnostics,
        diagnostic_min_signals,
    )
    _print_group_diagnostics(
        "DIAGNOSTICA SEGNALE × TICKER",
        "TICKER",
        result.ticker_diagnostics,
        diagnostic_min_signals,
    )
    _print_group_diagnostics(
        "DIAGNOSTICA SEGNALE × ENTRY PATH",
        "ENTRY_PATH",
        result.entry_path_diagnostics,
        diagnostic_min_signals,
    )
    _print_group_diagnostics(
        "DIAGNOSTICA SEGNALE × ORIZZONTE",
        "BARRE",
        result.horizon_diagnostics,
        diagnostic_min_signals,
    )


def _print_promotion_gate(champion, challenger, annualization_factor: int) -> None:
    gate = evaluate_challenger(
        champion.daily_returns,
        challenger.daily_returns,
        annualization_factor=annualization_factor,
    )
    print("\nCONFRONTO CHAMPION STATICO / LONG BURST CHALLENGER")
    for fold, row in gate.fold_results.iterrows():
        outcome = "WIN" if bool(row["challenger_wins"]) else "LOSS"
        print(
            f"Fold {fold}: {row['start']} → {row['end']} | "
            f"ret C={row['champion_return']:.2%} B={row['challenger_return']:.2%} | "
            f"Sharpe C={row['champion_sharpe']:.2f} "
            f"B={row['challenger_sharpe']:.2f} | {outcome}"
        )
    verdict = (
        "ELIGIBLE_FOR_REVIEW"
        if gate.eligible_for_review
        else "REJECT_CHALLENGER"
    )
    print(f"Esito: {verdict}")
    for reason in gate.reasons:
        print(f"- {reason}")


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    tickers = tuple(
        dict.fromkeys(
            str(item).upper().strip()
            for item in arguments.tickers
            if str(item).strip()
        )
    )
    markets, download_errors = load_markets(
        MarketDataPipeline(),
        tickers,
        arguments.period,
        arguments.interval,
    )
    if not markets:
        raise ValueError("Nessun mercato disponibile per il backtest.")

    signal_config = LongBurstConfig(
        forecast_horizon_bars=arguments.horizon_bars,
        round_trip_cost_bps=arguments.cost_bps,
    )
    backtest_config = LongBurstBacktestConfig(
        initial_equity=arguments.capital,
        maximum_holding_bars=arguments.max_holding_bars,
        round_trip_cost_bps=arguments.cost_bps,
    )
    result = LongBurstBacktester(
        LongBurstSignalEngine(signal_config),
        backtest_config,
    ).run(markets)
    _print_primary_result(result, arguments.diagnostic_min_signals)

    all_errors = {**download_errors, **result.errors}
    if all_errors:
        print("\nERRORI ISOLATI")
        for ticker, error in sorted(all_errors.items()):
            print(f"- {ticker}: {error}")

    if not arguments.skip_champion_comparison:
        champion = AdaptiveBacktester(
            config=AdaptiveBacktestConfig(
                initial_equity=arguments.capital,
                rebalance_every=5,
                cost_bps_per_turnover=arguments.cost_bps,
            )
        ).run(markets)
        _print_promotion_gate(
            champion,
            result,
            backtest_config.annualization_factor,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
