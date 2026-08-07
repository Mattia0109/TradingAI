"""CLI per il backtest causale della pipeline adaptive paper-only."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from adaptive.backtester import AdaptiveBacktestConfig, AdaptiveBacktester
from adaptive.run_adaptive_scan import infer_asset_class, load_markets
from adaptive.validation import evaluate_challenger
from data_engine.pipeline import MarketDataPipeline
from market.research_universe import RESEARCH_UNIVERSE


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Backtest causale multi-asset della pipeline adaptive.")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--tickers", nargs="+", default=list(RESEARCH_UNIVERSE))
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--diagnostic-min-signals", type=int, default=20)
    parser.add_argument(
        "--no-challenger-comparison",
        action="store_true",
        help="Salta il Challenger regime-aware walk-forward.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    tickers = tuple(dict.fromkeys(str(value).upper().strip() for value in args.tickers if str(value).strip()))
    markets, errors = load_markets(MarketDataPipeline(), tickers, args.period, args.interval)
    if errors:
        print("\nERRORI DOWNLOAD")
        for ticker, error in sorted(errors.items()):
            print(f"- {ticker}: {error}")
    config = AdaptiveBacktestConfig(
        initial_equity=args.capital,
        rebalance_every=args.rebalance_every,
        cost_bps_per_turnover=args.cost_bps,
    )
    asset_classes = {ticker: infer_asset_class(ticker) for ticker in markets}
    result = AdaptiveBacktester(config=config).run(
        markets, asset_classes=asset_classes
    )
    challenger_result = None
    promotion_gate = None
    if not args.no_challenger_comparison:
        challenger_result = AdaptiveBacktester(
            config=replace(config, online_regime_learning=True)
        ).run(markets, asset_classes=asset_classes)
        promotion_gate = evaluate_challenger(
            result.daily_returns,
            challenger_result.daily_returns,
            annualization_factor=config.annualization_factor,
        )
    print("\nTRADINGAI CHAMPION STATICO — PAPER ONLY, CAUSALE t → t+1")
    print(f"Periodo valutato:       {result.daily_returns.index[0]} → {result.daily_returns.index[-1]}")
    print(f"Rendimento totale:      {result.total_return:10.2%}")
    print(f"CAGR:                   {result.cagr:10.2%}")
    print(f"Volatilità annuale:     {result.annualized_volatility:10.2%}")
    print(f"Sharpe:                 {result.sharpe:10.2f}")
    print(f"Max drawdown:           {result.max_drawdown:10.2%}")
    print(f"Turnover totale:        {result.total_turnover:10.2f}")
    print(f"Costo cumulato stimato: {result.total_cost_return:10.2%}")
    print(f"Decisioni simulate:     {result.decision_count:10d}")
    print(f"Errori analisi isolati: {result.analysis_error_count:10d}")
    print(f"Esposizione gross media:{result.average_gross_exposure:10.2%}")
    print(f"Esposizione net media:  {result.average_net_exposure:10.2%}")
    print(f"Tempo investito:        {result.invested_fraction:10.2%}")
    if challenger_result is not None:
        print("\nCONFRONTO CHALLENGER REGIME-AWARE — STESSI DATI E COSTI")
        print(f"Rendimento Champion:    {result.total_return:10.2%}")
        print(f"Rendimento Challenger:  {challenger_result.total_return:10.2%}")
        print(f"Differenza:             {challenger_result.total_return - result.total_return:10.2%}")
        print(f"Sharpe Champion:        {result.sharpe:10.2f}")
        print(f"Sharpe Challenger:      {challenger_result.sharpe:10.2f}")
        print(f"Drawdown Champion:      {result.max_drawdown:10.2%}")
        print(f"Drawdown Challenger:    {challenger_result.max_drawdown:10.2%}")
        print(f"Eventi quarantena:      {challenger_result.quarantine_events:10d}")
        print("\nPROMOTION GATE TEMPORALE")
        for fold, row in promotion_gate.fold_results.iterrows():
            outcome = "WIN" if bool(row["challenger_wins"]) else "LOSS"
            print(
                f"Fold {fold}: {row['start']} → {row['end']} | "
                f"ret C={row['champion_return']:.2%} R={row['challenger_return']:.2%} | "
                f"Sharpe C={row['champion_sharpe']:.2f} R={row['challenger_sharpe']:.2f} | {outcome}"
            )
        verdict = (
            "ELIGIBLE_FOR_REVIEW"
            if promotion_gate.eligible_for_review
            else "REJECT_CHALLENGER"
        )
        print(f"Esito: {verdict}")
        for reason in promotion_gate.reasons:
            print(f"- {reason}")
    print("\nBENCHMARK EQUAL WEIGHT — STESSO PERIODO")
    print(f"Rendimento totale:      {result.equal_weight_total_return:10.2%}")
    print(f"CAGR:                   {result.equal_weight_cagr:10.2%}")
    print(f"Sharpe:                 {result.equal_weight_sharpe:10.2f}")
    print(f"Max drawdown:           {result.equal_weight_max_drawdown:10.2%}")
    print("\nDIAGNOSTICA PREDITTIVA PER MODELLO")
    print(f"{'MODELLO':28} {'N':>6} {'HIT':>8} {'SIGNED':>10} {'NET':>10} {'MAE':>10} {'CORR':>8}")
    for model, row in result.strategy_diagnostics.iterrows():
        print(
            f"{str(model):28} {int(row['signals']):6d} "
            f"{row['hit_rate']:8.2%} {row['mean_signed_return']:10.3%} "
            f"{row['mean_net_return']:10.3%} {row['mean_absolute_error']:10.3%} "
            f"{row['forecast_correlation']:8.3f}"
        )
    print("\nDIAGNOSTICA MODELLO × REGIME")
    print(f"{'MODELLO':25} {'REGIME':20} {'N':>6} {'HIT':>8} {'NET':>10} {'CORR':>8}")
    for (model, regime), row in result.strategy_regime_diagnostics.iterrows():
        if int(row["signals"]) < args.diagnostic_min_signals:
            continue
        print(
            f"{str(model):25} {str(regime):20} {int(row['signals']):6d} "
            f"{row['hit_rate']:8.2%} {row['mean_net_return']:10.3%} "
            f"{row['forecast_correlation']:8.3f}"
        )
    if challenger_result is not None and not challenger_result.learning_states.empty:
        print("\nSTATO CHALLENGER PER MODELLO × REGIME")
        print(f"{'MODELLO':28} {'REGIME':20} {'N':>6} {'UTILITY':>9} {'PESO':>8} {'STATO':>12}")
        for _, row in challenger_result.learning_states.iterrows():
            status = (
                "QUARANTENA"
                if bool(row["quarantined"])
                else (
                    "POTENZIATO"
                    if float(row["weight"]) > 1.01
                    else (
                        "RIDOTTO"
                        if float(row["weight"]) < 0.99
                        else "NEUTRO"
                    )
                )
            )
            print(
                f"{str(row['strategy_id']):28} {str(row['regime']):20} "
                f"{int(row['observations']):6d} {row['ewma_utility']:9.3f} "
                f"{row['weight']:8.3f} {status:>12}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
