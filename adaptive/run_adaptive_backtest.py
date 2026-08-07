"""CLI per il backtest causale della pipeline adaptive paper-only."""

from __future__ import annotations

import argparse
import sys

from adaptive.backtester import AdaptiveBacktestConfig, AdaptiveBacktester
from adaptive.run_adaptive_scan import infer_asset_class, load_markets
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
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_arguments(argv)
    tickers = tuple(dict.fromkeys(str(value).upper().strip() for value in args.tickers if str(value).strip()))
    markets, errors = load_markets(MarketDataPipeline(), tickers, args.period, args.interval)
    if errors:
        print("\nERRORI DOWNLOAD")
        for ticker, error in sorted(errors.items()):
            print(f"- {ticker}: {error}")
    result = AdaptiveBacktester(config=AdaptiveBacktestConfig(
        initial_equity=args.capital,
        rebalance_every=args.rebalance_every,
        cost_bps_per_turnover=args.cost_bps,
    )).run(markets, asset_classes={ticker: infer_asset_class(ticker) for ticker in markets})
    print("\nTRADINGAI ADAPTIVE BACKTEST — PAPER ONLY, CAUSALE t → t+1")
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
    print("\nBENCHMARK EQUAL WEIGHT — STESSO PERIODO")
    print(f"Rendimento totale:      {result.equal_weight_total_return:10.2%}")
    print(f"CAGR:                   {result.equal_weight_cagr:10.2%}")
    print(f"Sharpe:                 {result.equal_weight_sharpe:10.2f}")
    print(f"Max drawdown:           {result.equal_weight_max_drawdown:10.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
