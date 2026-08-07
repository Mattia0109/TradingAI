import argparse
import math
import sys

from backtesting.backtester import Backtester
from data_engine.pipeline import MarketDataPipeline


def format_number(
    value,
    decimals=2
):
    if value is None:
        return "N/D"

    if (
        isinstance(value, float)
        and math.isinf(value)
    ):
        return "INF"

    return f"{value:.{decimals}f}"


def print_report(result):

    metrics = result["metrics"]
    trades = result["trades"]

    print()
    print("=" * 60)
    print("TRADING AI - BACKTEST REPORT")
    print("=" * 60)

    print(
        f"Asset:                 "
        f"{result['ticker']}"
    )

    print(
        f"Tipo:                  "
        f"{result['asset_type']}"
    )

    print(
        f"Capitale iniziale:     "
        f"{metrics['initial_capital']:.2f}"
    )

    print(
        f"Capitale finale:       "
        f"{metrics['final_capital']:.2f}"
    )

    print(
        f"Profitto netto:        "
        f"{metrics['net_profit']:.2f}"
    )

    print(
        f"Rendimento totale:     "
        f"{metrics['total_return_percent']:.2f}%"
    )

    print("-" * 60)

    print(
        f"Numero trade:          "
        f"{metrics['total_trades']}"
    )

    print(
        f"Trade vincenti:        "
        f"{metrics['winning_trades']}"
    )

    print(
        f"Trade perdenti:        "
        f"{metrics['losing_trades']}"
    )

    print(
        f"Win rate:              "
        f"{metrics['win_rate_percent']:.2f}%"
    )

    print(
        f"Profit factor:         "
        f"{format_number(metrics['profit_factor'])}"
    )

    print(
        f"Max drawdown:          "
        f"{metrics['max_drawdown_percent']:.2f}%"
    )

    print(
        f"Commissioni totali:    "
        f"{metrics.get('total_commissions', 0):.2f}"
    )

    print("=" * 60)

    if not trades:
        print()
        print("Nessuna operazione generata.")
        return

    print()
    print("ULTIME OPERAZIONI")
    print("-" * 60)

    for trade in trades[-10:]:
        print(
            f"{trade['ticker']:10} | "
            f"{trade['direction']:5} | "
            f"Entry: {trade['entry_price']:.4f} | "
            f"Exit: {trade['exit_price']:.4f} | "
            f"P/L: {trade['pnl']:.2f} | "
            f"{trade['exit_reason']}"
        )


def run_backtest(
    ticker,
    initial_capital,
    max_position_percent,
    period,
    interval
):
    pipeline = MarketDataPipeline()

    print(
        f"Scaricamento {ticker}: "
        f"period={period}, interval={interval}"
    )

    historical_data = (
        pipeline.get_historical_data(
            ticker=ticker,
            period=period,
            interval=interval
        )
    )

    if historical_data is None:
        raise RuntimeError(
            f"Nessun dato disponibile per {ticker}."
        )

    print(
        f"Candele scaricate: "
        f"{len(historical_data)}"
    )

    backtester = Backtester(
        initial_capital=initial_capital,
        max_position_percent=(
            max_position_percent
        ),
        minimum_history=30,
        commission_percent=0.001,
        slippage_percent=0.0005
    )

    result = backtester.run(
        ticker=ticker,
        data=historical_data
    )

    print_report(result)

    return result


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Esegue un backtest TradingAI."
        )
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0
    )

    parser.add_argument(
        "--position",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--period",
        default="6mo"
    )

    parser.add_argument(
        "--interval",
        default="1d"
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    try:
        run_backtest(
            ticker=args.ticker.upper(),
            initial_capital=args.capital,
            max_position_percent=(
                args.position
            ),
            period=args.period,
            interval=args.interval
        )

    except Exception as error:
        print()
        print(f"ERRORE: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()