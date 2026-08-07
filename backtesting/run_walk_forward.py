import argparse
import math
import sys

from backtesting.walk_forward import (
    WalkForwardValidator
)
from data_engine.pipeline import (
    MarketDataPipeline
)
from strategies.trend_following import (
    TrendFollowingStrategy
)


def format_value(
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


def build_strategy(
    strategy_name
):
    normalized_name = (
        strategy_name.lower().strip()
    )

    if normalized_name == "legacy":
        return None

    if normalized_name == "trend":
        return TrendFollowingStrategy()

    raise ValueError(
        f"Strategia sconosciuta: {strategy_name}"
    )


def print_fold_results(
    folds
):
    print()
    print("=" * 110)
    print(
        "RISULTATI DELLE FINESTRE WALK-FORWARD"
    )
    print("=" * 110)

    print(
        f"{'FOLD':>5} "
        f"{'INIZIO TEST':20} "
        f"{'FINE TEST':20} "
        f"{'RETURN':>10} "
        f"{'P/L':>10} "
        f"{'TRADES':>8} "
        f"{'WIN RATE':>10} "
        f"{'PF':>8} "
        f"{'DRAWDOWN':>11}"
    )

    print("-" * 110)

    for fold in folds:
        metrics = fold["metrics"]

        test_start = str(
            fold["test_start"]
        )[:19]

        test_end = str(
            fold["test_end"]
        )[:19]

        print(
            f"{fold['fold']:5} "
            f"{test_start:20} "
            f"{test_end:20} "
            f"{metrics['total_return_percent']:9.2f}% "
            f"{metrics['net_profit']:10.2f} "
            f"{metrics['total_trades']:8} "
            f"{metrics['win_rate_percent']:9.2f}% "
            f"{format_value(metrics['profit_factor']):>8} "
            f"{metrics['max_drawdown_percent']:10.2f}%"
        )


def print_summary(
    result
):
    summary = result["summary"]
    settings = result["settings"]

    print()
    print("=" * 70)
    print(
        "TRADING AI - WALK-FORWARD REPORT"
    )
    print("=" * 70)

    print(
        f"Asset:                         "
        f"{result['ticker']}"
    )

    print(
        f"Strategia:                     "
        f"{result['strategy']}"
    )

    print(
        f"Train bars:                    "
        f"{settings['train_bars']}"
    )

    print(
        f"Test bars:                     "
        f"{settings['test_bars']}"
    )

    print(
        f"Step bars:                     "
        f"{settings['step_bars']}"
    )

    print("-" * 70)

    print(
        f"Finestre totali:               "
        f"{summary['total_folds']}"
    )

    print(
        f"Finestre profittevoli:         "
        f"{summary['profitable_folds']}"
    )

    print(
        f"Finestre in perdita:           "
        f"{summary['losing_folds']}"
    )

    print(
        f"Percentuale finestre positive: "
        f"{summary['profitable_fold_percent']:.2f}%"
    )

    print(
        f"Trade complessivi:             "
        f"{summary['total_trades']}"
    )

    print(
        f"Profitto netto complessivo:    "
        f"{summary['total_net_profit']:.2f}"
    )

    print(
        f"Rendimento medio finestra:     "
        f"{summary['average_fold_return_percent']:.2f}%"
    )

    print(
        f"Rendimento mediano finestra:   "
        f"{summary['median_fold_return_percent']:.2f}%"
    )

    print(
        f"Migliore finestra:             "
        f"{summary['best_fold_return_percent']:.2f}%"
    )

    print(
        f"Peggiore finestra:             "
        f"{summary['worst_fold_return_percent']:.2f}%"
    )

    print(
        f"Walk-forward efficiency:       "
        f"{format_value(summary['walk_forward_efficiency'])}"
    )

    print("-" * 70)

    robustness = (
        "ROBUSTA"
        if summary["is_robust"]
        else "NON ROBUSTA"
    )

    print(
        f"Valutazione finale:            "
        f"{robustness}"
    )

    print("=" * 70)

    if summary["total_trades"] < 30:
        print(
            "ATTENZIONE: numero di trade troppo basso "
            "per una valutazione statistica affidabile."
        )

    if (
        summary[
            "profitable_fold_percent"
        ] < 60
    ):
        print(
            "ATTENZIONE: meno del 60% delle finestre "
            "fuori campione è profittevole."
        )

    if summary["total_net_profit"] <= 0:
        print(
            "ATTENZIONE: il risultato complessivo "
            "fuori campione non è positivo."
        )


def run_walk_forward(
    ticker,
    strategy_name,
    period,
    interval,
    train_bars,
    test_bars,
    step_bars,
    capital,
    max_position_percent,
    commission_percent,
    slippage_percent,
    max_holding_bars,
    cooldown_bars,
    max_trades_per_day
):
    pipeline = MarketDataPipeline()

    strategy = build_strategy(
        strategy_name
    )

    print(
        f"Scaricamento dati per {ticker}: "
        f"period={period}, interval={interval}"
    )

    print(
        f"Strategia richiesta: "
        f"{strategy_name}"
    )

    data = pipeline.get_historical_data(
        ticker=ticker,
        period=period,
        interval=interval
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"Nessun dato disponibile per {ticker}."
        )

    print(
        f"Candele disponibili: {len(data)}"
    )

    minimum_required = (
        train_bars +
        test_bars
    )

    if len(data) < minimum_required:
        raise ValueError(
            "Storico insufficiente. "
            f"Richieste almeno {minimum_required} candele, "
            f"ricevute {len(data)}."
        )

    validator = WalkForwardValidator(
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        initial_capital=capital,
        max_position_percent=(
            max_position_percent
        ),
        minimum_history=30,
        commission_percent=(
            commission_percent
        ),
        slippage_percent=(
            slippage_percent
        ),
        max_holding_bars=(
            max_holding_bars
        ),
        cooldown_bars=(
            cooldown_bars
        ),
        max_trades_per_day=(
            max_trades_per_day
        ),
        strategy=strategy
    )

    result = validator.run(
        ticker=ticker,
        data=data
    )

    print_fold_results(
        result["folds"]
    )

    print_summary(
        result
    )

    return result


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Esegue una validazione walk-forward "
            "di una strategia TradingAI."
        )
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL"
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "legacy",
            "trend"
        ],
        default="legacy"
    )

    parser.add_argument(
        "--period",
        default="60d"
    )

    parser.add_argument(
        "--interval",
        default="5m"
    )

    parser.add_argument(
        "--train-bars",
        type=int,
        default=1500
    )

    parser.add_argument(
        "--test-bars",
        type=int,
        default=500
    )

    parser.add_argument(
        "--step-bars",
        type=int,
        default=500
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
        "--commission",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005
    )

    parser.add_argument(
        "--max-holding-bars",
        type=int,
        default=78
    )

    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=12
    )

    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=3
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        run_walk_forward(
            ticker=args.ticker.upper(),
            strategy_name=args.strategy,
            period=args.period,
            interval=args.interval,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            capital=args.capital,
            max_position_percent=(
                args.position
            ),
            commission_percent=(
                args.commission
            ),
            slippage_percent=(
                args.slippage
            ),
            max_holding_bars=(
                args.max_holding_bars
            ),
            cooldown_bars=(
                args.cooldown_bars
            ),
            max_trades_per_day=(
                args.max_trades_per_day
            )
        )

    except Exception as error:
        print()
        print(f"ERRORE: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()