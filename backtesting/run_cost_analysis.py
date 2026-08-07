import argparse
import sys

from backtesting.cost_scenarios import (
    CostScenarioAnalyzer
)
from data_engine.pipeline import (
    MarketDataPipeline
)
from strategies.trend_following import (
    TrendFollowingStrategy
)


def build_strategy(
    strategy_name
):
    normalized = (
        strategy_name
        .lower()
        .strip()
    )

    if normalized == "legacy":
        return None

    if normalized == "trend":
        return TrendFollowingStrategy()

    raise ValueError(
        f"Strategia sconosciuta: "
        f"{strategy_name}"
    )


def print_scenarios(
    result
):
    print()
    print("=" * 116)
    print(
        "TRADING AI - COST SCENARIO ANALYSIS"
    )
    print("=" * 116)

    print(
        f"Asset: {result['ticker']}"
    )

    print(
        f"Strategia: {result['strategy']}"
    )

    print("-" * 116)

    print(
        f"{'SCENARIO':15} "
        f"{'ROUND TRIP':>12} "
        f"{'FOLDS':>7} "
        f"{'POSITIVE':>10} "
        f"{'TRADES':>8} "
        f"{'NET PROFIT':>13} "
        f"{'WFE':>8} "
        f"{'ROBUST':>9}"
    )

    print("-" * 116)

    for (
        scenario_name,
        scenario
    ) in result["scenarios"].items():
        summary = scenario[
            "summary"
        ]

        robust = (
            "YES"
            if summary["is_robust"]
            else "NO"
        )

        print(
            f"{scenario_name:15} "
            f"{scenario['round_trip_cost_percent']:11.3f}% "
            f"{summary['total_folds']:7} "
            f"{summary['profitable_fold_percent']:9.2f}% "
            f"{summary['total_trades']:8} "
            f"{summary['total_net_profit']:13.2f} "
            f"{summary['walk_forward_efficiency']:8.2f} "
            f"{robust:>9}"
        )

    print("-" * 116)


def print_comparison(
    result
):
    comparison = result[
        "comparison"
    ]

    print()
    print("=" * 76)
    print("DIAGNOSI DELLA STRATEGIA")
    print("=" * 76)

    print(
        f"Profitto senza costi:           "
        f"{comparison['zero_cost_net_profit']:.2f}"
    )

    print(
        f"Profitto con costi realistici: "
        f"{comparison['realistic_net_profit']:.2f}"
    )

    print(
        f"Profitto con costi stressati:  "
        f"{comparison['stress_net_profit']:.2f}"
    )

    print(
        f"Impatto costi realistici:      "
        f"{comparison['realistic_cost_drag']:.2f}"
    )

    print(
        f"Impatto costi stressati:       "
        f"{comparison['stress_cost_drag']:.2f}"
    )

    print(
        f"Ritenzione profitto realistico:"
        f" "
        f"{comparison['realistic_profit_retention_percent']:.2f}%"
    )

    print(
        f"Ritenzione profitto stress:    "
        f"{comparison['stress_profit_retention_percent']:.2f}%"
    )

    print("-" * 76)

    print(
        f"Classificazione:               "
        f"{comparison['classification']}"
    )

    print(
        f"Diagnosi:                      "
        f"{comparison['explanation']}"
    )

    print(
        f"Continua ricerca:              "
        f"{comparison['accepted_for_research']}"
    )

    print(
        f"Candidata al paper trading:    "
        f"{comparison['accepted_for_paper_trading']}"
    )

    print("=" * 76)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Analizza una strategia con "
            "differenti scenari di costi."
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
        default="trend"
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
        strategy = build_strategy(
            args.strategy
        )

        pipeline = MarketDataPipeline()

        print(
            f"Scaricamento dati per "
            f"{args.ticker.upper()}: "
            f"period={args.period}, "
            f"interval={args.interval}"
        )

        data = pipeline.get_historical_data(
            ticker=args.ticker.upper(),
            period=args.period,
            interval=args.interval
        )

        if data is None or data.empty:
            raise RuntimeError(
                "Nessun dato disponibile."
            )

        print(
            f"Candele disponibili: "
            f"{len(data)}"
        )

        analyzer = CostScenarioAnalyzer(
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
            initial_capital=args.capital,
            max_position_percent=(
                args.position
            ),
            minimum_history=30,
            max_holding_bars=(
                args.max_holding_bars
            ),
            cooldown_bars=(
                args.cooldown_bars
            ),
            max_trades_per_day=(
                args.max_trades_per_day
            ),
            strategy=strategy
        )

        result = analyzer.run(
            ticker=args.ticker.upper(),
            data=data
        )

        print_scenarios(
            result
        )

        print_comparison(
            result
        )

    except Exception as error:
        print()
        print(
            f"ERRORE: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()