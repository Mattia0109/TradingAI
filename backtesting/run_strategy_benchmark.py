import argparse
import sys

from backtesting.strategy_benchmark import (
    StrategyBenchmarkRunner
)
from data_engine.pipeline import (
    MarketDataPipeline
)
from strategies.trend_following import (
    TrendFollowingStrategy
)


QUICK_ASSETS = [
    "AAPL",
    "NVDA",
    "SPY",
    "BTC-USD",
    "EURUSD=X",
    "GC=F"
]


FULL_ASSETS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "VTI",
    "IWM",
    "BTC-USD",
    "ETH-USD",
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "GC=F",
    "CL=F"
]


TIMEFRAME_CONFIGS = {
    "5m": {
        "period": "60d",
        "train_bars": 1500,
        "test_bars": 500,
        "step_bars": 500
    },
    "15m": {
        "period": "60d",
        "train_bars": 600,
        "test_bars": 200,
        "step_bars": 200
    },
    "1h": {
        "period": "60d",
        "train_bars": 180,
        "test_bars": 60,
        "step_bars": 60
    },
    "1d": {
        "period": "5y",
        "train_bars": 500,
        "test_bars": 125,
        "step_bars": 125
    }
}


def build_strategies(
    requested_strategies
):
    strategies = {}

    for strategy_name in (
        requested_strategies
    ):
        if strategy_name == "legacy":
            strategies[
                "legacy_momentum"
            ] = None

        elif strategy_name == "trend":
            strategies[
                "trend_following"
            ] = TrendFollowingStrategy()

        else:
            raise ValueError(
                f"Strategia sconosciuta: "
                f"{strategy_name}"
            )

    return strategies


def build_market_cases(
    assets,
    timeframes
):
    market_cases = []

    for ticker in assets:
        for interval in timeframes:
            configuration = (
                TIMEFRAME_CONFIGS[
                    interval
                ]
            )

            market_cases.append(
                {
                    "ticker": ticker,
                    "period": configuration[
                        "period"
                    ],
                    "interval": interval,
                    "train_bars": configuration[
                        "train_bars"
                    ],
                    "test_bars": configuration[
                        "test_bars"
                    ],
                    "step_bars": configuration[
                        "step_bars"
                    ]
                }
            )

    return market_cases


def parse_margin_overrides(
    entries
):
    """
    Formato:

    CL=F:5000
    GC=F:12000
    """

    overrides = {}

    for entry in entries or []:
        if ":" not in entry:
            raise ValueError(
                "Formato margin override non valido: "
                f"{entry}. Usa TICKER:VALORE."
            )

        ticker, value = entry.rsplit(
            ":",
            1
        )

        normalized_ticker = (
            ticker
            .upper()
            .strip()
        )

        margin_value = float(
            value
        )

        if not normalized_ticker:
            raise ValueError(
                "Ticker vuoto nel margin override."
            )

        if margin_value <= 0:
            raise ValueError(
                "Il margine deve essere positivo."
            )

        overrides[
            normalized_ticker
        ] = margin_value

    return overrides


def print_case_ranking(
    result
):
    settings = result[
        "settings"
    ]

    print()
    print("=" * 155)
    print(
        "STRATEGY BENCHMARK - "
        "CLASSIFICA COMBINAZIONI"
    )
    print("=" * 155)

    print(
        f"Modalità costi: "
        f"{settings['cost_mode']}"
    )

    print(
        f"Modello costi:  "
        f"{settings['cost_model']}"
    )

    print("-" * 155)

    print(
        f"{'RANK':>4} "
        f"{'STRATEGIA':20} "
        f"{'ASSET':12} "
        f"{'TF':>5} "
        f"{'FOLDS':>6} "
        f"{'POSITIVE':>10} "
        f"{'TRADES':>8} "
        f"{'NET PROFIT':>12} "
        f"{'COMM.':>10} "
        f"{'WFE':>7} "
        f"{'CLASSIFICAZIONE':30}"
    )

    print("-" * 155)

    completed_results = [
        item
        for item in result[
            "ranked_results"
        ]
        if item[
            "status"
        ] == "COMPLETED"
    ]

    for rank, item in enumerate(
        completed_results,
        start=1
    ):
        print(
            f"{rank:4} "
            f"{item['strategy']:20} "
            f"{item['ticker']:12} "
            f"{item['interval']:>5} "
            f"{item['total_folds']:6} "
            f"{item['profitable_fold_percent']:9.2f}% "
            f"{item['total_trades']:8} "
            f"{item['total_net_profit']:12.2f} "
            f"{item['total_commissions']:10.2f} "
            f"{item['walk_forward_efficiency']:7.2f} "
            f"{item['classification']:30}"
        )


def print_strategy_ranking(
    result
):
    print()
    print("=" * 145)
    print(
        "CLASSIFICA AGGREGATA DELLE STRATEGIE"
    )
    print("=" * 145)

    print(
        f"{'RANK':>4} "
        f"{'STRATEGIA':22} "
        f"{'TEST':>6} "
        f"{'POSITIVI':>10} "
        f"{'CANDIDATI':>11} "
        f"{'ROBUSTI':>8} "
        f"{'MEDIAN P/L':>12} "
        f"{'COMM.':>11} "
        f"{'MEDIAN WFE':>11} "
        f"{'STATO':20}"
    )

    print("-" * 145)

    for rank, item in enumerate(
        result[
            "strategy_ranking"
        ],
        start=1
    ):
        print(
            f"{rank:4} "
            f"{item['strategy']:22} "
            f"{item['completed_cases']:6} "
            f"{item['positive_cases']:10} "
            f"{item['candidate_cases']:11} "
            f"{item['robust_cases']:8} "
            f"{item['median_net_profit']:12.2f} "
            f"{item['total_commissions']:11.2f} "
            f"{item['median_walk_forward_efficiency']:11.2f} "
            f"{item['overall_status']:20}"
        )


def print_candidates(
    result
):
    candidates = result[
        "candidates"
    ]

    cost_mode = result[
        "settings"
    ][
        "cost_mode"
    ]

    if cost_mode == "market":
        title = (
            "CANDIDATE NETTE AMMESSE "
            "ALLA VALIDAZIONE AVANZATA"
        )

    else:
        title = (
            "COMBINAZIONI AMMESSE "
            "AL TEST DEI COSTI"
        )

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    if not candidates:
        print(
            "Nessuna combinazione ha superato "
            "i criteri di selezione."
        )

        return

    for candidate in candidates:
        print(
            f"{candidate['strategy']} | "
            f"{candidate['ticker']} | "
            f"{candidate['interval']} | "
            f"P/L {candidate['total_net_profit']:.2f} | "
            f"commissioni "
            f"{candidate['total_commissions']:.2f} | "
            f"fold positive "
            f"{candidate['profitable_fold_percent']:.2f}% | "
            f"WFE "
            f"{candidate['walk_forward_efficiency']:.2f} | "
            f"{candidate['classification']}"
        )


def print_failures(
    result
):
    failures = [
        item
        for item in result[
            "results"
        ]
        if item[
            "status"
        ] in {
            "ERROR",
            "SKIPPED"
        }
    ]

    if not failures:
        return

    print()
    print("=" * 110)
    print(
        "TEST NON COMPLETATI"
    )
    print("=" * 110)

    for failure in failures:
        print(
            f"{failure['strategy']} | "
            f"{failure['ticker']} | "
            f"{failure['interval']} | "
            f"{failure['status']} | "
            f"{failure.get('error', 'Errore sconosciuto')}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Esegue un benchmark multi-asset "
            "e multi-timeframe."
        )
    )

    parser.add_argument(
        "--profile",
        choices=[
            "quick",
            "full"
        ],
        default="quick"
    )

    parser.add_argument(
        "--cost-mode",
        choices=[
            "gross",
            "market"
        ],
        default="gross"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[
            "legacy",
            "trend"
        ],
        default=[
            "legacy",
            "trend"
        ]
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=[
            "5m",
            "15m",
            "1h",
            "1d"
        ],
        default=[
            "5m",
            "15m",
            "1h",
            "1d"
        ]
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

    parser.add_argument(
        "--margin",
        action="append",
        default=[],
        help=(
            "Margine per contratto. "
            "Formato: CL=F:5000. "
            "Può essere ripetuto."
        )
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    try:
        assets = (
            QUICK_ASSETS
            if args.profile == "quick"
            else FULL_ASSETS
        )

        strategies = build_strategies(
            args.strategies
        )

        market_cases = build_market_cases(
            assets=assets,
            timeframes=args.timeframes
        )

        margin_overrides = (
            parse_margin_overrides(
                args.margin
            )
        )

        pipeline = MarketDataPipeline()

        def load_data(
            ticker,
            period,
            interval
        ):
            return pipeline.get_historical_data(
                ticker=ticker,
                period=period,
                interval=interval
            )

        runner = StrategyBenchmarkRunner(
            strategies=strategies,
            data_loader=load_data,
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
            cost_mode=args.cost_mode,
            margin_overrides=(
                margin_overrides
            ),
            progress_callback=print
        )

        result = runner.run(
            market_cases=market_cases
        )

        print_case_ranking(
            result
        )

        print_strategy_ranking(
            result
        )

        print_candidates(
            result
        )

        print_failures(
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