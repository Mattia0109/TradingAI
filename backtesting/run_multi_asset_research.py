import argparse
import sys

from backtesting.multi_asset_research import (
    MultiAssetResearchBenchmark
)
from backtesting.portfolio_risk_allocator import (
    PortfolioRiskAllocator
)
from data_engine.pipeline import (
    MarketDataPipeline
)
from data_engine.research_calendar import (
    align_market_data_to_reference_calendar
)
from market.research_universe import (
    get_research_universe,
    register_research_universe_specifications
)
from strategies.multi_asset_tsmom import (
    MultiAssetTimeSeriesMomentumStrategy
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark reale multi-asset TSMOM "
            "su calendario comune."
        )
    )

    parser.add_argument(
        "--period",
        default="max"
    )

    parser.add_argument(
        "--interval",
        default="1d"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0
    )

    parser.add_argument(
        "--frequencies",
        nargs="+",
        type=int,
        default=[
            5,
            21,
            63
        ]
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None
    )

    parser.add_argument(
        "--minimum-active-assets",
        type=int,
        default=3
    )

    parser.add_argument(
        "--signal-sizing",
        choices=[
            "continuous",
            "directional"
        ],
        default="directional",
        help=(
            "Dimensionamento del segnale TSMOM. "
            "Directional usa solo la direzione e lascia "
            "il rischio all'allocatore."
        )
    )

    parser.add_argument(
        "--signal-method",
        choices=[
            "return",
            "linear_trend"
        ],
        default="linear_trend",
        help=(
            "Metodo dei componenti TSMOM. Linear trend "
            "opera solo su trend statisticamente significativi."
        )
    )

    parser.add_argument(
        "--trend-significance-threshold",
        type=float,
        default=2.0,
        help=(
            "Soglia assoluta del t-stat Newey-West per il "
            "metodo linear_trend. Default: 2.0."
        )
    )

    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.15,
        help=(
            "Forza minima del segnale aggregato. "
            "Default: 0.15."
        )
    )

    parser.add_argument(
        "--maximum-resize-cost-ratio",
        type=float,
        default=None,
        help=(
            "Costo massimo accettato per un piccolo aumento "
            "della stessa posizione, espresso come quota "
            "del nozionale dell'ordine. Disabilitato per "
            "default perché il filtro è asimmetrico."
        )
    )

    parser.add_argument(
        "--calendar-anchor",
        default="SPY",
        help=(
            "Ticker le cui sessioni vengono usate "
            "come calendario comune."
        )
    )

    return parser.parse_args()


def load_market_data(
    pipeline,
    universe,
    period,
    interval
):
    market_data = {}
    errors = {}

    total = len(
        universe
    )

    for index, ticker in enumerate(
        universe,
        start=1
    ):
        print(
            f"[{index}/{total}] "
            f"Scaricamento {ticker} "
            f"{period} {interval}"
        )

        try:
            data = (
                pipeline.get_historical_data(
                    ticker=ticker,
                    period=period,
                    interval=interval
                )
            )

            if data is None or data.empty:
                errors[ticker] = (
                    "Nessun dato disponibile."
                )

                continue

            market_data[
                ticker
            ] = data

            print(
                f"    {len(data)} candele"
            )

        except Exception as error:
            errors[ticker] = str(
                error
            )

            print(
                f"    ERRORE: {error}"
            )

    return market_data, errors


def print_calendar_report(
    report
):
    print()
    print("=" * 100)
    print("ALLINEAMENTO CALENDARIO")
    print("=" * 100)

    print(
        f"Ticker di riferimento:    "
        f"{report['reference_ticker']}"
    )

    print(
        f"Sessioni di riferimento:  "
        f"{report['reference_sessions']}"
    )

    print(
        f"Asset allineati:          "
        f"{report['asset_count']}"
    )

    print(
        f"Righe originali:          "
        f"{report['total_rows_before']}"
    )

    print(
        f"Righe dopo allineamento:  "
        f"{report['total_rows_after']}"
    )

    print(
        f"Righe eliminate:          "
        f"{report['total_removed_rows']}"
    )

    print("-" * 100)

    print(
        f"{'ASSET':12} "
        f"{'PRIMA':>9} "
        f"{'DOPO':>9} "
        f"{'RIMOSSE':>9} "
        f"{'WEEKEND PRIMA':>15} "
        f"{'WEEKEND DOPO':>14}"
    )

    print("-" * 100)

    for ticker in sorted(
        report[
            "assets"
        ]
    ):
        asset_report = report[
            "assets"
        ][
            ticker
        ]

        print(
            f"{ticker:12} "
            f"{asset_report['rows_before']:9} "
            f"{asset_report['rows_after']:9} "
            f"{asset_report['removed_rows']:9} "
            f"{asset_report['weekend_rows_before']:15} "
            f"{asset_report['weekend_rows_after']:14}"
        )


def print_ranking(
    result,
    calendar_report
):
    print()
    print("=" * 170)
    print(
        "MULTI-ASSET TSMOM RESEARCH BENCHMARK"
    )
    print("=" * 170)

    print(
        f"Calendario comune: "
        f"{calendar_report['reference_ticker']}"
    )

    print(
        f"Inizio valutazione comune: "
        f"{result['trade_start_time']}"
    )

    print(
        f"Asset valutati: "
        f"{len(result['eligible_assets'])}"
    )

    print(
        ", ".join(
            result[
                "eligible_assets"
            ]
        )
    )

    print("-" * 170)

    print(
        f"{'RANK':>4} "
        f"{'MODELLO':26} "
        f"{'RETURN':>10} "
        f"{'CAGR':>10} "
        f"{'VOL':>9} "
        f"{'SHARPE':>9} "
        f"{'MAX DD':>9} "
        f"{'CALMAR':>9} "
        f"{'COSTI':>10} "
        f"{'ORDINI':>8} "
        f"{'BATCH':>7} "
        f"{'TURNOVER':>10} "
        f"{'AVG GROSS':>11}"
    )

    print("-" * 170)

    for rank, item in enumerate(
        result[
            "ranked_results"
        ],
        start=1
    ):
        metrics = item[
            "metrics"
        ]

        average_gross = float(
            metrics.get(
                "average_gross_exposure",
                0.0
            )
        )

        print(
            f"{rank:4} "
            f"{item['benchmark_name']:26} "
            f"{metrics['total_return_percent']:9.2f}% "
            f"{metrics['annualized_return_percent']:9.2f}% "
            f"{metrics['annualized_volatility_percent']:8.2f}% "
            f"{metrics['sharpe_ratio']:9.2f} "
            f"{metrics['maximum_drawdown_percent']:8.2f}% "
            f"{metrics['calmar_ratio']:9.2f} "
            f"{metrics['total_costs']:10.2f} "
            f"{int(metrics['order_count']):8} "
            f"{int(item.get('uneconomic_resize_skip_count', 0)):7} "
            f"{metrics['total_turnover_ratio']:10.2f} "
            f"{average_gross:10.2f}"
        )


def print_frequency_summary(
    result
):
    summary = result[
        "frequency_summary"
    ]

    print()
    print("=" * 92)
    print("STABILITÀ TRA FREQUENZE")
    print("=" * 92)

    print(
        f"Frequenze testate:          "
        f"{summary['tested_frequencies']}"
    )

    print(
        f"Frequenze positive:         "
        f"{summary['positive_frequencies']} "
        f"({summary['positive_frequency_percent']:.2f}%)"
    )

    print(
        f"Rendimento mediano:         "
        f"{summary['median_return_percent']:.2f}%"
    )

    print(
        f"Sharpe mediano:             "
        f"{summary['median_sharpe_ratio']:.2f}"
    )

    print(
        f"Peggiore rendimento:        "
        f"{summary['worst_return_percent']:.2f}%"
    )

    print(
        f"Peggiore drawdown:          "
        f"{summary['worst_drawdown_percent']:.2f}%"
    )

    print(
        f"Classificazione:            "
        f"{summary['status']}"
    )


def print_yearly_returns(
    result
):
    print()
    print("=" * 110)
    print("RENDIMENTI ANNUALI TSMOM")
    print("=" * 110)

    tsmom_results = result[
        "tsmom_results"
    ]

    all_years = sorted(
        {
            year
            for item in tsmom_results
            for year in item[
                "yearly_returns"
            ]
        }
    )

    if not all_years:
        print(
            "Nessun rendimento annuale disponibile."
        )

        return

    header = (
        f"{'ANNO':>6} "
        +
        " ".join(
            f"{item['benchmark_name']:>16}"
            for item in tsmom_results
        )
    )

    print(header)
    print("-" * len(header))

    for year in all_years:
        values = []

        for item in tsmom_results:
            value = item[
                "yearly_returns"
            ].get(
                year
            )

            if value is None:
                values.append(
                    f"{'-':>16}"
                )

            else:
                values.append(
                    f"{value:15.2f}%"
                )

        print(
            f"{year:6} "
            +
            " ".join(
                values
            )
        )


def print_failures(
    download_errors,
    result
):
    excluded = result[
        "excluded_assets"
    ]

    if not download_errors and not excluded:
        return

    print()
    print("=" * 100)
    print("ASSET NON VALUTATI")
    print("=" * 100)

    for ticker, error in (
        download_errors.items()
    ):
        print(
            f"{ticker} | DOWNLOAD | {error}"
        )

    for ticker, reason in (
        excluded.items()
    ):
        print(
            f"{ticker} | STORICO | {reason}"
        )


def main():
    args = parse_arguments()

    try:
        if args.capital <= 0:
            raise ValueError(
                "capital deve essere positivo."
            )

        if any(
            frequency <= 0
            for frequency in args.frequencies
        ):
            raise ValueError(
                "Le frequenze devono essere positive."
            )

        if args.minimum_active_assets <= 0:
            raise ValueError(
                "minimum-active-assets deve "
                "essere positivo."
            )

        register_research_universe_specifications()

        universe = get_research_universe(
            selected_tickers=(
                args.tickers
            )
        )

        calendar_anchor = (
            str(
                args.calendar_anchor
            )
            .upper()
            .strip()
        )

        if calendar_anchor not in universe:
            raise ValueError(
                f"Il calendario anchor "
                f"{calendar_anchor} deve essere "
                "presente nell'universo selezionato."
            )

        pipeline = MarketDataPipeline()

        raw_market_data, download_errors = (
            load_market_data(
                pipeline=pipeline,
                universe=universe,
                period=args.period,
                interval=args.interval
            )
        )

        if len(raw_market_data) < 3:
            raise ValueError(
                "Servono almeno tre asset "
                "con dati validi."
            )

        calendar_result = (
            align_market_data_to_reference_calendar(
                market_data=raw_market_data,
                reference_ticker=(
                    calendar_anchor
                )
            )
        )

        market_data = calendar_result[
            "market_data"
        ]

        calendar_report = calendar_result[
            "report"
        ]

        print_calendar_report(
            calendar_report
        )

        usable_market_data = {
            ticker: data
            for ticker, data
            in market_data.items()
            if not data.empty
        }

        if len(usable_market_data) < 3:
            raise ValueError(
                "Dopo l'allineamento rimangono "
                "meno di tre asset utilizzabili."
            )

        available_asset_classes = {
            ticker: universe[
                ticker
            ]
            for ticker in usable_market_data
        }

        strategy = (
            MultiAssetTimeSeriesMomentumStrategy(
                lookback_weights={
                    21: 0.10,
                    63: 0.20,
                    126: 0.30,
                    252: 0.40
                },
                volatility_span=60,
                annualization_factor=252,
                no_trade_threshold=(
                    args.signal_threshold
                ),
                component_clip=1.0,
                signal_sizing=(
                    args.signal_sizing
                ),
                component_method=(
                    args.signal_method
                ),
                trend_significance_threshold=(
                    args.trend_significance_threshold
                )
            )
        )

        allocator = PortfolioRiskAllocator(
            target_portfolio_volatility=0.10,
            max_gross_exposure=1.0,
            max_asset_weight=0.10,
            max_asset_class_weight=0.30,
            minimum_trade_weight=0.0025,
            turnover_buffer=0.005,
            drawdown_start=0.05,
            drawdown_medium=0.10,
            drawdown_severe=0.15,
            drawdown_kill_switch=0.20
        )

        benchmark = MultiAssetResearchBenchmark(
            strategy=strategy,
            allocator=allocator,
            initial_capital=(
                args.capital
            ),
            rebalance_frequencies=(
                args.frequencies
            ),
            use_market_costs=True,
            minimum_active_assets=(
                args.minimum_active_assets
            ),
            maximum_resize_cost_ratio=(
                args.maximum_resize_cost_ratio
            )
        )

        result = benchmark.run(
            market_data=usable_market_data,
            asset_classes=(
                available_asset_classes
            )
        )

        result[
            "calendar_report"
        ] = calendar_report

        print_ranking(
            result=result,
            calendar_report=(
                calendar_report
            )
        )

        print_frequency_summary(
            result
        )

        print_yearly_returns(
            result
        )

        print_failures(
            download_errors=(
                download_errors
            ),
            result=result
        )

    except Exception as error:
        print()
        print(
            f"ERRORE: {error}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
