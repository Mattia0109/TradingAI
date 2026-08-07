import math

from backtesting.backtester import Backtester
from data_engine.pipeline import MarketDataPipeline
from market.universe import MarketUniverse


def format_metric(value, decimals=2):
    """
    Formatta valori numerici, inclusi infinito e None.
    """

    if value is None:
        return "N/D"

    if isinstance(value, float) and math.isinf(value):
        return "INF"

    return f"{value:.{decimals}f}"


def run_single_backtest(
    ticker,
    data_pipeline,
    initial_capital=10000.0
):
    """
    Esegue il backtest di un singolo asset.
    """

    historical_data = (
        data_pipeline.get_historical_data(
            ticker
        )
    )

    if historical_data is None:
        return {
            "ticker": ticker,
            "status": "NO_DATA"
        }

    backtester = Backtester(
        initial_capital=initial_capital,
        max_position_percent=0.10,
        minimum_history=30,
        commission_percent=0.001,
        slippage_percent=0.0005
    )

    try:
        result = backtester.run(
            ticker=ticker,
            data=historical_data
        )

    except Exception as error:
        return {
            "ticker": ticker,
            "status": "ERROR",
            "error": str(error)
        }

    metrics = result["metrics"]

    return {
        "ticker": ticker,
        "asset_type": result["asset_type"],
        "status": "OK",
        "return_percent": (
            metrics["total_return_percent"]
        ),
        "net_profit": metrics["net_profit"],
        "total_trades": metrics["total_trades"],
        "win_rate": (
            metrics["win_rate_percent"]
        ),
        "profit_factor": (
            metrics["profit_factor"]
        ),
        "max_drawdown": (
            metrics["max_drawdown_percent"]
        ),
        "commissions": metrics.get(
            "total_commissions",
            0
        )
    }


def calculate_summary(results):
    """
    Calcola un riepilogo generale.
    """

    valid_results = [
        result
        for result in results
        if result["status"] == "OK"
    ]

    profitable = [
        result
        for result in valid_results
        if result["return_percent"] > 0
    ]

    total_net_profit = sum(
        result["net_profit"]
        for result in valid_results
    )

    total_trades = sum(
        result["total_trades"]
        for result in valid_results
    )

    average_return = (
        sum(
            result["return_percent"]
            for result in valid_results
        ) / len(valid_results)
        if valid_results
        else 0
    )

    return {
        "tested_assets": len(valid_results),
        "profitable_assets": len(profitable),
        "losing_assets": (
            len(valid_results) -
            len(profitable)
        ),
        "total_net_profit": total_net_profit,
        "total_trades": total_trades,
        "average_return": average_return
    }


def print_results(results):
    """
    Stampa la classifica dei backtest.
    """

    valid_results = [
        result
        for result in results
        if result["status"] == "OK"
    ]

    valid_results.sort(
        key=lambda item: item["return_percent"],
        reverse=True
    )

    print()
    print("=" * 105)
    print("TRADING AI - MULTI-ASSET BACKTEST")
    print("=" * 105)

    header = (
        f"{'ASSET':12} "
        f"{'TIPO':11} "
        f"{'RETURN':>9} "
        f"{'P/L':>10} "
        f"{'TRADES':>8} "
        f"{'WIN RATE':>10} "
        f"{'PF':>8} "
        f"{'DRAWDOWN':>11} "
        f"{'COSTI':>9}"
    )

    print(header)
    print("-" * 105)

    for result in valid_results:
        print(
            f"{result['ticker']:12} "
            f"{result['asset_type']:11} "
            f"{result['return_percent']:8.2f}% "
            f"{result['net_profit']:10.2f} "
            f"{result['total_trades']:8} "
            f"{result['win_rate']:9.2f}% "
            f"{format_metric(result['profit_factor']):>8} "
            f"{result['max_drawdown']:10.2f}% "
            f"{result['commissions']:9.2f}"
        )

    failed_results = [
        result
        for result in results
        if result["status"] != "OK"
    ]

    if failed_results:
        print()
        print("ASSET NON ANALIZZATI")
        print("-" * 105)

        for result in failed_results:
            message = result.get(
                "error",
                result["status"]
            )

            print(
                f"{result['ticker']}: {message}"
            )

    summary = calculate_summary(results)

    print()
    print("=" * 105)
    print("RIEPILOGO")
    print("=" * 105)

    print(
        f"Asset analizzati:        "
        f"{summary['tested_assets']}"
    )
    print(
        f"Asset profittevoli:      "
        f"{summary['profitable_assets']}"
    )
    print(
        f"Asset in perdita:        "
        f"{summary['losing_assets']}"
    )
    print(
        f"Trade complessivi:       "
        f"{summary['total_trades']}"
    )
    print(
        f"Rendimento medio:        "
        f"{summary['average_return']:.2f}%"
    )
    print(
        f"Somma P/L simulati:      "
        f"{summary['total_net_profit']:.2f}"
    )


def main():
    universe = MarketUniverse()

    assets = universe.get_all_assets()

    data_pipeline = MarketDataPipeline()

    results = []

    print(
        f"Avvio backtest su "
        f"{len(assets)} asset..."
    )

    for index, ticker in enumerate(
        assets,
        start=1
    ):
        print(
            f"[{index}/{len(assets)}] "
            f"Analizzo {ticker}..."
        )

        result = run_single_backtest(
            ticker=ticker,
            data_pipeline=data_pipeline,
            initial_capital=10000
        )

        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()