import pandas as pd

from backtesting.cost_scenarios import (
    CostScenarioAnalyzer
)


class FakeWalkForwardValidator:
    """
    Simula risultati differenti
    in base ai costi configurati.
    """

    def __init__(
        self,
        commission_percent,
        slippage_percent,
        strategy=None,
        **kwargs
    ):
        self.commission_percent = float(
            commission_percent
        )

        self.slippage_percent = float(
            slippage_percent
        )

        self.strategy = strategy


    def run(
        self,
        ticker,
        data
    ):
        total_cost = (
            self.commission_percent
            +
            self.slippage_percent
        )

        if total_cost == 0:
            net_profit = 100.0
            is_robust = True

        elif total_cost <= 0.0015:
            net_profit = 25.0
            is_robust = True

        else:
            net_profit = -20.0
            is_robust = False

        return {
            "ticker": ticker,
            "strategy": "fake_strategy",
            "summary": {
                "total_folds": 5,
                "profitable_folds": (
                    4
                    if net_profit > 0
                    else 1
                ),
                "losing_folds": (
                    1
                    if net_profit > 0
                    else 4
                ),
                "profitable_fold_percent": (
                    80.0
                    if net_profit > 0
                    else 20.0
                ),
                "total_trades": 50,
                "total_net_profit": (
                    net_profit
                ),
                "average_fold_return_percent": (
                    net_profit /
                    10000
                ),
                "median_fold_return_percent": (
                    net_profit /
                    10000
                ),
                "worst_fold_return_percent": -0.2,
                "best_fold_return_percent": 0.4,
                "walk_forward_efficiency": (
                    1.5
                    if net_profit > 0
                    else 0.5
                ),
                "is_robust": is_robust
            },
            "folds": []
        }


def create_data():
    dates = pd.date_range(
        start="2026-01-01",
        periods=300,
        freq="5min"
    )

    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * 300,
            "high": [101.0] * 300,
            "low": [99.0] * 300,
            "close": [100.0] * 300,
            "volume": [1000] * 300
        }
    )


def build_scenarios(
    zero_profit,
    realistic_profit,
    stress_profit,
    realistic_robust=False,
    stress_robust=False
):
    return {
        "zero_costs": {
            "summary": {
                "total_net_profit": (
                    zero_profit
                ),
                "is_robust": (
                    zero_profit > 0
                )
            }
        },
        "realistic": {
            "summary": {
                "total_net_profit": (
                    realistic_profit
                ),
                "is_robust": (
                    realistic_robust
                )
            }
        },
        "stress": {
            "summary": {
                "total_net_profit": (
                    stress_profit
                ),
                "is_robust": (
                    stress_robust
                )
            }
        }
    }


def test_all_cost_profiles_are_executed():

    analyzer = CostScenarioAnalyzer(
        train_bars=100,
        test_bars=50,
        step_bars=50,
        validator_class=(
            FakeWalkForwardValidator
        )
    )

    result = analyzer.run(
        ticker="TEST",
        data=create_data()
    )

    assert set(
        result["scenarios"]
    ) == {
        "zero_costs",
        "realistic",
        "stress"
    }

    assert (
        result["scenarios"][
            "zero_costs"
        ][
            "round_trip_cost_percent"
        ]
        == 0
    )

    assert (
        result["scenarios"][
            "realistic"
        ][
            "round_trip_cost_percent"
        ]
        == 0.3
    )


def test_strategy_without_gross_edge_is_rejected():

    scenarios = build_scenarios(
        zero_profit=-10,
        realistic_profit=-30,
        stress_profit=-50
    )

    result = (
        CostScenarioAnalyzer.build_comparison(
            scenarios
        )
    )

    assert result["classification"] == (
        "REJECT_NO_GROSS_EDGE"
    )

    assert (
        result[
            "accepted_for_research"
        ]
        is False
    )


def test_cost_sensitive_strategy_is_rejected():

    scenarios = build_scenarios(
        zero_profit=100,
        realistic_profit=-5,
        stress_profit=-50
    )

    result = (
        CostScenarioAnalyzer.build_comparison(
            scenarios
        )
    )

    assert result["classification"] == (
        "REJECT_COST_SENSITIVE"
    )

    assert result[
        "gross_edge_positive"
    ] is True

    assert result[
        "realistic_edge_positive"
    ] is False


def test_robust_candidate_is_detected():

    scenarios = build_scenarios(
        zero_profit=150,
        realistic_profit=100,
        stress_profit=50,
        realistic_robust=True,
        stress_robust=True
    )

    result = (
        CostScenarioAnalyzer.build_comparison(
            scenarios
        )
    )

    assert result["classification"] == (
        "ROBUST_CANDIDATE"
    )

    assert (
        result[
            "accepted_for_paper_trading"
        ]
        is True
    )

    assert (
        result[
            "realistic_profit_retention_percent"
        ]
        > 0
    )