import pandas as pd
import pytest

from data_engine.research_calendar import (
    ResearchCalendarAligner,
    align_market_data_to_reference_calendar
)


def create_spy_data():
    dates = pd.bdate_range(
        start="2025-01-01",
        periods=20
    )

    return pd.DataFrame(
        {
            "date": dates,
            "close": [
                100 + index
                for index in range(
                    len(dates)
                )
            ]
        }
    )


def create_crypto_data():
    dates = pd.date_range(
        start="2025-01-01",
        periods=30,
        freq="1D"
    )

    return pd.DataFrame(
        {
            "date": dates,
            "close": [
                1000 + index
                for index in range(
                    len(dates)
                )
            ]
        }
    )


def test_crypto_is_aligned_to_spy_sessions():

    result = (
        align_market_data_to_reference_calendar(
            market_data={
                "SPY": create_spy_data(),
                "BTC-USD": (
                    create_crypto_data()
                )
            },
            reference_ticker="SPY"
        )
    )

    aligned = result[
        "market_data"
    ]

    spy_dates = set(
        aligned["SPY"][
            "date"
        ]
    )

    bitcoin_dates = set(
        aligned["BTC-USD"][
            "date"
        ]
    )

    assert bitcoin_dates.issubset(
        spy_dates
    )

    assert all(
        date.dayofweek < 5
        for date in bitcoin_dates
    )

    assert (
        result[
            "report"
        ][
            "assets"
        ][
            "BTC-USD"
        ][
            "weekend_rows_after"
        ]
        == 0
    )


def test_weekend_rows_are_removed():

    result = (
        align_market_data_to_reference_calendar(
            market_data={
                "SPY": create_spy_data(),
                "ETH-USD": (
                    create_crypto_data()
                )
            }
        )
    )

    ethereum_report = (
        result[
            "report"
        ][
            "assets"
        ][
            "ETH-USD"
        ]
    )

    assert (
        ethereum_report[
            "weekend_rows_before"
        ]
        > 0
    )

    assert (
        ethereum_report[
            "weekend_rows_after"
        ]
        == 0
    )

    assert (
        ethereum_report[
            "removed_rows"
        ]
        > 0
    )


def test_missing_reference_ticker_is_rejected():

    aligner = ResearchCalendarAligner(
        reference_ticker="SPY"
    )

    with pytest.raises(
        ValueError
    ):
        aligner.align(
            {
                "BTC-USD": (
                    create_crypto_data()
                )
            }
        )


def test_duplicate_dates_are_removed():

    spy = create_spy_data()

    duplicated_spy = pd.concat(
        [
            spy,
            spy.iloc[
                [-1]
            ]
        ],
        ignore_index=True
    )

    result = (
        align_market_data_to_reference_calendar(
            market_data={
                "SPY": duplicated_spy,
                "BTC-USD": (
                    create_crypto_data()
                )
            }
        )
    )

    aligned_spy = result[
        "market_data"
    ][
        "SPY"
    ]

    assert (
        aligned_spy[
            "date"
        ].duplicated().sum()
        == 0
    )

    assert len(
        aligned_spy
    ) == len(
        spy
    )


def test_index_can_be_used_as_date():

    spy = create_spy_data().set_index(
        "date"
    )

    result = (
        align_market_data_to_reference_calendar(
            market_data={
                "SPY": spy,
                "BTC-USD": (
                    create_crypto_data()
                )
            }
        )
    )

    assert not result[
        "market_data"
    ][
        "SPY"
    ].empty

    assert "date" in result[
        "market_data"
    ][
        "SPY"
    ].columns