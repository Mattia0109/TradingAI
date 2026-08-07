import pandas as pd

import data_engine.providers.yahoo_provider as yahoo_provider_module
from data_engine.providers.yahoo_provider import YahooProvider


def sample_history():
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.5],
            "Volume": [1_000_000, 1_100_000],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="B"),
    )


def test_yahoo_history(monkeypatch):
    monkeypatch.setattr(
        yahoo_provider_module.yf,
        "download",
        lambda **kwargs: sample_history(),
    )

    data = YahooProvider().get_history("AAPL", period="5d")

    assert len(data) == 2
    assert float(data["Close"].iloc[-1]) == 102.5


def test_latest_price(monkeypatch):
    provider = YahooProvider()
    monkeypatch.setattr(
        provider,
        "get_history",
        lambda *args, **kwargs: sample_history(),
    )

    price = provider.get_latest_price("AAPL")

    assert price == 102.5
