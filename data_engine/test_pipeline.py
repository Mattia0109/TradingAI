import pandas as pd

from data_engine.pipeline import MarketDataPipeline


def test_pipeline_market_data():
    pipeline = MarketDataPipeline()
    history = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000_000, 1_100_000],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="B"),
    )
    pipeline.provider.get_history = lambda **kwargs: history

    data = pipeline.get_market_data("AAPL")

    assert data is not None
    assert data["ticker"] == "AAPL"
    assert data["close"] == 102.0
    assert data["volume"] == 1_100_000
