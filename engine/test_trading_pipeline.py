import pandas as pd

import engine.trading_pipeline as trading_pipeline_module
from engine.trading_pipeline import TradingPipeline


class StaticDecisionEngine:
    def __init__(self, action):
        self.action = action

    def decide(self, analysis):
        return {
            "action": self.action,
            "confidence": 80,
            "risk": "LOW",
            "reasons": [],
        }


def configured_pipeline(monkeypatch, action):
    pipeline = TradingPipeline()
    market_data = pd.DataFrame({"close": [100.0], "Volatility": [2.0]})
    pipeline.data_pipeline.get_historical_data = lambda ticker: market_data
    pipeline.prepare_data = lambda data: data
    pipeline.decision_engine = StaticDecisionEngine(action)
    monkeypatch.setattr(
        trading_pipeline_module,
        "analyze_asset",
        lambda data: {"score": 0, "risk": "LOW", "reasons": []},
    )
    return pipeline


def test_trading_pipeline_buy(monkeypatch):
    result = configured_pipeline(monkeypatch, "BUY").generate_signal("AAPL")

    assert result is not None
    assert result["signal"]["direction"] == "LONG"


def test_trading_pipeline_preserves_wait(monkeypatch):
    result = configured_pipeline(monkeypatch, "WAIT").generate_signal("AAPL")

    assert result is None
