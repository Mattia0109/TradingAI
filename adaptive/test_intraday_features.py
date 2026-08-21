from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from adaptive.intraday_audit import IntradayReadiness
from adaptive.intraday_features import (
    FeatureParity,
    IntradayFeatureConfig,
    IntradayReferenceFeatureEngine,
    lorentzian_distance,
)
from adaptive.run_intraday_feature_report import main, parse_arguments


def feature_market(rows: int = 180) -> pd.DataFrame:
    dates = pd.date_range(
        "2025-01-06 09:30",
        periods=rows,
        freq="15min",
        tz="America/New_York",
    )
    x = np.arange(rows, dtype=float)
    close = 100.0 + 0.025 * x + 0.8 * np.sin(x / 5.0)
    open_price = close - 0.05 * np.cos(x / 4.0)
    high = np.maximum(open_price, close) + 0.30 + 0.02 * np.sin(x / 3.0)
    low = np.minimum(open_price, close) - 0.30 - 0.02 * np.cos(x / 3.0)
    volume = 10_000.0 + 1_000.0 * (1.0 + np.sin(x / 7.0))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def compact_config(**overrides) -> IntradayFeatureConfig:
    values = {
        "squeeze_bb_length": 5,
        "squeeze_kc_length": 5,
        "chop_length": 5,
        "cmf_length": 5,
        "cmf_gradient_length": 7,
    }
    values.update(overrides)
    return IntradayFeatureConfig(**values)


def test_defaults_match_supplied_reference_parameters() -> None:
    config = IntradayFeatureConfig()

    assert config.squeeze_bb_length == 20
    assert config.squeeze_declared_bb_multiplier == 2.0
    assert config.squeeze_kc_length == 20
    assert config.squeeze_kc_multiplier == 1.5
    assert config.squeeze_use_true_range is True
    assert config.chop_length == 10
    assert config.chop_choppy_threshold == 60.0
    assert config.chop_trending_threshold == 40.0
    assert config.cmf_length == 20


def test_lorentzian_distance_matches_reference_formula() -> None:
    left = [10.0, 20.0, 30.0, 40.0, 50.0]
    right = [9.0, 23.0, 30.0, 36.0, 52.0]
    expected = sum(math.log(1.0 + abs(a - b)) for a, b in zip(left, right))

    assert lorentzian_distance(left, right) == pytest.approx(expected)
    assert lorentzian_distance(left, left) == 0.0


def test_cmf_matches_manual_rolling_money_flow() -> None:
    market = feature_market(20)
    config = compact_config(cmf_length=3)
    report = IntradayReferenceFeatureEngine(config).compute(market)
    position = 8
    window = market.iloc[position - 2 : position + 1]
    ad = (
        (2.0 * window["close"] - window["low"] - window["high"])
        / (window["high"] - window["low"])
        * window["volume"]
    )
    expected = float(ad.sum() / window["volume"].sum())

    assert report.values["cmf"].iloc[position] == pytest.approx(expected)


def test_chop_uses_pine_wilder_atr_and_strict_thresholds() -> None:
    market = feature_market(24)
    config = compact_config(chop_length=3)
    engine = IntradayReferenceFeatureEngine(config)
    report = engine.compute(market)
    frame = market.iloc[:8]
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    rma = float(true_range.iloc[:3].mean())
    for value in true_range.iloc[3:8]:
        rma = (2.0 * rma + float(value)) / 3.0
    price_range = float(frame["high"].iloc[-3:].max() - frame["low"].iloc[-3:].min())
    expected = 100.0 + 100.0 * math.log10(rma / price_range) / math.log10(3)

    assert report.values["choppiness"].iloc[7] == pytest.approx(expected)
    segment = report.values["chop_segment"].iloc[7]
    expected_segment = "CHOPPY" if expected > 60 else "TRENDING" if expected < 40 else "NEUTRAL"
    assert segment == expected_segment


def test_squeeze_preserves_unused_declared_bb_multiplier() -> None:
    market = feature_market()
    base = IntradayReferenceFeatureEngine(
        compact_config(squeeze_declared_bb_multiplier=2.0)
    ).compute(market)
    changed = IntradayReferenceFeatureEngine(
        compact_config(squeeze_declared_bb_multiplier=9.0)
    ).compute(market)

    pd.testing.assert_series_equal(
        base.values["squeeze_momentum"],
        changed.values["squeeze_momentum"],
    )
    pd.testing.assert_series_equal(
        base.values["squeeze_state"],
        changed.values["squeeze_state"],
    )


def test_squeeze_reports_constant_price_inside_keltner_channel() -> None:
    market = feature_market(30)
    market[["open", "close"]] = 100.0
    market["high"] = 101.0
    market["low"] = 99.0
    report = IntradayReferenceFeatureEngine(compact_config()).compute(market)

    assert report.values["squeeze_state"].iloc[-1] == "SQUEEZE_ON"
    assert report.values["squeeze_momentum"].iloc[-1] == pytest.approx(0.0)


def test_pine_linreg_returns_endpoint_for_a_linear_window() -> None:
    series = pd.Series(np.arange(20, dtype=float) * 2.5 - 3.0)
    result = IntradayReferenceFeatureEngine._pine_linreg(series, 5)

    pd.testing.assert_series_equal(
        result.dropna(),
        series.iloc[4:],
        check_names=False,
    )


def test_future_mutation_cannot_change_past_feature_values() -> None:
    market = feature_market(180)
    engine = IntradayReferenceFeatureEngine()
    baseline = engine.compute(market).values
    cutoff = 120
    changed = market.copy()
    changed.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 1.25
    changed.loc[cutoff + 1 :, "volume"] *= 4.0
    mutated = engine.compute(changed).values

    pd.testing.assert_frame_equal(
        baseline.iloc[: cutoff + 1],
        mutated.iloc[: cutoff + 1],
    )


def test_report_is_descriptive_and_marks_lorentzian_as_partial() -> None:
    report = IntradayReferenceFeatureEngine().compute(feature_market())

    assert report.research_only is True
    assert report.parity["choppiness"] is FeatureParity.EXACT_REFERENCE_FORMULA
    assert report.parity["squeeze_momentum"] is FeatureParity.EXACT_REFERENCE_FORMULA
    assert report.parity["cmf"] is FeatureParity.EXACT_REFERENCE_FORMULA
    assert report.parity["lorentzian_distance"] is FeatureParity.EXACT_REFERENCE_FORMULA
    assert report.parity["lorentzian_classifier"] is FeatureParity.PARTIAL_MISSING_LIBRARY_SOURCE
    forbidden = {"direction", "order", "position", "pnl", "expected_return"}
    assert forbidden.isdisjoint(report.values.columns)


def test_zero_volume_is_rejected_like_cmf_reference() -> None:
    market = feature_market()
    market["volume"] = 0.0

    with pytest.raises(ValueError, match="richiede dati di volume"):
        IntradayReferenceFeatureEngine().compute(market)


def test_cli_defaults_to_historical_distribution_report() -> None:
    arguments = parse_arguments([])

    assert arguments.period == "60d"
    assert arguments.interval == "15m"
    assert arguments.tickers == ["SPY", "QQQ", "IWM", "GLD", "TLT"]


def test_cli_reports_distributions_and_parity_without_current_signal(
    monkeypatch,
    capsys,
) -> None:
    market = feature_market(160)
    monkeypatch.setattr(
        "adaptive.run_intraday_feature_report.load_markets",
        lambda pipeline, tickers, period, interval: ({"SPY": market}, {}),
    )
    monkeypatch.setattr(
        "adaptive.run_intraday_feature_report.IntradayResearchAuditor.audit",
        lambda self, ticker, data: type(
            "Audit",
            (),
            {"status": IntradayReadiness.LIMITED},
        )(),
    )

    assert main(["--tickers", "SPY"]) == 0
    output = capsys.readouterr().out

    assert "FEATURE DISTRIBUTIONS — RESEARCH-ONLY" in output
    assert "Le percentuali descrivono l'intero campione" in output
    assert "PARTIAL_MISSING_LIBRARY_SOURCE" in output
    assert "P&L" in output
    assert "ultima barra:" not in output.lower()
