from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.kibot_intraday_research import (
    KibotIntradayResearchConfig,
    KibotIntradayResearchEngine,
    summarize_phase_microstructure,
)
from adaptive.run_kibot_intraday_research import main, parse_arguments


def synthetic_ticks(sessions: int = 21) -> pd.DataFrame:
    rows: list[list[object]] = []
    days = pd.bdate_range("2026-07-01", periods=sessions)
    for day_number, day in enumerate(days):
        start = pd.Timestamp(day.date()) + pd.Timedelta(hours=9, minutes=30)
        for bucket in range(26):
            timestamp = start + pd.Timedelta(minutes=15 * bucket)
            wave = np.sin((day_number * 26 + bucket) / 11.0)
            price = 100.0 + day_number * 0.04 + bucket * 0.01 + wave * 0.12
            rows.append(
                [
                    day.strftime("%m/%d/%Y"),
                    timestamp.strftime("%H:%M:%S"),
                    price,
                    price - 0.01,
                    price + 0.01,
                    100 + bucket,
                ]
            )
    return pd.DataFrame(rows)


def write_ticks(path: Path, sessions: int = 21) -> None:
    synthetic_ticks(sessions).to_csv(path, index=False, header=False)


def compact_config() -> KibotIntradayResearchConfig:
    return KibotIntradayResearchConfig(
        context_normalization_window=12,
        context_normalization_min_periods=6,
        neighbors=4,
        minimum_candidates=8,
        sample_stride=2,
    )


def test_integrated_report_remains_descriptive(tmp_path) -> None:
    path = tmp_path / "IVE.txt"
    write_ticks(path)

    report = KibotIntradayResearchEngine(compact_config()).analyze(path, "IVE")

    assert report.research_only
    assert report.tick_result.audit.research_only
    assert report.stability.research_only
    assert report.lorentzian.research_only
    assert len(report.features) == 21 * 26
    assert report.stability.complete_blocks == 4
    assert not report.phase_microstructure.empty
    assert (
        report.lorentzian.descriptors["lorentzian_neighbor_count"] > 0
    ).any()
    forbidden = ("signal", "direction", "prediction", "pnl", "entry", "exit")
    assert all(
        not any(token in column.lower() for token in forbidden)
        for column in report.lorentzian.descriptors.columns
    )


def test_future_session_cannot_change_past_features_or_lorentzian(tmp_path) -> None:
    prefix_path = tmp_path / "prefix.txt"
    full_path = tmp_path / "full.txt"
    write_ticks(prefix_path, sessions=20)
    write_ticks(full_path, sessions=21)
    engine = KibotIntradayResearchEngine(compact_config())

    prefix = engine.analyze(prefix_path)
    full = engine.analyze(full_path)

    pd.testing.assert_frame_equal(
        prefix.features,
        full.features.iloc[: len(prefix.features)],
    )
    pd.testing.assert_frame_equal(
        prefix.lorentzian.normalized_features,
        full.lorentzian.normalized_features.iloc[: len(prefix.features)],
    )
    pd.testing.assert_frame_equal(
        prefix.lorentzian.descriptors,
        full.lorentzian.descriptors.iloc[: len(prefix.features)],
    )


def test_phase_microstructure_has_declared_order_and_complete_coverage(tmp_path) -> None:
    path = tmp_path / "IVE.txt"
    write_ticks(path)
    bars = (
        KibotIntradayResearchEngine(compact_config())
        .analyze(path)
        .tick_result.bars
    )

    phase = summarize_phase_microstructure(bars)

    assert list(phase["session_phase"]) == ["OPEN", "MID_SESSION", "CLOSE"]
    assert int(phase["bars"].sum()) == len(bars)
    assert (phase["median_spread_bps"] > 0.0).all()


def test_cli_defaults_target_short_sample_without_5m(tmp_path, capsys) -> None:
    path = tmp_path / "IVE.txt"
    write_ticks(path)

    arguments = parse_arguments(["--file", str(path)])
    assert arguments.interval == "15m"
    assert arguments.sessions_per_block == 5
    assert arguments.context_normalization_min_periods == 10
    assert main(["--file", str(path), "--ticker", "IVE"]) == 0
    output = capsys.readouterr().out
    assert "KIBOT 15M INTEGRATED RESEARCH" in output
    assert "MICROSTRUTTURA PER FASE" in output
    assert "Nessuna direzione, previsione" in output
    assert "non la redditivita'" in output
