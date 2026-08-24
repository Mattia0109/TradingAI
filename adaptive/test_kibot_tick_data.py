from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from adaptive.kibot_tick_data import (
    KibotTickConfig,
    KibotTickLoader,
    KibotTickReadiness,
)
from adaptive.run_kibot_tick_audit import main, parse_arguments


def complete_tick_session(date: str = "08/03/2026") -> pd.DataFrame:
    start = pd.Timestamp("2026-08-03 09:30:00")
    rows = []
    for bucket in range(26):
        timestamp = start + pd.Timedelta(minutes=15 * bucket)
        price = 100.0 + bucket * 0.1
        rows.append(
            [
                date,
                timestamp.strftime("%H:%M:%S"),
                price,
                price - 0.01,
                price + 0.01,
                100,
            ]
        )
    return pd.DataFrame(rows)


def write_ticks(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, header=False)


def compact_config(**overrides) -> KibotTickConfig:
    values = {"minimum_descriptive_sessions": 1}
    values.update(overrides)
    return KibotTickConfig(**values)


def test_tick_rows_are_preserved_and_aggregated_to_ohlcv(tmp_path) -> None:
    frame = complete_tick_session()
    extra = pd.DataFrame(
        [
            ["08/03/2026", "09:30:00", 100.02, 100.01, 100.03, 200],
            ["08/03/2026", "09:30:00", 100.02, 100.01, 100.03, 200],
            ["08/03/2026", "09:44:59", 100.04, 100.03, 100.05, 300],
        ]
    )
    frame = pd.concat([frame, extra], ignore_index=True)
    path = tmp_path / "IVE.txt"
    write_ticks(path, frame)

    result = KibotTickLoader(compact_config()).load(path, "IVE")

    assert result.audit.status is KibotTickReadiness.READY_DESCRIPTIVE
    assert result.audit.observed_sessions == 1
    assert result.audit.output_bars == 26
    assert result.audit.bucket_coverage == 1.0
    assert result.audit.exact_duplicate_rows == 2
    first = result.bars.iloc[0]
    assert first["open"] == pytest.approx(100.0)
    assert first["high"] == pytest.approx(100.04)
    assert first["low"] == pytest.approx(100.0)
    assert first["close"] == pytest.approx(100.04)
    assert first["volume"] == pytest.approx(800.0)
    assert first["tick_count"] == 4


def test_short_tick_history_is_explicitly_limited(tmp_path) -> None:
    path = tmp_path / "IVE.txt"
    write_ticks(path, complete_tick_session())

    result = KibotTickLoader().load(path)

    assert result.audit.status is KibotTickReadiness.LIMITED
    assert any("Sessioni" in reason for reason in result.audit.reasons)


def test_crossed_and_outside_quotes_are_reported_not_silently_changed(tmp_path) -> None:
    frame = complete_tick_session()
    frame.iloc[0, 2:6] = [99.0, 100.02, 100.01, 100]
    path = tmp_path / "IVE.txt"
    write_ticks(path, frame)

    audit = KibotTickLoader(compact_config()).load(path).audit

    assert audit.crossed_quote_fraction == pytest.approx(1 / 26)
    assert audit.outside_nbbo_fraction == pytest.approx(1 / 26)


def test_future_ticks_cannot_change_closed_15m_bar(tmp_path) -> None:
    frame = complete_tick_session()
    prefix_path = tmp_path / "prefix.txt"
    full_path = tmp_path / "full.txt"
    write_ticks(prefix_path, frame.iloc[:10])
    write_ticks(full_path, frame)
    loader = KibotTickLoader(compact_config(minimum_bucket_coverage=0.01))

    prefix = loader.load(prefix_path).bars
    complete = loader.load(full_path).bars

    pd.testing.assert_frame_equal(prefix, complete.iloc[: len(prefix)].reset_index(drop=True))


def test_invalid_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.txt"
    pd.DataFrame([["08/03/2026", "09:30:00", 100.0]]).to_csv(
        path,
        index=False,
        header=False,
    )

    with pytest.raises(ValueError, match="Schema Kibot"):
        KibotTickLoader(compact_config()).load(path)


def test_cli_contract_is_descriptive_only(tmp_path, capsys) -> None:
    path = tmp_path / "IVE.txt"
    write_ticks(path, complete_tick_session())

    arguments = parse_arguments(["--file", str(path)])
    assert arguments.interval == "15m"
    assert main(["--file", str(path), "--ticker", "IVE"]) == 0
    output = capsys.readouterr().out
    assert "KIBOT TICK DATA AUDIT" in output
    assert "Nessun segnale, ordine" in output
    assert "LIMITED" in output
