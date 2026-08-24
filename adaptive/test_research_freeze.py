from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive.intraday_report_audit import STANDARD_INTRADAY_REPORT_FILENAMES
from adaptive.local_intraday_data import (
    LocalIntradaySourceAudit,
    LocalSourceReadiness,
)
from adaptive.research_freeze import (
    RESEARCH_FREEZE_FILENAME,
    IntradayResearchSpecification,
    build_research_freeze,
    load_research_freeze,
    write_research_freeze,
)
from adaptive.run_research_freeze import parse_arguments


def market_frame(dates: list[str], *, adjustment: float = 0.0) -> pd.DataFrame:
    rows = []
    for session_number, date in enumerate(dates):
        for bar_number, timestamp in enumerate(
            pd.date_range(
                f"{date} 09:30",
                periods=2,
                freq="15min",
                tz="America/New_York",
            )
        ):
            value = 100.0 + session_number + bar_number * 0.1
            if session_number == 0 and bar_number == 0:
                value += adjustment
            rows.append(
                {
                    "date": timestamp,
                    "open": value,
                    "high": value + 0.2,
                    "low": value - 0.2,
                    "close": value + 0.05,
                    "volume": 1_000.0 + bar_number,
                }
            )
    return pd.DataFrame(rows)


def source_audit(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    ticker: str = "SPY",
    status: LocalSourceReadiness = LocalSourceReadiness.READY_DESCRIPTIVE,
) -> LocalIntradaySourceAudit:
    source = tmp_path / f"{ticker}_1min_sample.zip"
    source.write_bytes(f"source-{ticker}".encode("ascii"))
    sessions = frame["date"].dt.date.nunique()
    return LocalIntradaySourceAudit(
        ticker=ticker,
        source_path=str(source),
        raw_rows=len(frame) * 15,
        regular_session_rows=len(frame) * 15,
        observed_sessions=int(sessions),
        has_volume="volume" in frame,
        median_minute_coverage=1.0,
        minimum_minute_coverage=1.0,
        complete_bucket_fraction=1.0,
        early_close_sessions=0,
        estimated_missing_minutes=0,
        excluded_rows=0,
        output_bars=len(frame),
        status=status,
        reasons=("research-only",),
        first_timestamp=frame["date"].iloc[0],
        last_timestamp=frame["date"].iloc[-1],
    )


def report_directory(tmp_path: Path, name: str = "reports") -> Path:
    root = tmp_path / name
    root.mkdir()
    for number, filename in enumerate(STANDARD_INTRADAY_REPORT_FILENAMES):
        pd.DataFrame(
            {"ticker": ["SPY"], "descriptive_value": [float(number)]}
        ).to_csv(root / filename, index=False)
    return root


def build_freeze(
    tmp_path: Path,
    dates: list[str],
    *,
    adjustment: float = 0.0,
    specification: IntradayResearchSpecification | None = None,
    reports_name: str = "reports",
) -> dict[str, object]:
    frame = market_frame(dates, adjustment=adjustment)
    audit = source_audit(tmp_path, frame)
    return build_research_freeze(
        markets={"SPY": frame},
        source_audits={"SPY": audit},
        report_directory=report_directory(tmp_path, reports_name),
        specification=specification or IntradayResearchSpecification(),
    )


def test_freeze_is_deterministic_and_records_session_fingerprints(tmp_path) -> None:
    freeze = build_freeze(tmp_path, ["2023-01-03", "2023-01-04"])
    again = build_research_freeze(
        markets={"SPY": market_frame(["2023-01-03", "2023-01-04"])},
        source_audits={
            "SPY": source_audit(
                tmp_path,
                market_frame(["2023-01-03", "2023-01-04"]),
            )
        },
        report_directory=tmp_path / "reports",
        specification=IntradayResearchSpecification(),
    )

    assert freeze["freeze_id"] == again["freeze_id"]
    assert freeze["research_only"] is True
    assert freeze["report_audit"]["state"] == "PASS"
    assert list(freeze["sources"][0]["session_fingerprints"]) == [
        "2023-01-03",
        "2023-01-04",
    ]
    assert len(freeze["implementation"]["sha256"]) == 64


def test_freeze_round_trip_and_tamper_detection(tmp_path) -> None:
    freeze = build_freeze(tmp_path, ["2023-01-03", "2023-01-04"])
    destination = write_research_freeze(freeze, tmp_path / RESEARCH_FREEZE_FILENAME)

    assert load_research_freeze(destination)["freeze_id"] == freeze["freeze_id"]
    write_research_freeze(freeze, destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["specification"]["neighbors"] = 99
    destination.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Freeze alterato"):
        load_research_freeze(destination)


def test_existing_different_freeze_cannot_be_overwritten(tmp_path) -> None:
    first = build_freeze(tmp_path, ["2023-01-03"], reports_name="reports_a")
    second = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-04"],
        reports_name="reports_b",
    )
    destination = write_research_freeze(first, tmp_path / "freeze.json")

    with pytest.raises(FileExistsError, match="nuova directory"):
        write_research_freeze(second, destination)


def test_nonfinite_market_value_is_rejected(tmp_path) -> None:
    frame = market_frame(["2023-01-03"])
    frame.loc[0, "close"] = np.inf
    audit = source_audit(tmp_path, frame)

    with pytest.raises(ValueError, match="non finiti"):
        build_research_freeze(
            markets={"SPY": frame},
            source_audits={"SPY": audit},
            report_directory=report_directory(tmp_path),
            specification=IntradayResearchSpecification(),
        )


def test_report_and_source_universes_must_match(tmp_path) -> None:
    frame = market_frame(["2023-01-03"])
    audit = source_audit(tmp_path, frame)
    reports = report_directory(tmp_path)
    first_report = reports / STANDARD_INTRADAY_REPORT_FILENAMES[0]
    pd.DataFrame(
        {"ticker": ["SPY", "QQQ"], "descriptive_value": [1.0, 2.0]}
    ).to_csv(first_report, index=False)

    with pytest.raises(ValueError, match="Universo report e sorgenti"):
        build_research_freeze(
            markets={"SPY": frame},
            source_audits={"SPY": audit},
            report_directory=reports,
            specification=IntradayResearchSpecification(),
        )


def test_freeze_cli_defaults_to_report_directory(tmp_path) -> None:
    arguments = parse_arguments(
        ["--data-dir", "C:/data", "--report-dir", "reports/v16"]
    )

    assert Path(arguments.output) == (
        Path("reports/v16") / "intraday_research_freeze.json"
    )
    assert arguments.neighbors == 8
    assert arguments.history_limit == 4_000
