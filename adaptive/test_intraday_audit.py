from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adaptive.intraday_audit import (
    IntradayAuditConfig,
    IntradayReadiness,
    IntradayResearchAuditor,
)
from adaptive.run_intraday_audit import main, parse_arguments


def intraday_market(
    sessions: int = 12,
    *,
    timezone: bool = True,
) -> pd.DataFrame:
    session_dates = pd.bdate_range("2025-01-06", periods=sessions)
    timestamps: list[pd.Timestamp] = []
    for session_date in session_dates:
        start = pd.Timestamp(session_date.date()).tz_localize(
            "America/New_York"
        ) + pd.Timedelta(hours=9, minutes=30)
        timestamps.extend(pd.date_range(start, periods=26, freq="15min"))

    index = pd.DatetimeIndex(timestamps)
    if not timezone:
        index = index.tz_localize(None)
    path = 100.0 + np.linspace(0.0, 2.0, len(index))
    return pd.DataFrame(
        {
            "date": index,
            "open": path,
            "high": path + 0.15,
            "low": path - 0.15,
            "close": path + 0.02,
            "volume": np.full(len(index), 10_000.0),
        }
    )


def compact_config(**overrides) -> IntradayAuditConfig:
    values = {
        "minimum_feature_sessions": 5,
        "minimum_validation_sessions": 10,
    }
    values.update(overrides)
    return IntradayAuditConfig(**values)


def test_complete_timezone_aware_sessions_are_ready_for_feature_research() -> None:
    audit = IntradayResearchAuditor(compact_config()).audit(
        "spy", intraday_market()
    )

    assert audit.ticker == "SPY"
    assert audit.status is IntradayReadiness.READY
    assert audit.observed_sessions == 12
    assert audit.expected_bars_per_session == 26
    assert audit.median_session_coverage == 1.0
    assert audit.estimated_missing_bars == 0
    assert audit.median_cadence_minutes == 15.0
    assert audit.timezone_assumed is False
    assert audit.research_only is True


def test_short_history_is_limited_not_declared_validated() -> None:
    audit = IntradayResearchAuditor().audit("SPY", intraday_market(20))

    assert audit.status is IntradayReadiness.LIMITED
    assert any("multi-regime" in reason for reason in audit.reasons)


def test_naive_timestamps_are_assumed_new_york_and_marked_limited() -> None:
    audit = IntradayResearchAuditor(compact_config()).audit(
        "SPY", intraday_market(timezone=False)
    )

    assert audit.status is IntradayReadiness.LIMITED
    assert audit.timezone_assumed is True
    assert any("Timezone assente" in reason for reason in audit.reasons)


def test_missing_open_and_internal_bar_reduce_coverage() -> None:
    market = intraday_market()
    missing = market.drop(index=[0, 5]).reset_index(drop=True)
    audit = IntradayResearchAuditor(compact_config()).audit("SPY", missing)

    assert audit.status is IntradayReadiness.LIMITED
    assert audit.estimated_missing_bars == 2
    assert audit.gap_fraction == pytest.approx(2 / (12 * 26))
    assert any("Barre mancanti" in reason for reason in audit.reasons)


@pytest.mark.parametrize("defect", ["duplicate", "off_grid", "bad_ohlc", "weekend"])
def test_structural_defects_are_rejected(defect: str) -> None:
    market = intraday_market()
    if defect == "duplicate":
        market = pd.concat([market, market.iloc[[0]]], ignore_index=True)
    elif defect == "off_grid":
        market.loc[0, "date"] = market.loc[0, "date"] + pd.Timedelta(minutes=1)
    elif defect == "bad_ohlc":
        market.loc[0, "high"] = market.loc[0, "low"] - 1.0
    else:
        weekend = market.iloc[[0]].copy()
        weekend["date"] = pd.Timestamp(
            "2025-01-11 09:30", tz="America/New_York"
        )
        market = pd.concat([market, weekend], ignore_index=True)

    audit = IntradayResearchAuditor(compact_config()).audit("SPY", market)

    assert audit.status is IntradayReadiness.REJECTED


def test_extended_hours_are_reported_but_excluded_from_rth_metrics() -> None:
    market = intraday_market()
    extended = market.iloc[[0]].copy()
    extended["date"] = pd.Timestamp(
        "2025-01-06 08:00", tz="America/New_York"
    )
    market = pd.concat([market, extended], ignore_index=True)

    audit = IntradayResearchAuditor(compact_config()).audit("SPY", market)

    assert audit.status is IntradayReadiness.READY
    assert audit.extended_hours_rows == 1
    assert audit.regular_session_rows == 12 * 26


def test_market_errors_are_isolated_and_aggregate_status_is_limited() -> None:
    auditor = IntradayResearchAuditor(compact_config())
    report = auditor.audit_markets(
        {"SPY": intraday_market(), "BAD": pd.DataFrame({"date": [None]})}
    )

    assert report.status is IntradayReadiness.LIMITED
    assert report.assets["SPY"].status is IntradayReadiness.READY
    assert "BAD" in report.errors
    assert report.research_only is True


def test_cli_defaults_to_non_operational_15m_audit() -> None:
    arguments = parse_arguments([])

    assert arguments.period == "30d"
    assert arguments.interval == "15m"
    assert arguments.tickers == ["SPY", "QQQ", "IWM", "GLD", "TLT"]


def test_cli_prints_data_only_boundaries(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "adaptive.run_intraday_audit.load_markets",
        lambda pipeline, tickers, period, interval: (
            {"SPY": intraday_market(20)},
            {},
        ),
    )

    assert main(["--tickers", "SPY"]) == 0
    output = capsys.readouterr().out

    assert "15M DATA READINESS — RESEARCH-ONLY" in output
    assert "Nessun segnale, ordine, size, stop, leva o P&L" in output
    assert "ESITO AGGREGATO DATI: LIMITED" in output
    assert "validazione multi-regime" in output
