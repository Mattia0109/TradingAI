from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive.local_intraday_data import (
    FirstRateIntradayLoader,
    LocalIntradayDataConfig,
    LocalSourceReadiness,
    discover_first_rate_archives,
)
from adaptive.run_local_intraday_research import (
    _standard_outputs_share_directory,
    parse_arguments,
)


def minute_session(date: str, minutes: int = 390) -> pd.DataFrame:
    start = pd.Timestamp(f"{date} 09:30", tz="America/New_York")
    timestamps = pd.date_range(start, periods=minutes, freq="1min")
    sequence = np.arange(minutes, dtype=float)
    close = 100.0 + sequence * 0.01
    return pd.DataFrame(
        {
            "timestamp": timestamps.tz_localize(None),
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.full(minutes, 100.0),
        }
    )


def write_zip(
    path: Path,
    frame: pd.DataFrame,
    *,
    header: bool = True,
    member: str = "sample.csv",
) -> None:
    payload = frame.to_csv(index=False, header=header)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)


def compact_config(**overrides) -> LocalIntradayDataConfig:
    values = {"minimum_descriptive_sessions": 1}
    values.update(overrides)
    return LocalIntradayDataConfig(**values)


def test_resampling_uses_ohlcv_contract_and_complete_early_close(tmp_path) -> None:
    regular = minute_session("2023-06-30")
    # Il file puo' contenere righe dopo la chiusura ufficiale; il calendario
    # deve escluderle prima del resampling RTH.
    early = minute_session("2023-07-03", minutes=390)
    archive = tmp_path / "SPY_1min_sample_firstratedata.zip"
    write_zip(archive, pd.concat([regular, early], ignore_index=True))

    result = FirstRateIntradayLoader(compact_config()).load(archive)

    assert result.ticker == "SPY"
    assert len(result.bars) == 26 + 14
    assert result.audit.observed_sessions == 2
    assert result.audit.early_close_sessions == 1
    assert result.audit.excluded_rows == 180
    assert result.audit.median_minute_coverage == 1.0
    assert result.audit.complete_bucket_fraction == 1.0
    assert result.audit.estimated_missing_minutes == 0
    assert result.audit.status is LocalSourceReadiness.READY_DESCRIPTIVE
    first = result.bars.iloc[0]
    assert first["open"] == pytest.approx(99.99)
    assert first["high"] == pytest.approx(100.19)
    assert first["low"] == pytest.approx(99.95)
    assert first["close"] == pytest.approx(100.14)
    assert first["volume"] == pytest.approx(1_500.0)


def test_headerless_price_only_index_is_kept_separate(tmp_path) -> None:
    frame = minute_session("2023-08-01").drop(columns="volume")
    archive = tmp_path / "NDX_1min_sample_firstratedata.zip"
    write_zip(archive, frame, header=False, member="NDX.txt")

    result = FirstRateIntradayLoader(compact_config()).load(archive)

    assert result.audit.status is LocalSourceReadiness.PRICE_ONLY
    assert result.audit.has_volume is False
    assert "volume" not in result.bars
    assert "CMF" in result.audit.reasons[0]


def test_incomplete_bucket_is_dropped_and_never_filled(tmp_path) -> None:
    frame = minute_session("2023-08-01").drop(index=[8]).reset_index(drop=True)
    archive = tmp_path / "SPY_1min_sample_firstratedata.zip"
    write_zip(archive, frame)

    result = FirstRateIntradayLoader(
        compact_config(minimum_complete_bucket_fraction=1.0)
    ).load(archive)

    assert len(result.bars) == 25
    assert result.audit.estimated_missing_minutes == 1
    assert result.audit.complete_bucket_fraction == pytest.approx(25 / 26)
    assert result.audit.status is LocalSourceReadiness.LIMITED


def test_future_minutes_do_not_change_already_closed_buckets(tmp_path) -> None:
    full = minute_session("2023-08-01")
    prefix_archive = tmp_path / "SPY_1min_prefix.zip"
    full_archive = tmp_path / "SPY_1min_full.zip"
    write_zip(prefix_archive, full.iloc[:150])
    write_zip(full_archive, full)
    loader = FirstRateIntradayLoader(compact_config())

    prefix = loader.load(prefix_archive, ticker="SPY").bars
    complete = loader.load(full_archive, ticker="SPY").bars

    pd.testing.assert_frame_equal(
        prefix.reset_index(drop=True),
        complete.iloc[: len(prefix)].reset_index(drop=True),
    )


def test_duplicate_timestamp_is_rejected(tmp_path) -> None:
    frame = minute_session("2023-08-01")
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    archive = tmp_path / "SPY_1min_sample_firstratedata.zip"
    write_zip(archive, frame)

    with pytest.raises(ValueError, match="duplicati"):
        FirstRateIntradayLoader(compact_config()).load(archive)


def test_unsafe_zip_member_is_rejected(tmp_path) -> None:
    archive = tmp_path / "SPY_1min_sample_firstratedata.zip"
    write_zip(archive, minute_session("2023-08-01"), member="../escape.csv")

    with pytest.raises(ValueError, match="non sicuro"):
        FirstRateIntradayLoader(compact_config()).load(archive)


def test_discovery_is_deterministic_and_filterable(tmp_path) -> None:
    for ticker in ("SPY", "QQQ"):
        write_zip(
            tmp_path / f"{ticker}_1min_sample_firstratedata.zip",
            minute_session("2023-08-01"),
        )

    discovered = discover_first_rate_archives(tmp_path, ["qqq"])

    assert list(discovered) == ["QQQ"]


def test_local_research_cli_requires_data_dir_and_stays_on_15m() -> None:
    arguments = parse_arguments(["--data-dir", "C:/data"])

    assert arguments.data_dir == "C:/data"
    assert arguments.interval == "15m"
    assert arguments.sessions_per_block == 30
    assert arguments.minimum_complete_blocks == 4
    assert arguments.output_dir is None
    assert arguments.matrix_output is None
    assert arguments.regime_atlas_output is None
    assert arguments.regime_transition_output is None
    assert arguments.context_stability_output is None
    assert arguments.lorentzian_stability_output is None
    assert arguments.lorentzian_soft_surface_output is None
    assert arguments.lorentzian_soft_blocks_output is None
    assert arguments.lorentzian_soft_stability_output is None
    assert arguments.cross_asset_dependence_output is None
    assert arguments.cross_asset_block_breadth_output is None
    assert arguments.cross_asset_breadth_output is None
    assert arguments.cross_asset_factor_loadings_output is None
    assert arguments.cross_asset_factor_blocks_output is None
    assert arguments.cross_asset_factor_summary_output is None
    assert arguments.cross_asset_residual_blocks_output is None
    assert arguments.cross_asset_residual_pairs_output is None
    assert arguments.cross_asset_proxy_clusters_output is None
    assert arguments.cross_asset_proxy_sensitivity_output is None
    assert arguments.cross_asset_proxy_blocks_output is None
    assert arguments.cross_asset_proxy_block_stability_output is None
    assert arguments.cross_asset_proxy_invariance_output is None
    assert arguments.cross_asset_phase_factor_loadings_output is None
    assert arguments.cross_asset_phase_factor_blocks_output is None
    assert arguments.cross_asset_phase_factor_summary_output is None
    assert arguments.cross_asset_phase_factor_contrast_output is None
    assert arguments.cross_asset_phase_residual_blocks_output is None
    assert arguments.cross_asset_phase_residual_pairs_output is None
    assert arguments.cross_asset_phase_residual_contrast_output is None
    assert arguments.context_minimum_observations == 60
    assert arguments.context_minimum_sessions == 10
    assert arguments.context_minimum_complete_blocks == 3


def test_output_directory_populates_every_standard_report_path() -> None:
    arguments = parse_arguments(
        ["--data-dir", "C:/data", "--output-dir", "reports/v15"]
    )

    assert arguments.matrix_output == "reports/v15/intraday_feature_stability_matrix.csv"
    assert arguments.cross_asset_phase_factor_contrast_output == (
        "reports/v15/intraday_phase_factor_contrast.csv"
    )
    assert arguments.cross_asset_phase_residual_blocks_output == (
        "reports/v15/intraday_phase_residual_pair_blocks.csv"
    )
    assert arguments.cross_asset_phase_residual_pairs_output == (
        "reports/v15/intraday_phase_residual_pairs.csv"
    )
    assert arguments.cross_asset_phase_residual_contrast_output == (
        "reports/v15/intraday_phase_residual_contrast.csv"
    )


def test_output_directory_normalizes_windows_separators_portably() -> None:
    arguments = parse_arguments(
        ["--data-dir", "C:/data", "--output-dir", r"reports\v15"]
    )

    assert arguments.output_dir == "reports/v15"
    assert arguments.matrix_output == (
        "reports/v15/intraday_feature_stability_matrix.csv"
    )
    assert _standard_outputs_share_directory(arguments) is True


def test_explicit_report_path_overrides_output_directory_default() -> None:
    arguments = parse_arguments(
        [
            "--data-dir",
            "C:/data",
            "--output-dir",
            "reports/v15",
            "--matrix-output",
            "custom/matrix.csv",
        ]
    )

    assert arguments.matrix_output == "custom/matrix.csv"
    assert arguments.regime_atlas_output == "reports/v15/intraday_regime_atlas.csv"
    assert _standard_outputs_share_directory(arguments) is False


def test_standard_output_directory_is_auditable_as_one_unit() -> None:
    arguments = parse_arguments(
        ["--data-dir", "C:/data", "--output-dir", "reports/v15"]
    )

    assert _standard_outputs_share_directory(arguments) is True
