"""Dipendenza descrittiva contemporanea tra asset intraday.

Il modulo misura quanta parte dell'evidenza osservata e' condivisa tra le
serie. Usa soltanto variazioni di close gia' concluse nella stessa seduta e
feature contemporanee. Non usa outcome futuri e non produce direzioni,
previsioni, operazioni, size, stop, leva o P&L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PAIR_COLUMNS = (
    "ticker_a",
    "ticker_b",
    "source_status_a",
    "source_status_b",
    "data_mode_a",
    "data_mode_b",
    "observations",
    "sessions",
    "eligible_blocks",
    "close_change_correlation",
    "close_change_spearman",
    "absolute_change_correlation",
    "median_block_close_change_correlation",
    "minimum_block_close_change_correlation",
    "maximum_block_close_change_correlation",
    "maximum_block_correlation_change",
    "choppiness_spearman",
    "squeeze_momentum_spearman",
    "cmf_spearman",
    "chop_state_agreement",
    "squeeze_state_agreement",
    "joint_state_agreement",
    "joint_context_nmi",
    "dependence_state",
)

BLOCK_BREADTH_COLUMNS = (
    "scope",
    "block_id",
    "block_start_session",
    "block_end_session",
    "block_sessions",
    "block_complete",
    "assets",
    "asset_list",
    "observations",
    "effective_asset_count",
    "effective_asset_fraction",
    "dominant_component_share",
    "median_pair_absolute_correlation",
    "p90_pair_absolute_correlation",
    "breadth_state",
)

BREADTH_COLUMNS = (
    "scope",
    "assets",
    "asset_list",
    "observations",
    "sessions",
    "eligible_blocks",
    "effective_asset_count",
    "effective_asset_fraction",
    "dominant_component_share",
    "median_pair_absolute_correlation",
    "p90_pair_absolute_correlation",
    "minimum_block_effective_asset_count",
    "median_block_effective_asset_count",
    "maximum_block_dominant_component_share",
    "minimum_block_observations",
    "breadth_state",
)

FACTOR_LOADING_COLUMNS = (
    "scope",
    "ticker",
    "assets",
    "observations",
    "loading",
    "absolute_loading",
    "loading_share",
    "common_variance_share",
    "absolute_loading_rank",
)

FACTOR_BLOCK_COLUMNS = (
    "scope",
    "block_id",
    "block_start_session",
    "block_end_session",
    "block_sessions",
    "block_complete",
    "assets",
    "asset_list",
    "observations",
    "common_factor_share",
    "raw_effective_asset_count",
    "residual_effective_asset_count",
    "raw_median_pair_absolute_correlation",
    "residual_median_pair_absolute_correlation",
    "loading_cosine_previous_block",
    "block_state",
)

FACTOR_SUMMARY_COLUMNS = (
    "scope",
    "assets",
    "asset_list",
    "observations",
    "sessions",
    "eligible_blocks",
    "common_factor_share",
    "raw_effective_asset_count",
    "residual_effective_asset_count",
    "raw_median_pair_absolute_correlation",
    "residual_median_pair_absolute_correlation",
    "median_block_common_factor_share",
    "maximum_adjacent_factor_share_change",
    "median_adjacent_loading_cosine",
    "minimum_adjacent_loading_cosine",
    "minimum_block_residual_effective_asset_count",
    "median_block_residual_effective_asset_count",
    "common_mode_state",
)

RESIDUAL_PAIR_BLOCK_COLUMNS = (
    "scope",
    "block_id",
    "block_start_session",
    "block_end_session",
    "block_sessions",
    "block_complete",
    "ticker_a",
    "ticker_b",
    "observations",
    "residual_correlation",
    "block_state",
)

RESIDUAL_PAIR_COLUMNS = (
    "scope",
    "ticker_a",
    "ticker_b",
    "observations",
    "eligible_blocks",
    "raw_correlation",
    "minimum_block_raw_correlation",
    "maximum_block_raw_correlation_change",
    "residual_correlation",
    "median_block_residual_correlation",
    "minimum_block_residual_correlation",
    "maximum_block_residual_correlation",
    "maximum_block_residual_correlation_change",
    "stable_proxy_link",
    "residual_pair_state",
)

PROXY_CLUSTER_COLUMNS = (
    "scope",
    "cluster_id",
    "cluster_members",
    "member_count",
    "ticker",
    "source_status",
    "data_mode",
    "stable_link_count",
    "cluster_state",
)

PROXY_SENSITIVITY_COLUMNS = (
    "scope",
    "scenario_type",
    "scenario_id",
    "retained_assets",
    "omitted_assets",
    "assets",
    "observations",
    "effective_asset_count",
    "effective_asset_fraction",
    "common_factor_share",
    "median_pair_absolute_correlation",
    "residual_effective_asset_count",
    "residual_median_pair_absolute_correlation",
    "effective_fraction_change_vs_baseline",
    "residual_effective_count_change_vs_baseline",
    "scenario_state",
)

PROXY_BLOCK_SENSITIVITY_COLUMNS = (
    "scope",
    "block_id",
    "block_start_session",
    "block_end_session",
    "block_sessions",
    "block_complete",
    "scenario_type",
    "scenario_id",
    "retained_assets",
    "omitted_assets",
    "assets",
    "observations",
    "effective_asset_count",
    "effective_asset_fraction",
    "common_factor_share",
    "median_pair_absolute_correlation",
    "residual_effective_asset_count",
    "residual_median_pair_absolute_correlation",
    "effective_fraction_change_vs_block_baseline",
    "residual_effective_count_change_vs_block_baseline",
    "scenario_state",
    "block_state",
)

PROXY_BLOCK_STABILITY_COLUMNS = (
    "scope",
    "scenario_type",
    "scenario_id",
    "retained_assets",
    "omitted_assets",
    "assets",
    "eligible_blocks",
    "minimum_block_observations",
    "minimum_effective_asset_count",
    "median_effective_asset_count",
    "maximum_effective_asset_count",
    "effective_asset_count_range",
    "minimum_common_factor_share",
    "median_common_factor_share",
    "maximum_common_factor_share",
    "common_factor_share_range",
    "minimum_residual_effective_asset_count",
    "median_residual_effective_asset_count",
    "maximum_residual_effective_asset_count",
    "residual_effective_asset_count_range",
    "minimum_median_pair_absolute_correlation",
    "median_median_pair_absolute_correlation",
    "maximum_median_pair_absolute_correlation",
    "median_pair_absolute_correlation_range",
    "minimum_residual_median_pair_absolute_correlation",
    "median_residual_median_pair_absolute_correlation",
    "maximum_residual_median_pair_absolute_correlation",
    "residual_median_pair_absolute_correlation_range",
    "scenario_state",
    "block_stability_state",
)

PROXY_REPRESENTATIVE_INVARIANCE_COLUMNS = (
    "scope",
    "block_id",
    "block_start_session",
    "block_end_session",
    "block_sessions",
    "block_complete",
    "expected_scenarios",
    "observed_scenarios",
    "effective_asset_count_range",
    "effective_asset_fraction_range",
    "common_factor_share_range",
    "median_pair_absolute_correlation_range",
    "residual_effective_asset_count_range",
    "residual_median_pair_absolute_correlation_range",
    "invariance_state",
)

SESSION_PHASES = ("OPEN", "MID_SESSION", "CLOSE")

PHASE_FACTOR_LOADING_COLUMNS = (
    "scope",
    "session_phase",
    "ticker",
    "assets",
    "observations",
    "loading",
    "absolute_loading",
    "loading_share",
    "common_variance_share",
    "absolute_loading_rank",
)

PHASE_FACTOR_BLOCK_COLUMNS = (
    "scope",
    "session_phase",
    "block_id",
    "block_start_session",
    "block_end_session",
    "block_sessions",
    "block_complete",
    "assets",
    "asset_list",
    "observations",
    "common_factor_share",
    "raw_effective_asset_count",
    "residual_effective_asset_count",
    "raw_median_pair_absolute_correlation",
    "residual_median_pair_absolute_correlation",
    "loading_cosine_previous_block",
    "block_state",
)

PHASE_FACTOR_SUMMARY_COLUMNS = (
    "scope",
    "session_phase",
    "assets",
    "asset_list",
    "observations",
    "sessions",
    "eligible_blocks",
    "common_factor_share",
    "raw_effective_asset_count",
    "residual_effective_asset_count",
    "raw_median_pair_absolute_correlation",
    "residual_median_pair_absolute_correlation",
    "median_block_common_factor_share",
    "maximum_adjacent_factor_share_change",
    "median_adjacent_loading_cosine",
    "minimum_adjacent_loading_cosine",
    "minimum_block_residual_effective_asset_count",
    "median_block_residual_effective_asset_count",
    "common_mode_state",
)

PHASE_FACTOR_CONTRAST_COLUMNS = (
    "scope",
    "assets",
    "asset_list",
    "phases_available",
    "minimum_phase_observations",
    "minimum_eligible_blocks",
    "minimum_common_factor_share",
    "median_common_factor_share",
    "maximum_common_factor_share",
    "common_factor_share_range",
    "maximum_common_factor_session_phase",
    "minimum_common_factor_session_phase",
    "minimum_raw_effective_asset_count",
    "maximum_raw_effective_asset_count",
    "raw_effective_asset_count_range",
    "raw_effective_fraction_range",
    "maximum_raw_effective_asset_session_phase",
    "minimum_residual_effective_asset_count",
    "maximum_residual_effective_asset_count",
    "residual_effective_asset_count_range",
    "residual_effective_fraction_range",
    "maximum_residual_effective_asset_session_phase",
    "minimum_raw_median_pair_absolute_correlation",
    "maximum_raw_median_pair_absolute_correlation",
    "raw_median_pair_absolute_correlation_range",
    "phase_structure_state",
)

PHASE_RESIDUAL_PAIR_BLOCK_COLUMNS = (
    "scope",
    "session_phase",
    "block_id",
    "block_start_session",
    "block_end_session",
    "block_sessions",
    "block_complete",
    "ticker_a",
    "ticker_b",
    "observations",
    "raw_correlation",
    "residual_correlation",
    "block_state",
)

PHASE_RESIDUAL_PAIR_COLUMNS = (
    "scope",
    "session_phase",
    "ticker_a",
    "ticker_b",
    "observations",
    "sessions",
    "eligible_blocks",
    "raw_correlation",
    "minimum_block_raw_correlation",
    "maximum_block_raw_correlation",
    "maximum_block_raw_correlation_change",
    "residual_correlation",
    "median_block_residual_correlation",
    "minimum_block_residual_correlation",
    "maximum_block_residual_correlation",
    "maximum_block_residual_correlation_change",
    "stable_proxy_link",
    "residual_pair_state",
)

PHASE_RESIDUAL_CONTRAST_COLUMNS = (
    "scope",
    "ticker_a",
    "ticker_b",
    "phases_available",
    "minimum_phase_observations",
    "minimum_eligible_blocks",
    "minimum_raw_correlation",
    "maximum_raw_correlation",
    "raw_correlation_range",
    "maximum_absolute_raw_correlation",
    "maximum_absolute_raw_session_phase",
    "minimum_residual_correlation",
    "maximum_residual_correlation",
    "residual_correlation_range",
    "minimum_absolute_residual_correlation",
    "maximum_absolute_residual_correlation",
    "absolute_residual_correlation_range",
    "maximum_absolute_residual_session_phase",
    "residual_sign_changes",
    "stable_proxy_phases",
    "phase_state_count",
    "phase_residual_state",
)


@dataclass(frozen=True)
class IntradayCrossAssetDependenceConfig:
    """Contratto e soglie dichiarate per la dipendenza descrittiva."""

    market_timezone: str = "America/New_York"
    interval_minutes: int = 15
    sessions_per_block: int = 30
    minimum_pair_observations: int = 500
    minimum_pair_sessions: int = 60
    minimum_block_observations: int = 180
    minimum_complete_blocks: int = 3
    moderate_absolute_correlation: float = 0.40
    high_absolute_correlation: float = 0.75
    moderate_context_nmi: float = 0.20
    high_context_nmi: float = 0.50
    concentrated_effective_fraction: float = 0.40
    distributed_effective_fraction: float = 0.70
    concentrated_dominant_share: float = 0.60
    distributed_dominant_share: float = 0.40
    stable_minimum_loading_cosine: float = 0.85
    variable_minimum_loading_cosine: float = 0.65
    stable_maximum_factor_share_change: float = 0.15
    variable_maximum_factor_share_change: float = 0.30
    proxy_minimum_raw_correlation: float = 0.95
    proxy_minimum_raw_block_correlation: float = 0.90
    proxy_maximum_raw_block_change: float = 0.05
    maximum_proxy_combinations: int = 256
    minimum_phase_block_observations: int = 60
    moderate_phase_factor_share_range: float = 0.05
    high_phase_factor_share_range: float = 0.10
    moderate_phase_effective_fraction_range: float = 0.10
    high_phase_effective_fraction_range: float = 0.20
    moderate_phase_residual_absolute_range: float = 0.15
    high_phase_residual_absolute_range: float = 0.30
    special_context_tickers: tuple[str, ...] = ("VXX",)

    def __post_init__(self) -> None:
        ZoneInfo(self.market_timezone)
        for name in (
            "interval_minutes",
            "sessions_per_block",
            "minimum_pair_observations",
            "minimum_pair_sessions",
            "minimum_block_observations",
            "minimum_complete_blocks",
            "maximum_proxy_combinations",
            "minimum_phase_block_observations",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} deve essere positivo.")
        pairs = (
            (
                self.moderate_absolute_correlation,
                self.high_absolute_correlation,
                "absolute_correlation",
            ),
            (
                self.moderate_context_nmi,
                self.high_context_nmi,
                "context_nmi",
            ),
            (
                self.concentrated_effective_fraction,
                self.distributed_effective_fraction,
                "effective_fraction",
            ),
        )
        for lower, upper, name in pairs:
            if not 0.0 <= float(lower) < float(upper) <= 1.0:
                raise ValueError(f"Soglie {name} non valide.")
        if not (
            0.0
            < float(self.distributed_dominant_share)
            < float(self.concentrated_dominant_share)
            <= 1.0
        ):
            raise ValueError("Soglie dominant_share non valide.")
        if not (
            0.0
            <= float(self.variable_minimum_loading_cosine)
            < float(self.stable_minimum_loading_cosine)
            <= 1.0
        ):
            raise ValueError("Soglie loading_cosine non valide.")
        if not (
            0.0
            <= float(self.stable_maximum_factor_share_change)
            < float(self.variable_maximum_factor_share_change)
            <= 1.0
        ):
            raise ValueError("Soglie factor_share_change non valide.")
        for name in (
            "proxy_minimum_raw_correlation",
            "proxy_minimum_raw_block_correlation",
            "proxy_maximum_raw_block_change",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"Soglia {name} non valida.")
        if (
            float(self.proxy_minimum_raw_block_correlation)
            > float(self.proxy_minimum_raw_correlation)
        ):
            raise ValueError("Soglie proxy raw non valide.")
        phase_pairs = (
            (
                self.moderate_phase_factor_share_range,
                self.high_phase_factor_share_range,
                "phase_factor_share_range",
            ),
            (
                self.moderate_phase_effective_fraction_range,
                self.high_phase_effective_fraction_range,
                "phase_effective_fraction_range",
            ),
            (
                self.moderate_phase_residual_absolute_range,
                self.high_phase_residual_absolute_range,
                "phase_residual_absolute_range",
            ),
        )
        for lower, upper, name in phase_pairs:
            if not 0.0 <= float(lower) < float(upper) <= 1.0:
                raise ValueError(f"Soglie {name} non valide.")


@dataclass(frozen=True)
class IntradayCrossAssetDependenceReport:
    """Tabelle separate dagli oggetti operativi del progetto."""

    pairs: pd.DataFrame
    block_breadth: pd.DataFrame
    breadth: pd.DataFrame
    factor_loadings: pd.DataFrame
    factor_blocks: pd.DataFrame
    factor_summary: pd.DataFrame
    residual_pair_blocks: pd.DataFrame
    residual_pairs: pd.DataFrame
    proxy_clusters: pd.DataFrame
    proxy_sensitivity: pd.DataFrame
    proxy_block_sensitivity: pd.DataFrame
    proxy_block_stability: pd.DataFrame
    proxy_representative_invariance: pd.DataFrame
    phase_factor_loadings: pd.DataFrame
    phase_factor_blocks: pd.DataFrame
    phase_factor_summary: pd.DataFrame
    phase_factor_contrast: pd.DataFrame
    phase_residual_pair_blocks: pd.DataFrame
    phase_residual_pairs: pd.DataFrame
    phase_residual_contrast: pd.DataFrame
    caveats: tuple[str, ...]
    research_only: bool = True


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _audit_metadata(audit: object) -> tuple[str, str]:
    if audit is None:
        raise ValueError("Audit sorgente mancante.")
    status = getattr(audit, "status", None)
    status_value = getattr(status, "value", status)
    if status_value is None:
        raise ValueError("Stato sorgente mancante.")
    has_volume = bool(getattr(audit, "has_volume", False))
    return str(status_value), "OHLCV" if has_volume else "PRICE"


def _localized_index(values: object, timezone: str) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values, errors="raise", utc=True)
    index = pd.DatetimeIndex(parsed).tz_convert(timezone)
    if index.hasnans or index.duplicated().any():
        raise ValueError("Timestamp mancanti o duplicati.")
    return index


def _feature_index(values: pd.DataFrame, timezone: str) -> pd.DatetimeIndex:
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("Le feature devono avere un DatetimeIndex.")
    if values.index.tz is None:
        raise ValueError("Le feature devono avere una timezone dichiarata.")
    index = values.index.tz_convert(timezone)
    if index.hasnans or index.duplicated().any():
        raise ValueError("Timestamp feature mancanti o duplicati.")
    return index


def _prepare_asset(
    ticker: str,
    market: pd.DataFrame,
    features: pd.DataFrame,
    audit: object,
    config: IntradayCrossAssetDependenceConfig,
) -> pd.DataFrame:
    symbol = str(ticker).upper().strip()
    if not symbol:
        raise ValueError("ticker non puo' essere vuoto.")
    if not isinstance(market, pd.DataFrame) or market.empty:
        raise ValueError(f"Mercato mancante per {symbol}.")
    required_market = {"date", "close"}
    missing_market = sorted(required_market.difference(market.columns))
    if missing_market:
        raise ValueError(f"Colonne mercato mancanti per {symbol}: {missing_market}.")
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError(f"Feature mancanti per {symbol}.")
    required_features = {
        "choppiness",
        "squeeze_momentum_pct_close",
        "chop_segment",
        "squeeze_state",
    }
    missing_features = sorted(required_features.difference(features.columns))
    if missing_features:
        raise ValueError(
            f"Feature descrittive mancanti per {symbol}: {missing_features}."
        )

    market_frame = market.loc[:, ["date", "close"]].copy()
    market_frame.index = _localized_index(market_frame.pop("date"), config.market_timezone)
    market_frame = market_frame.sort_index()
    market_frame["close"] = pd.to_numeric(
        market_frame["close"], errors="coerce"
    )
    market_frame.loc[market_frame["close"] <= 0.0, "close"] = np.nan

    feature_frame = features.copy()
    feature_frame.index = _feature_index(feature_frame, config.market_timezone)
    feature_columns = [
        "choppiness",
        "squeeze_momentum_pct_close",
        "chop_segment",
        "squeeze_state",
    ]
    if "cmf" in feature_frame.columns:
        feature_columns.append("cmf")
    frame = market_frame.join(feature_frame.loc[:, feature_columns], how="inner")
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError(f"Nessun timestamp comune per {symbol}.")

    local = frame.index.tz_convert(config.market_timezone)
    session = pd.Series(
        [value.isoformat() for value in local.date],
        index=frame.index,
        dtype="object",
    )
    elapsed = pd.Series(frame.index, index=frame.index).diff()
    consecutive = session.eq(session.shift(1)) & elapsed.eq(
        pd.Timedelta(minutes=config.interval_minutes)
    )
    log_close = np.log(frame["close"])
    frame["intraday_log_close_change"] = log_close.diff().where(consecutive)
    frame["intraday_absolute_log_close_change"] = frame[
        "intraday_log_close_change"
    ].abs()
    frame["session_date"] = session
    minutes = local.hour * 60 + local.minute
    phase = pd.Series(
        np.select(
            [minutes < 10 * 60 + 30, minutes >= 15 * 60],
            ["OPEN", "CLOSE"],
            default="MID_SESSION",
        ),
        index=frame.index,
        dtype="object",
    )
    frame["session_phase"] = phase
    same_phase = phase.eq(phase.shift(1))
    frame["phase_consistent_log_close_change"] = frame[
        "intraday_log_close_change"
    ].where(same_phase)
    valid_chop = frame["chop_segment"].notna() & frame["chop_segment"].ne(
        "INSUFFICIENT"
    )
    valid_squeeze = frame["squeeze_state"].notna() & frame[
        "squeeze_state"
    ].ne("INSUFFICIENT")
    frame["joint_context"] = pd.Series(pd.NA, index=frame.index, dtype="object")
    valid_context = valid_chop & valid_squeeze
    frame.loc[valid_context, "joint_context"] = (
        frame.loc[valid_context, "chop_segment"].astype(str)
        + "|"
        + frame.loc[valid_context, "squeeze_state"].astype(str)
    )
    source_status, data_mode = _audit_metadata(audit)
    frame["ticker"] = symbol
    frame["source_status"] = source_status
    frame["data_mode"] = data_mode
    return frame


def _session_blocks(
    prepared: Mapping[str, pd.DataFrame],
    config: IntradayCrossAssetDependenceConfig,
) -> tuple[dict[str, int], pd.DataFrame]:
    sessions = sorted(
        {
            str(value)
            for frame in prepared.values()
            for value in frame["session_date"].dropna().unique()
        }
    )
    mapping = {
        session: position // int(config.sessions_per_block)
        for position, session in enumerate(sessions)
    }
    rows = []
    for block_id in sorted(set(mapping.values())):
        block_sessions = [
            session for session in sessions if mapping[session] == block_id
        ]
        rows.append(
            {
                "block_id": int(block_id),
                "block_start_session": block_sessions[0],
                "block_end_session": block_sessions[-1],
                "block_sessions": len(block_sessions),
                "block_complete": len(block_sessions)
                == int(config.sessions_per_block),
            }
        )
    return mapping, pd.DataFrame(rows)


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    values = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(values) < 3 or values.iloc[:, 0].nunique() < 2 or values.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(values.iloc[:, 0].corr(values.iloc[:, 1], method=method))


def _agreement(left: pd.Series, right: pd.Series) -> float:
    values = pd.concat([left, right], axis=1).dropna()
    if values.empty:
        return math.nan
    return float(values.iloc[:, 0].eq(values.iloc[:, 1]).mean())


def _normalized_mutual_information(left: pd.Series, right: pd.Series) -> float:
    values = pd.concat([left, right], axis=1).dropna()
    if values.empty:
        return math.nan
    table = pd.crosstab(values.iloc[:, 0], values.iloc[:, 1], normalize=True)
    probabilities = table.to_numpy(dtype=float)
    row_probability = probabilities.sum(axis=1, keepdims=True)
    column_probability = probabilities.sum(axis=0, keepdims=True)
    expected = row_probability @ column_probability
    valid = probabilities > 0.0
    mutual_information = float(
        np.sum(probabilities[valid] * np.log(probabilities[valid] / expected[valid]))
    )
    row_values = row_probability.ravel()
    column_values = column_probability.ravel()
    row_entropy = float(-np.sum(row_values[row_values > 0.0] * np.log(row_values[row_values > 0.0])))
    column_entropy = float(
        -np.sum(column_values[column_values > 0.0] * np.log(column_values[column_values > 0.0]))
    )
    denominator = math.sqrt(row_entropy * column_entropy)
    if denominator <= 0.0:
        return math.nan
    return float(np.clip(mutual_information / denominator, 0.0, 1.0))


def _max_adjacent_change(values: list[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size < 2:
        return math.nan
    return float(np.max(np.abs(np.diff(finite))))


def _dependence_state(
    observations: int,
    sessions: int,
    eligible_blocks: int,
    correlation: float,
    context_nmi: float,
    config: IntradayCrossAssetDependenceConfig,
) -> str:
    if (
        observations < int(config.minimum_pair_observations)
        or sessions < int(config.minimum_pair_sessions)
        or eligible_blocks < int(config.minimum_complete_blocks)
    ):
        return "INSUFFICIENT_OVERLAP"
    absolute_correlation = abs(correlation) if math.isfinite(correlation) else 0.0
    nmi = context_nmi if math.isfinite(context_nmi) else 0.0
    if (
        absolute_correlation >= float(config.high_absolute_correlation)
        or nmi >= float(config.high_context_nmi)
    ):
        return "HIGH_DEPENDENCE"
    if (
        absolute_correlation >= float(config.moderate_absolute_correlation)
        or nmi >= float(config.moderate_context_nmi)
    ):
        return "MODERATE_DEPENDENCE"
    return "LOW_DEPENDENCE"


def _pair_row(
    ticker_a: str,
    ticker_b: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    block_metadata: pd.DataFrame,
    config: IntradayCrossAssetDependenceConfig,
) -> dict[str, object]:
    aligned = left.add_suffix("_a").join(right.add_suffix("_b"), how="inner")
    changes = aligned.loc[
        aligned["intraday_log_close_change_a"].notna()
        & aligned["intraday_log_close_change_b"].notna()
    ]
    observations = len(changes)
    sessions = int(changes["session_date_a"].nunique())
    correlation = _safe_corr(
        changes["intraday_log_close_change_a"],
        changes["intraday_log_close_change_b"],
        "pearson",
    )
    context_nmi = _normalized_mutual_information(
        aligned["joint_context_a"], aligned["joint_context_b"]
    )

    complete_blocks = set(
        block_metadata.loc[block_metadata["block_complete"], "block_id"]
    )
    block_correlations: list[float] = []
    if "block_id_a" in changes:
        for block_id, group in changes.groupby("block_id_a", sort=True):
            if block_id not in complete_blocks or len(group) < int(
                config.minimum_block_observations
            ):
                continue
            value = _safe_corr(
                group["intraday_log_close_change_a"],
                group["intraday_log_close_change_b"],
                "pearson",
            )
            if math.isfinite(value):
                block_correlations.append(value)

    finite_blocks = np.asarray(block_correlations, dtype=float)
    source_status_a = str(left["source_status"].iloc[0])
    source_status_b = str(right["source_status"].iloc[0])
    data_mode_a = str(left["data_mode"].iloc[0])
    data_mode_b = str(right["data_mode"].iloc[0])
    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "source_status_a": source_status_a,
        "source_status_b": source_status_b,
        "data_mode_a": data_mode_a,
        "data_mode_b": data_mode_b,
        "observations": observations,
        "sessions": sessions,
        "eligible_blocks": len(block_correlations),
        "close_change_correlation": correlation,
        "close_change_spearman": _safe_corr(
            changes["intraday_log_close_change_a"],
            changes["intraday_log_close_change_b"],
            "spearman",
        ),
        "absolute_change_correlation": _safe_corr(
            changes["intraday_absolute_log_close_change_a"],
            changes["intraday_absolute_log_close_change_b"],
            "pearson",
        ),
        "median_block_close_change_correlation": (
            float(np.median(finite_blocks)) if finite_blocks.size else math.nan
        ),
        "minimum_block_close_change_correlation": (
            float(np.min(finite_blocks)) if finite_blocks.size else math.nan
        ),
        "maximum_block_close_change_correlation": (
            float(np.max(finite_blocks)) if finite_blocks.size else math.nan
        ),
        "maximum_block_correlation_change": _max_adjacent_change(
            block_correlations
        ),
        "choppiness_spearman": _safe_corr(
            aligned["choppiness_a"], aligned["choppiness_b"], "spearman"
        ),
        "squeeze_momentum_spearman": _safe_corr(
            aligned["squeeze_momentum_pct_close_a"],
            aligned["squeeze_momentum_pct_close_b"],
            "spearman",
        ),
        "cmf_spearman": (
            _safe_corr(aligned["cmf_a"], aligned["cmf_b"], "spearman")
            if "cmf_a" in aligned and "cmf_b" in aligned
            else math.nan
        ),
        "chop_state_agreement": _agreement(
            aligned["chop_segment_a"], aligned["chop_segment_b"]
        ),
        "squeeze_state_agreement": _agreement(
            aligned["squeeze_state_a"], aligned["squeeze_state_b"]
        ),
        "joint_state_agreement": _agreement(
            aligned["joint_context_a"], aligned["joint_context_b"]
        ),
        "joint_context_nmi": context_nmi,
        "dependence_state": _dependence_state(
            observations,
            sessions,
            len(block_correlations),
            correlation,
            context_nmi,
            config,
        ),
    }


def _spectral_metrics(values: pd.DataFrame) -> dict[str, float]:
    if values.shape[1] < 2 or len(values) < 3:
        return {
            "effective_asset_count": math.nan,
            "effective_asset_fraction": math.nan,
            "dominant_component_share": math.nan,
            "median_pair_absolute_correlation": math.nan,
            "p90_pair_absolute_correlation": math.nan,
        }
    correlation = values.corr(method="pearson").to_numpy(dtype=float)
    if not np.isfinite(correlation).all():
        return {
            "effective_asset_count": math.nan,
            "effective_asset_fraction": math.nan,
            "dominant_component_share": math.nan,
            "median_pair_absolute_correlation": math.nan,
            "p90_pair_absolute_correlation": math.nan,
        }
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    total = float(eigenvalues.sum())
    squares = float(np.square(eigenvalues).sum())
    effective = total * total / squares if squares > 0.0 else math.nan
    upper = np.abs(correlation[np.triu_indices(values.shape[1], k=1)])
    return {
        "effective_asset_count": effective,
        "effective_asset_fraction": effective / values.shape[1],
        "dominant_component_share": (
            float(eigenvalues.max() / total) if total > 0.0 else math.nan
        ),
        "median_pair_absolute_correlation": float(np.median(upper)),
        "p90_pair_absolute_correlation": float(np.quantile(upper, 0.90)),
    }


def _breadth_state(
    metrics: Mapping[str, float],
    observations: int,
    config: IntradayCrossAssetDependenceConfig,
) -> str:
    fraction = float(metrics["effective_asset_fraction"])
    dominant = float(metrics["dominant_component_share"])
    if (
        observations < int(config.minimum_pair_observations)
        or not math.isfinite(fraction)
        or not math.isfinite(dominant)
    ):
        return "INSUFFICIENT_OVERLAP"
    if (
        fraction < float(config.concentrated_effective_fraction)
        or dominant > float(config.concentrated_dominant_share)
    ):
        return "CONCENTRATED_EVIDENCE"
    if (
        fraction >= float(config.distributed_effective_fraction)
        and dominant <= float(config.distributed_dominant_share)
    ):
        return "DISTRIBUTED_EVIDENCE"
    return "PARTIAL_EFFECTIVE_BREADTH"


def _scope_tables(
    scope: str,
    tickers: list[str],
    prepared: Mapping[str, pd.DataFrame],
    block_metadata: pd.DataFrame,
    config: IntradayCrossAssetDependenceConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    changes = pd.concat(
        [
            prepared[ticker]["intraday_log_close_change"].rename(ticker)
            for ticker in tickers
        ],
        axis=1,
        join="outer",
    ).sort_index()
    complete = changes.dropna()
    overall_metrics = _spectral_metrics(complete)
    sessions = int(
        pd.Index(complete.index.tz_convert(config.market_timezone).date).nunique()
    ) if not complete.empty else 0
    asset_list = ";".join(tickers)
    block_rows: list[dict[str, object]] = []
    for meta in block_metadata.itertuples(index=False):
        start = pd.Timestamp(meta.block_start_session).date()
        end = pd.Timestamp(meta.block_end_session).date()
        dates = pd.Index(complete.index.tz_convert(config.market_timezone).date)
        selected = complete.loc[(dates >= start) & (dates <= end)]
        metrics = _spectral_metrics(selected)
        state = _breadth_state(metrics, len(selected), config)
        block_rows.append(
            {
                "scope": scope,
                "block_id": int(meta.block_id),
                "block_start_session": meta.block_start_session,
                "block_end_session": meta.block_end_session,
                "block_sessions": int(meta.block_sessions),
                "block_complete": bool(meta.block_complete),
                "assets": len(tickers),
                "asset_list": asset_list,
                "observations": len(selected),
                **metrics,
                "breadth_state": state,
            }
        )
    blocks = pd.DataFrame(block_rows, columns=BLOCK_BREADTH_COLUMNS)
    eligible = blocks.loc[
        blocks["block_complete"]
        & blocks["observations"].ge(int(config.minimum_block_observations))
        & blocks["effective_asset_count"].notna()
    ]
    summary = {
        "scope": scope,
        "assets": len(tickers),
        "asset_list": asset_list,
        "observations": len(complete),
        "sessions": sessions,
        "eligible_blocks": len(eligible),
        **overall_metrics,
        "minimum_block_effective_asset_count": (
            float(eligible["effective_asset_count"].min())
            if not eligible.empty
            else math.nan
        ),
        "median_block_effective_asset_count": (
            float(eligible["effective_asset_count"].median())
            if not eligible.empty
            else math.nan
        ),
        "maximum_block_dominant_component_share": (
            float(eligible["dominant_component_share"].max())
            if not eligible.empty
            else math.nan
        ),
        "minimum_block_observations": (
            int(eligible["observations"].min()) if not eligible.empty else 0
        ),
        "breadth_state": (
            _breadth_state(overall_metrics, len(complete), config)
            if len(eligible) >= int(config.minimum_complete_blocks)
            else "INSUFFICIENT_BLOCKS"
        ),
    }
    return blocks, summary


def _orient_loading(vector: np.ndarray) -> np.ndarray:
    """Rende deterministico il segno senza usare un asset di riferimento."""

    oriented = np.asarray(vector, dtype=float).copy()
    if oriented.size == 0:
        return oriented
    anchor = int(np.argmax(np.abs(oriented)))
    if oriented[anchor] < 0.0:
        oriented *= -1.0
    return oriented


def _residual_spectral_metrics(values: pd.DataFrame) -> dict[str, float]:
    standard_deviation = values.std(axis=0, ddof=0)
    active = standard_deviation.loc[standard_deviation > 1e-10].index
    if len(active) == 0:
        return {
            "effective_asset_count": 0.0,
            "median_pair_absolute_correlation": math.nan,
        }
    if len(active) == 1:
        return {
            "effective_asset_count": 1.0,
            "median_pair_absolute_correlation": math.nan,
        }
    metrics = _spectral_metrics(values.loc[:, active])
    return {
        "effective_asset_count": float(metrics["effective_asset_count"]),
        "median_pair_absolute_correlation": float(
            metrics["median_pair_absolute_correlation"]
        ),
    }


def _factor_decomposition(values: pd.DataFrame) -> dict[str, object] | None:
    if values.shape[1] < 2 or len(values) < 3:
        return None
    numeric = values.apply(pd.to_numeric, errors="coerce").dropna()
    if len(numeric) < 3:
        return None
    standard_deviation = numeric.std(axis=0, ddof=0)
    if standard_deviation.le(1e-12).any():
        return None
    standardized = (numeric - numeric.mean(axis=0)) / standard_deviation
    correlation = standardized.corr(method="pearson").to_numpy(dtype=float)
    if not np.isfinite(correlation).all():
        return None
    correlation = (correlation + correlation.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    vector = _orient_loading(eigenvectors[:, order[0]])
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return None
    factor_score = standardized.to_numpy(dtype=float) @ vector
    reconstructed = np.outer(factor_score, vector)
    residual = pd.DataFrame(
        standardized.to_numpy(dtype=float) - reconstructed,
        index=standardized.index,
        columns=standardized.columns,
    )
    raw_metrics = _spectral_metrics(numeric)
    residual_metrics = _residual_spectral_metrics(residual)
    eigenvalue = float(eigenvalues[0])
    return {
        "observations": len(numeric),
        "factor_share": eigenvalue / total,
        "loading": vector,
        "residual": residual,
        "loading_share": np.square(vector),
        "common_variance_share": np.clip(
            eigenvalue * np.square(vector), 0.0, 1.0
        ),
        "raw_effective_asset_count": float(
            raw_metrics["effective_asset_count"]
        ),
        "residual_effective_asset_count": float(
            residual_metrics["effective_asset_count"]
        ),
        "raw_median_pair_absolute_correlation": float(
            raw_metrics["median_pair_absolute_correlation"]
        ),
        "residual_median_pair_absolute_correlation": float(
            residual_metrics["median_pair_absolute_correlation"]
        ),
    }


def _loading_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return math.nan
    # A principal-component direction is unchanged when every loading changes
    # sign.  The absolute cosine therefore compares the subspace rather than
    # treating an arbitrary eigensolver sign flip as structural drift.
    return float(np.clip(abs(np.dot(left, right)) / denominator, 0.0, 1.0))


def _common_mode_state(
    eligible_blocks: int,
    adjacent_cosines: list[float],
    factor_shares: list[float],
    config: IntradayCrossAssetDependenceConfig,
) -> str:
    finite_cosines = [value for value in adjacent_cosines if math.isfinite(value)]
    finite_shares = [value for value in factor_shares if math.isfinite(value)]
    if (
        eligible_blocks < int(config.minimum_complete_blocks)
        or len(finite_cosines) < max(int(config.minimum_complete_blocks) - 1, 1)
        or len(finite_shares) < int(config.minimum_complete_blocks)
    ):
        return "INSUFFICIENT_BLOCKS"
    minimum_cosine = min(finite_cosines)
    maximum_share_change = max(
        abs(current - previous)
        for previous, current in zip(finite_shares, finite_shares[1:])
    )
    if (
        minimum_cosine >= float(config.stable_minimum_loading_cosine)
        and maximum_share_change
        <= float(config.stable_maximum_factor_share_change)
    ):
        return "STABLE_COMMON_MODE"
    if (
        minimum_cosine >= float(config.variable_minimum_loading_cosine)
        and maximum_share_change
        <= float(config.variable_maximum_factor_share_change)
    ):
        return "VARIABLE_COMMON_MODE"
    return "UNSTABLE_COMMON_MODE"


def _factor_scope_tables(
    scope: str,
    tickers: list[str],
    prepared: Mapping[str, pd.DataFrame],
    block_metadata: pd.DataFrame,
    config: IntradayCrossAssetDependenceConfig,
    *,
    change_column: str = "intraday_log_close_change",
    session_phase: str | None = None,
    minimum_block_observations: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if session_phase is not None and session_phase not in SESSION_PHASES:
        raise ValueError(f"Fase di sessione non valida: {session_phase}.")
    block_minimum = (
        int(minimum_block_observations)
        if minimum_block_observations is not None
        else int(config.minimum_block_observations)
    )
    if block_minimum <= 0:
        raise ValueError("minimum_block_observations deve essere positivo.")
    change_series = []
    for ticker in tickers:
        frame = prepared[ticker]
        if change_column not in frame:
            raise ValueError(
                f"Colonna variazione mancante per {ticker}: {change_column}."
            )
        if session_phase is not None:
            frame = frame.loc[frame["session_phase"].eq(session_phase)]
        change_series.append(frame[change_column].rename(ticker))
    changes = pd.concat(
        change_series,
        axis=1,
        join="outer",
    ).sort_index()
    complete = changes.dropna()
    overall = _factor_decomposition(complete)
    asset_list = ";".join(tickers)
    if overall is None:
        loadings = _empty(FACTOR_LOADING_COLUMNS)
    else:
        loading_rows = []
        absolute_order = np.argsort(np.abs(overall["loading"]))[::-1]
        ranks = {
            int(position): rank + 1
            for rank, position in enumerate(absolute_order)
        }
        for position, ticker in enumerate(tickers):
            loading_rows.append(
                {
                    "scope": scope,
                    "ticker": ticker,
                    "assets": len(tickers),
                    "observations": int(overall["observations"]),
                    "loading": float(overall["loading"][position]),
                    "absolute_loading": float(abs(overall["loading"][position])),
                    "loading_share": float(overall["loading_share"][position]),
                    "common_variance_share": float(
                        overall["common_variance_share"][position]
                    ),
                    "absolute_loading_rank": int(ranks[position]),
                }
            )
        loadings = pd.DataFrame(loading_rows, columns=FACTOR_LOADING_COLUMNS)

    block_rows: list[dict[str, object]] = []
    previous_loading: np.ndarray | None = None
    eligible_factor_shares: list[float] = []
    eligible_cosines: list[float] = []
    eligible_residual_counts: list[float] = []
    dates = pd.Index(complete.index.tz_convert(config.market_timezone).date)
    for meta in block_metadata.itertuples(index=False):
        start = pd.Timestamp(meta.block_start_session).date()
        end = pd.Timestamp(meta.block_end_session).date()
        selected = complete.loc[(dates >= start) & (dates <= end)]
        decomposition = (
            _factor_decomposition(selected)
            if bool(meta.block_complete)
            and len(selected) >= block_minimum
            else None
        )
        if not bool(meta.block_complete):
            state = "INCOMPLETE_BLOCK"
        elif len(selected) < block_minimum:
            state = "INSUFFICIENT_OBSERVATIONS"
        elif decomposition is None:
            state = "UNAVAILABLE_DECOMPOSITION"
        else:
            state = "BLOCK_AVAILABLE"
        cosine = math.nan
        if decomposition is not None:
            loading = np.asarray(decomposition["loading"], dtype=float)
            if previous_loading is not None:
                cosine = _loading_cosine(previous_loading, loading)
                eligible_cosines.append(cosine)
            previous_loading = loading
            eligible_factor_shares.append(float(decomposition["factor_share"]))
            eligible_residual_counts.append(
                float(decomposition["residual_effective_asset_count"])
            )
        block_rows.append(
            {
                "scope": scope,
                "block_id": int(meta.block_id),
                "block_start_session": meta.block_start_session,
                "block_end_session": meta.block_end_session,
                "block_sessions": int(meta.block_sessions),
                "block_complete": bool(meta.block_complete),
                "assets": len(tickers),
                "asset_list": asset_list,
                "observations": len(selected),
                "common_factor_share": (
                    float(decomposition["factor_share"])
                    if decomposition is not None
                    else math.nan
                ),
                "raw_effective_asset_count": (
                    float(decomposition["raw_effective_asset_count"])
                    if decomposition is not None
                    else math.nan
                ),
                "residual_effective_asset_count": (
                    float(decomposition["residual_effective_asset_count"])
                    if decomposition is not None
                    else math.nan
                ),
                "raw_median_pair_absolute_correlation": (
                    float(decomposition["raw_median_pair_absolute_correlation"])
                    if decomposition is not None
                    else math.nan
                ),
                "residual_median_pair_absolute_correlation": (
                    float(
                        decomposition[
                            "residual_median_pair_absolute_correlation"
                        ]
                    )
                    if decomposition is not None
                    else math.nan
                ),
                "loading_cosine_previous_block": cosine,
                "block_state": state,
            }
        )
    blocks = pd.DataFrame(block_rows, columns=FACTOR_BLOCK_COLUMNS)
    eligible_blocks = len(eligible_factor_shares)
    sessions = int(
        pd.Index(complete.index.tz_convert(config.market_timezone).date).nunique()
    ) if not complete.empty else 0
    maximum_share_change = (
        max(
            abs(current - previous)
            for previous, current in zip(
                eligible_factor_shares, eligible_factor_shares[1:]
            )
        )
        if len(eligible_factor_shares) >= 2
        else math.nan
    )
    summary = {
        "scope": scope,
        "assets": len(tickers),
        "asset_list": asset_list,
        "observations": len(complete),
        "sessions": sessions,
        "eligible_blocks": eligible_blocks,
        "common_factor_share": (
            float(overall["factor_share"]) if overall is not None else math.nan
        ),
        "raw_effective_asset_count": (
            float(overall["raw_effective_asset_count"])
            if overall is not None
            else math.nan
        ),
        "residual_effective_asset_count": (
            float(overall["residual_effective_asset_count"])
            if overall is not None
            else math.nan
        ),
        "raw_median_pair_absolute_correlation": (
            float(overall["raw_median_pair_absolute_correlation"])
            if overall is not None
            else math.nan
        ),
        "residual_median_pair_absolute_correlation": (
            float(overall["residual_median_pair_absolute_correlation"])
            if overall is not None
            else math.nan
        ),
        "median_block_common_factor_share": (
            float(np.median(eligible_factor_shares))
            if eligible_factor_shares
            else math.nan
        ),
        "maximum_adjacent_factor_share_change": maximum_share_change,
        "median_adjacent_loading_cosine": (
            float(np.median(eligible_cosines)) if eligible_cosines else math.nan
        ),
        "minimum_adjacent_loading_cosine": (
            float(np.min(eligible_cosines)) if eligible_cosines else math.nan
        ),
        "minimum_block_residual_effective_asset_count": (
            float(np.min(eligible_residual_counts))
            if eligible_residual_counts
            else math.nan
        ),
        "median_block_residual_effective_asset_count": (
            float(np.median(eligible_residual_counts))
            if eligible_residual_counts
            else math.nan
        ),
        "common_mode_state": _common_mode_state(
            eligible_blocks,
            eligible_cosines,
            eligible_factor_shares,
            config,
        ),
    }
    return loadings, blocks, summary


def _phase_extreme(
    values: pd.DataFrame,
    column: str,
    *,
    maximum: bool,
) -> str:
    finite = values.loc[pd.to_numeric(values[column], errors="coerce").notna()]
    if finite.empty:
        return "UNAVAILABLE"
    ordered = finite.copy()
    ordered["_phase_order"] = ordered["session_phase"].map(
        {phase: position for position, phase in enumerate(SESSION_PHASES)}
    )
    ordered = ordered.sort_values("_phase_order", kind="mergesort")
    index = ordered[column].idxmax() if maximum else ordered[column].idxmin()
    return str(ordered.loc[index, "session_phase"])


def _phase_structure_state(
    phases_available: int,
    minimum_eligible_blocks: int,
    factor_share_range: float,
    raw_effective_fraction_range: float,
    residual_effective_fraction_range: float,
    config: IntradayCrossAssetDependenceConfig,
) -> str:
    values = (
        factor_share_range,
        raw_effective_fraction_range,
        residual_effective_fraction_range,
    )
    if (
        phases_available != len(SESSION_PHASES)
        or minimum_eligible_blocks < int(config.minimum_complete_blocks)
        or not all(math.isfinite(float(value)) for value in values)
    ):
        return "INSUFFICIENT_PHASES"
    if (
        factor_share_range >= float(config.high_phase_factor_share_range)
        or raw_effective_fraction_range
        >= float(config.high_phase_effective_fraction_range)
        or residual_effective_fraction_range
        >= float(config.high_phase_effective_fraction_range)
    ):
        return "HIGH_PHASE_HETEROGENEITY"
    if (
        factor_share_range >= float(config.moderate_phase_factor_share_range)
        or raw_effective_fraction_range
        >= float(config.moderate_phase_effective_fraction_range)
        or residual_effective_fraction_range
        >= float(config.moderate_phase_effective_fraction_range)
    ):
        return "MODERATE_PHASE_HETEROGENEITY"
    return "LOW_PHASE_HETEROGENEITY"


def _phase_factor_scope_tables(
    scope: str,
    tickers: list[str],
    prepared: Mapping[str, pd.DataFrame],
    block_metadata: pd.DataFrame,
    config: IntradayCrossAssetDependenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    loading_tables: list[pd.DataFrame] = []
    block_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for phase in SESSION_PHASES:
        loadings, blocks, summary = _factor_scope_tables(
            scope,
            tickers,
            prepared,
            block_metadata,
            config,
            change_column="phase_consistent_log_close_change",
            session_phase=phase,
            minimum_block_observations=(
                config.minimum_phase_block_observations
            ),
        )
        loadings.insert(1, "session_phase", phase)
        blocks.insert(1, "session_phase", phase)
        loading_tables.append(
            loadings.loc[:, list(PHASE_FACTOR_LOADING_COLUMNS)]
        )
        block_tables.append(blocks.loc[:, list(PHASE_FACTOR_BLOCK_COLUMNS)])
        summary_rows.append(
            {
                "scope": scope,
                "session_phase": phase,
                **{
                    key: value
                    for key, value in summary.items()
                    if key != "scope"
                },
            }
        )
    phase_loadings = pd.concat(loading_tables, ignore_index=True)
    phase_blocks = pd.concat(block_tables, ignore_index=True)
    phase_summary = pd.DataFrame(
        summary_rows,
        columns=PHASE_FACTOR_SUMMARY_COLUMNS,
    )
    valid = phase_summary.loc[
        phase_summary["common_factor_share"].notna()
        & phase_summary["raw_effective_asset_count"].notna()
        & phase_summary["residual_effective_asset_count"].notna()
    ].copy()
    phases_available = int(valid["session_phase"].nunique())
    assets = len(tickers)

    def metric_range(column: str) -> tuple[float, float, float]:
        values = pd.to_numeric(valid[column], errors="coerce").dropna()
        if values.empty:
            return math.nan, math.nan, math.nan
        minimum = float(values.min())
        maximum = float(values.max())
        return minimum, maximum, maximum - minimum

    common_minimum, common_maximum, common_range = metric_range(
        "common_factor_share"
    )
    raw_minimum, raw_maximum, raw_range = metric_range(
        "raw_effective_asset_count"
    )
    residual_minimum, residual_maximum, residual_range = metric_range(
        "residual_effective_asset_count"
    )
    raw_correlation_minimum, raw_correlation_maximum, raw_correlation_range = (
        metric_range("raw_median_pair_absolute_correlation")
    )
    minimum_eligible_blocks = (
        int(valid["eligible_blocks"].min()) if not valid.empty else 0
    )
    raw_fraction_range = raw_range / assets if assets else math.nan
    residual_fraction_range = (
        residual_range / assets if assets else math.nan
    )
    contrast = {
        "scope": scope,
        "assets": assets,
        "asset_list": ";".join(tickers),
        "phases_available": phases_available,
        "minimum_phase_observations": (
            int(valid["observations"].min()) if not valid.empty else 0
        ),
        "minimum_eligible_blocks": minimum_eligible_blocks,
        "minimum_common_factor_share": common_minimum,
        "median_common_factor_share": (
            float(valid["common_factor_share"].median())
            if not valid.empty
            else math.nan
        ),
        "maximum_common_factor_share": common_maximum,
        "common_factor_share_range": common_range,
        "maximum_common_factor_session_phase": _phase_extreme(
            valid, "common_factor_share", maximum=True
        ),
        "minimum_common_factor_session_phase": _phase_extreme(
            valid, "common_factor_share", maximum=False
        ),
        "minimum_raw_effective_asset_count": raw_minimum,
        "maximum_raw_effective_asset_count": raw_maximum,
        "raw_effective_asset_count_range": raw_range,
        "raw_effective_fraction_range": raw_fraction_range,
        "maximum_raw_effective_asset_session_phase": _phase_extreme(
            valid, "raw_effective_asset_count", maximum=True
        ),
        "minimum_residual_effective_asset_count": residual_minimum,
        "maximum_residual_effective_asset_count": residual_maximum,
        "residual_effective_asset_count_range": residual_range,
        "residual_effective_fraction_range": residual_fraction_range,
        "maximum_residual_effective_asset_session_phase": _phase_extreme(
            valid, "residual_effective_asset_count", maximum=True
        ),
        "minimum_raw_median_pair_absolute_correlation": (
            raw_correlation_minimum
        ),
        "maximum_raw_median_pair_absolute_correlation": (
            raw_correlation_maximum
        ),
        "raw_median_pair_absolute_correlation_range": raw_correlation_range,
        "phase_structure_state": _phase_structure_state(
            phases_available,
            minimum_eligible_blocks,
            common_range,
            raw_fraction_range,
            residual_fraction_range,
            config,
        ),
    }
    return phase_loadings, phase_blocks, phase_summary, contrast


def _stable_proxy_link(
    raw_row: object,
    config: IntradayCrossAssetDependenceConfig,
) -> bool:
    return _stable_proxy_metrics(
        int(getattr(raw_row, "eligible_blocks")),
        float(getattr(raw_row, "close_change_correlation")),
        float(getattr(raw_row, "minimum_block_close_change_correlation")),
        float(getattr(raw_row, "maximum_block_correlation_change")),
        config,
    )


def _stable_proxy_metrics(
    eligible_blocks: int,
    raw_correlation: float,
    minimum_block_raw_correlation: float,
    maximum_block_raw_correlation_change: float,
    config: IntradayCrossAssetDependenceConfig,
) -> bool:
    if int(eligible_blocks) < int(config.minimum_complete_blocks):
        return False
    values = (
        float(raw_correlation),
        float(minimum_block_raw_correlation),
        float(maximum_block_raw_correlation_change),
    )
    if not all(math.isfinite(v) for v in values):
        return False
    return bool(
        values[0] >= float(config.proxy_minimum_raw_correlation)
        and values[1] >= float(config.proxy_minimum_raw_block_correlation)
        and values[2] <= float(config.proxy_maximum_raw_block_change)
    )


def _residual_pair_state(
    observations: int,
    eligible_blocks: int,
    residual_correlation: float,
    stable_proxy: bool,
    config: IntradayCrossAssetDependenceConfig,
) -> str:
    if (
        observations < int(config.minimum_pair_observations)
        or eligible_blocks < int(config.minimum_complete_blocks)
        or not math.isfinite(float(residual_correlation))
    ):
        return "INSUFFICIENT_RESIDUAL_OVERLAP"
    if stable_proxy:
        return "STABLE_PROXY_LINK"
    magnitude = abs(float(residual_correlation))
    if magnitude >= float(config.high_absolute_correlation):
        return "HIGH_RESIDUAL_DEPENDENCE"
    if magnitude >= float(config.moderate_absolute_correlation):
        return "MODERATE_RESIDUAL_DEPENDENCE"
    return "LOW_RESIDUAL_DEPENDENCE"


def _residual_scope_tables(
    scope: str,
    tickers: list[str],
    prepared: Mapping[str, pd.DataFrame],
    raw_pairs: pd.DataFrame,
    block_metadata: pd.DataFrame,
    config: IntradayCrossAssetDependenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    changes = pd.concat(
        [
            prepared[ticker]["intraday_log_close_change"].rename(ticker)
            for ticker in tickers
        ],
        axis=1,
        join="outer",
    ).sort_index()
    complete = changes.dropna()
    overall = _factor_decomposition(complete)
    overall_residual = (
        overall["residual"]
        if overall is not None
        else pd.DataFrame(index=complete.index, columns=tickers, dtype=float)
    )
    dates = (
        pd.Index(complete.index.tz_convert(config.market_timezone).date)
        if not complete.empty
        else pd.Index([])
    )
    pair_keys = [
        (ticker_a, ticker_b)
        for position, ticker_a in enumerate(tickers)
        for ticker_b in tickers[position + 1 :]
    ]
    block_values: dict[tuple[str, str], list[float]] = {
        key: [] for key in pair_keys
    }
    block_rows: list[dict[str, object]] = []
    for meta in block_metadata.itertuples(index=False):
        start = pd.Timestamp(meta.block_start_session).date()
        end = pd.Timestamp(meta.block_end_session).date()
        selected = complete.loc[(dates >= start) & (dates <= end)]
        decomposition = (
            _factor_decomposition(selected)
            if bool(meta.block_complete)
            and len(selected) >= int(config.minimum_block_observations)
            else None
        )
        if not bool(meta.block_complete):
            state = "INCOMPLETE_BLOCK"
        elif len(selected) < int(config.minimum_block_observations):
            state = "INSUFFICIENT_OBSERVATIONS"
        elif decomposition is None:
            state = "UNAVAILABLE_DECOMPOSITION"
        else:
            state = "BLOCK_AVAILABLE"
        residual = decomposition["residual"] if decomposition is not None else None
        for ticker_a, ticker_b in pair_keys:
            correlation = (
                float(residual[ticker_a].corr(residual[ticker_b]))
                if residual is not None
                else math.nan
            )
            if state == "BLOCK_AVAILABLE" and math.isfinite(correlation):
                block_values[(ticker_a, ticker_b)].append(correlation)
            block_rows.append(
                {
                    "scope": scope,
                    "block_id": int(meta.block_id),
                    "block_start_session": meta.block_start_session,
                    "block_end_session": meta.block_end_session,
                    "block_sessions": int(meta.block_sessions),
                    "block_complete": bool(meta.block_complete),
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "observations": len(selected),
                    "residual_correlation": correlation,
                    "block_state": state,
                }
            )

    raw_lookup = {
        (str(row.ticker_a), str(row.ticker_b)): row
        for row in raw_pairs.itertuples(index=False)
        if str(row.ticker_a) in tickers and str(row.ticker_b) in tickers
    }
    pair_rows: list[dict[str, object]] = []
    for ticker_a, ticker_b in pair_keys:
        raw = raw_lookup[(ticker_a, ticker_b)]
        residual_correlation = (
            float(overall_residual[ticker_a].corr(overall_residual[ticker_b]))
            if not overall_residual.empty
            else math.nan
        )
        values = block_values[(ticker_a, ticker_b)]
        stable_proxy = _stable_proxy_link(raw, config)
        pair_rows.append(
            {
                "scope": scope,
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "observations": len(complete),
                "eligible_blocks": len(values),
                "raw_correlation": float(raw.close_change_correlation),
                "minimum_block_raw_correlation": float(
                    raw.minimum_block_close_change_correlation
                ),
                "maximum_block_raw_correlation_change": float(
                    raw.maximum_block_correlation_change
                ),
                "residual_correlation": residual_correlation,
                "median_block_residual_correlation": (
                    float(np.median(values)) if values else math.nan
                ),
                "minimum_block_residual_correlation": (
                    float(np.min(values)) if values else math.nan
                ),
                "maximum_block_residual_correlation": (
                    float(np.max(values)) if values else math.nan
                ),
                "maximum_block_residual_correlation_change": (
                    _max_adjacent_change(values) if values else math.nan
                ),
                "stable_proxy_link": stable_proxy,
                "residual_pair_state": _residual_pair_state(
                    len(complete),
                    len(values),
                    residual_correlation,
                    stable_proxy,
                    config,
                ),
            }
        )
    blocks = pd.DataFrame(block_rows, columns=RESIDUAL_PAIR_BLOCK_COLUMNS)
    pairs = pd.DataFrame(pair_rows, columns=RESIDUAL_PAIR_COLUMNS)
    return blocks, pairs, complete


def _phase_residual_state(
    phases_available: int,
    minimum_eligible_blocks: int,
    absolute_residual_range: float,
    maximum_absolute_residual: float,
    residual_sign_changes: int,
    stable_proxy_phases: int,
    config: IntradayCrossAssetDependenceConfig,
) -> str:
    if (
        phases_available != len(SESSION_PHASES)
        or minimum_eligible_blocks < int(config.minimum_complete_blocks)
        or not math.isfinite(float(absolute_residual_range))
    ):
        return "INSUFFICIENT_PHASES"
    if stable_proxy_phases == len(SESSION_PHASES):
        return "STABLE_PROXY_ACROSS_PHASES"
    if 0 < stable_proxy_phases < len(SESSION_PHASES):
        return "PHASE_SPECIFIC_PROXY_LINK"
    if (
        residual_sign_changes > 0
        and maximum_absolute_residual
        >= float(config.moderate_absolute_correlation)
    ):
        return "PHASE_SIGN_REVERSAL"
    if absolute_residual_range >= float(
        config.high_phase_residual_absolute_range
    ):
        return "HIGH_PHASE_RESIDUAL_VARIATION"
    if absolute_residual_range >= float(
        config.moderate_phase_residual_absolute_range
    ):
        return "MODERATE_PHASE_RESIDUAL_VARIATION"
    return "LOW_PHASE_RESIDUAL_VARIATION"


def _phase_residual_scope_tables(
    scope: str,
    tickers: list[str],
    prepared: Mapping[str, pd.DataFrame],
    block_metadata: pd.DataFrame,
    config: IntradayCrossAssetDependenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_keys = [
        (ticker_a, ticker_b)
        for position, ticker_a in enumerate(tickers)
        for ticker_b in tickers[position + 1 :]
    ]
    block_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for phase in SESSION_PHASES:
        changes = pd.concat(
            [
                prepared[ticker]
                .loc[prepared[ticker]["session_phase"].eq(phase),
                     "phase_consistent_log_close_change"]
                .rename(ticker)
                for ticker in tickers
            ],
            axis=1,
            join="outer",
        ).sort_index()
        complete = changes.dropna()
        overall = _factor_decomposition(complete)
        overall_residual = (
            overall["residual"]
            if overall is not None
            else pd.DataFrame(index=complete.index, columns=tickers, dtype=float)
        )
        dates = (
            pd.Index(complete.index.tz_convert(config.market_timezone).date)
            if not complete.empty
            else pd.Index([])
        )
        raw_block_values: dict[tuple[str, str], list[float]] = {
            key: [] for key in pair_keys
        }
        residual_block_values: dict[tuple[str, str], list[float]] = {
            key: [] for key in pair_keys
        }
        for meta in block_metadata.itertuples(index=False):
            start = pd.Timestamp(meta.block_start_session).date()
            end = pd.Timestamp(meta.block_end_session).date()
            selected = complete.loc[(dates >= start) & (dates <= end)]
            decomposition = (
                _factor_decomposition(selected)
                if bool(meta.block_complete)
                and len(selected) >= int(config.minimum_phase_block_observations)
                else None
            )
            if not bool(meta.block_complete):
                state = "INCOMPLETE_BLOCK"
            elif len(selected) < int(config.minimum_phase_block_observations):
                state = "INSUFFICIENT_OBSERVATIONS"
            elif decomposition is None:
                state = "UNAVAILABLE_DECOMPOSITION"
            else:
                state = "BLOCK_AVAILABLE"
            residual = (
                decomposition["residual"]
                if decomposition is not None
                else None
            )
            for ticker_a, ticker_b in pair_keys:
                raw_correlation = (
                    float(selected[ticker_a].corr(selected[ticker_b]))
                    if decomposition is not None
                    else math.nan
                )
                residual_correlation = (
                    float(residual[ticker_a].corr(residual[ticker_b]))
                    if residual is not None
                    else math.nan
                )
                if state == "BLOCK_AVAILABLE":
                    if math.isfinite(raw_correlation):
                        raw_block_values[(ticker_a, ticker_b)].append(
                            raw_correlation
                        )
                    if math.isfinite(residual_correlation):
                        residual_block_values[(ticker_a, ticker_b)].append(
                            residual_correlation
                        )
                block_rows.append(
                    {
                        "scope": scope,
                        "session_phase": phase,
                        "block_id": int(meta.block_id),
                        "block_start_session": meta.block_start_session,
                        "block_end_session": meta.block_end_session,
                        "block_sessions": int(meta.block_sessions),
                        "block_complete": bool(meta.block_complete),
                        "ticker_a": ticker_a,
                        "ticker_b": ticker_b,
                        "observations": len(selected),
                        "raw_correlation": raw_correlation,
                        "residual_correlation": residual_correlation,
                        "block_state": state,
                    }
                )
        sessions = (
            int(
                pd.Index(
                    complete.index.tz_convert(config.market_timezone).date
                ).nunique()
            )
            if not complete.empty
            else 0
        )
        for ticker_a, ticker_b in pair_keys:
            raw_values = raw_block_values[(ticker_a, ticker_b)]
            residual_values = residual_block_values[(ticker_a, ticker_b)]
            eligible_blocks = min(len(raw_values), len(residual_values))
            raw_correlation = (
                float(complete[ticker_a].corr(complete[ticker_b]))
                if not complete.empty
                else math.nan
            )
            residual_correlation = (
                float(
                    overall_residual[ticker_a].corr(
                        overall_residual[ticker_b]
                    )
                )
                if not overall_residual.empty
                else math.nan
            )
            minimum_raw = (
                float(np.min(raw_values)) if raw_values else math.nan
            )
            maximum_raw = (
                float(np.max(raw_values)) if raw_values else math.nan
            )
            maximum_raw_change = (
                _max_adjacent_change(raw_values) if raw_values else math.nan
            )
            stable_proxy = _stable_proxy_metrics(
                eligible_blocks,
                raw_correlation,
                minimum_raw,
                maximum_raw_change,
                config,
            )
            pair_rows.append(
                {
                    "scope": scope,
                    "session_phase": phase,
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "observations": len(complete),
                    "sessions": sessions,
                    "eligible_blocks": eligible_blocks,
                    "raw_correlation": raw_correlation,
                    "minimum_block_raw_correlation": minimum_raw,
                    "maximum_block_raw_correlation": maximum_raw,
                    "maximum_block_raw_correlation_change": (
                        maximum_raw_change
                    ),
                    "residual_correlation": residual_correlation,
                    "median_block_residual_correlation": (
                        float(np.median(residual_values))
                        if residual_values
                        else math.nan
                    ),
                    "minimum_block_residual_correlation": (
                        float(np.min(residual_values))
                        if residual_values
                        else math.nan
                    ),
                    "maximum_block_residual_correlation": (
                        float(np.max(residual_values))
                        if residual_values
                        else math.nan
                    ),
                    "maximum_block_residual_correlation_change": (
                        _max_adjacent_change(residual_values)
                        if residual_values
                        else math.nan
                    ),
                    "stable_proxy_link": stable_proxy,
                    "residual_pair_state": _residual_pair_state(
                        len(complete),
                        eligible_blocks,
                        residual_correlation,
                        stable_proxy,
                        config,
                    ),
                }
            )

    blocks = pd.DataFrame(
        block_rows,
        columns=PHASE_RESIDUAL_PAIR_BLOCK_COLUMNS,
    )
    pairs = pd.DataFrame(pair_rows, columns=PHASE_RESIDUAL_PAIR_COLUMNS)
    contrast_rows: list[dict[str, object]] = []
    for (ticker_a, ticker_b), group in pairs.groupby(
        ["ticker_a", "ticker_b"], sort=True
    ):
        valid = group.loc[
            group["raw_correlation"].notna()
            & group["residual_correlation"].notna()
        ].copy()
        raw = pd.to_numeric(valid["raw_correlation"], errors="coerce").dropna()
        residual = pd.to_numeric(
            valid["residual_correlation"], errors="coerce"
        ).dropna()
        absolute_residual = residual.abs()
        raw_minimum = float(raw.min()) if not raw.empty else math.nan
        raw_maximum = float(raw.max()) if not raw.empty else math.nan
        residual_minimum = (
            float(residual.min()) if not residual.empty else math.nan
        )
        residual_maximum = (
            float(residual.max()) if not residual.empty else math.nan
        )
        absolute_residual_minimum = (
            float(absolute_residual.min())
            if not absolute_residual.empty
            else math.nan
        )
        absolute_residual_maximum = (
            float(absolute_residual.max())
            if not absolute_residual.empty
            else math.nan
        )
        absolute_residual_range = (
            absolute_residual_maximum - absolute_residual_minimum
            if math.isfinite(absolute_residual_minimum)
            and math.isfinite(absolute_residual_maximum)
            else math.nan
        )
        phases_available = int(valid["session_phase"].nunique())
        minimum_eligible_blocks = (
            int(valid["eligible_blocks"].min()) if not valid.empty else 0
        )
        residual_sign_changes = int(
            not residual.empty
            and float(residual.min()) < 0.0 < float(residual.max())
        )
        stable_proxy_phases = int(
            valid["stable_proxy_link"].astype(bool).sum()
        )
        raw_extreme = valid.assign(
            absolute_value=valid["raw_correlation"].abs()
        ).rename(columns={"absolute_value": "_absolute_value"})
        residual_extreme = valid.assign(
            absolute_value=valid["residual_correlation"].abs()
        ).rename(columns={"absolute_value": "_absolute_value"})
        contrast_rows.append(
            {
                "scope": scope,
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "phases_available": phases_available,
                "minimum_phase_observations": (
                    int(valid["observations"].min()) if not valid.empty else 0
                ),
                "minimum_eligible_blocks": minimum_eligible_blocks,
                "minimum_raw_correlation": raw_minimum,
                "maximum_raw_correlation": raw_maximum,
                "raw_correlation_range": (
                    raw_maximum - raw_minimum
                    if math.isfinite(raw_minimum)
                    and math.isfinite(raw_maximum)
                    else math.nan
                ),
                "maximum_absolute_raw_correlation": (
                    float(raw.abs().max()) if not raw.empty else math.nan
                ),
                "maximum_absolute_raw_session_phase": _phase_extreme(
                    raw_extreme.rename(
                        columns={"_absolute_value": "absolute_value"}
                    ),
                    "absolute_value",
                    maximum=True,
                ),
                "minimum_residual_correlation": residual_minimum,
                "maximum_residual_correlation": residual_maximum,
                "residual_correlation_range": (
                    residual_maximum - residual_minimum
                    if math.isfinite(residual_minimum)
                    and math.isfinite(residual_maximum)
                    else math.nan
                ),
                "minimum_absolute_residual_correlation": (
                    absolute_residual_minimum
                ),
                "maximum_absolute_residual_correlation": (
                    absolute_residual_maximum
                ),
                "absolute_residual_correlation_range": (
                    absolute_residual_range
                ),
                "maximum_absolute_residual_session_phase": _phase_extreme(
                    residual_extreme.rename(
                        columns={"_absolute_value": "absolute_value"}
                    ),
                    "absolute_value",
                    maximum=True,
                ),
                "residual_sign_changes": residual_sign_changes,
                "stable_proxy_phases": stable_proxy_phases,
                "phase_state_count": int(
                    valid["residual_pair_state"].nunique()
                ),
                "phase_residual_state": _phase_residual_state(
                    phases_available,
                    minimum_eligible_blocks,
                    absolute_residual_range,
                    absolute_residual_maximum,
                    residual_sign_changes,
                    stable_proxy_phases,
                    config,
                ),
            }
        )
    contrast = pd.DataFrame(
        contrast_rows,
        columns=PHASE_RESIDUAL_CONTRAST_COLUMNS,
    )
    return blocks, pairs, contrast


def _proxy_components(
    tickers: list[str], residual_pairs: pd.DataFrame
) -> list[list[str]]:
    parent = {ticker: ticker for ticker in tickers}

    def find(ticker: str) -> str:
        while parent[ticker] != ticker:
            parent[ticker] = parent[parent[ticker]]
            ticker = parent[ticker]
        return ticker

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for row in residual_pairs.loc[
        residual_pairs["stable_proxy_link"].astype(bool)
    ].itertuples(index=False):
        union(str(row.ticker_a), str(row.ticker_b))
    groups: dict[str, list[str]] = {}
    for ticker in tickers:
        groups.setdefault(find(ticker), []).append(ticker)
    return sorted(
        (sorted(group) for group in groups.values()),
        key=lambda group: (-len(group), tuple(group)),
    )


def _proxy_membership_table(
    scope: str,
    tickers: list[str],
    prepared: Mapping[str, pd.DataFrame],
    residual_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, list[list[str]]]:
    components = _proxy_components(tickers, residual_pairs)
    linked = residual_pairs.loc[
        residual_pairs["stable_proxy_link"].astype(bool)
    ]
    rows: list[dict[str, object]] = []
    for number, component in enumerate(components, start=1):
        cluster_id = f"C{number:02d}"
        state = (
            "STABLE_PROXY_GROUP"
            if len(component) > 1
            else "SINGLETON_AT_DECLARED_THRESHOLDS"
        )
        members = ";".join(component)
        for ticker in component:
            frame = prepared[ticker]
            stable_link_count = int(
                (
                    linked["ticker_a"].eq(ticker)
                    | linked["ticker_b"].eq(ticker)
                ).sum()
            )
            rows.append(
                {
                    "scope": scope,
                    "cluster_id": cluster_id,
                    "cluster_members": members,
                    "member_count": len(component),
                    "ticker": ticker,
                    "source_status": str(frame["source_status"].iloc[0]),
                    "data_mode": str(frame["data_mode"].iloc[0]),
                    "stable_link_count": stable_link_count,
                    "cluster_state": state,
                }
            )
    return pd.DataFrame(rows, columns=PROXY_CLUSTER_COLUMNS), components


def _sensitivity_metrics(
    complete: pd.DataFrame,
    retained: list[str],
) -> dict[str, float | int]:
    if len(retained) < 2 or len(complete) < 3:
        return {
            "observations": len(complete),
            "effective_asset_count": math.nan,
            "effective_asset_fraction": math.nan,
            "common_factor_share": math.nan,
            "median_pair_absolute_correlation": math.nan,
            "residual_effective_asset_count": math.nan,
            "residual_median_pair_absolute_correlation": math.nan,
        }
    values = complete.loc[:, retained]
    spectral = _spectral_metrics(values)
    factor = _factor_decomposition(values)
    return {
        "observations": len(values),
        "effective_asset_count": float(spectral["effective_asset_count"]),
        "effective_asset_fraction": float(spectral["effective_asset_fraction"]),
        "common_factor_share": (
            float(factor["factor_share"]) if factor is not None else math.nan
        ),
        "median_pair_absolute_correlation": float(
            spectral["median_pair_absolute_correlation"]
        ),
        "residual_effective_asset_count": (
            float(factor["residual_effective_asset_count"])
            if factor is not None
            else math.nan
        ),
        "residual_median_pair_absolute_correlation": (
            float(factor["residual_median_pair_absolute_correlation"])
            if factor is not None
            else math.nan
        ),
    }


def _proxy_scenarios(
    tickers: list[str],
    components: list[list[str]],
    config: IntradayCrossAssetDependenceConfig,
) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = [
        {
            "scenario_type": "BASELINE",
            "scenario_id": "BASELINE",
            "retained": sorted(tickers),
            "scenario_state": "BASELINE_SCOPE",
        }
    ]
    proxy_groups = [group for group in components if len(group) > 1]
    singletons = [group[0] for group in components if len(group) == 1]
    combinations = math.prod(len(group) for group in proxy_groups)
    if proxy_groups and combinations <= int(config.maximum_proxy_combinations):
        for number, choices in enumerate(product(*proxy_groups), start=1):
            scenarios.append(
                {
                    "scenario_type": "COLLAPSE_STABLE_PROXY_GROUPS",
                    "scenario_id": f"COLLAPSE_{number:03d}",
                    "retained": sorted(singletons + list(choices)),
                    "scenario_state": (
                        "ENUMERATED_WITHOUT_REPRESENTATIVE_SELECTION"
                    ),
                }
            )
    elif proxy_groups:
        scenarios.append(
            {
                "scenario_type": "COLLAPSE_STABLE_PROXY_GROUPS",
                "scenario_id": "COLLAPSE_LIMIT",
                "retained": sorted(tickers),
                "scenario_state": "COMBINATION_LIMIT_EXCEEDED",
            }
        )
    for number, group in enumerate(proxy_groups, start=1):
        scenarios.append(
            {
                "scenario_type": "LEAVE_STABLE_PROXY_GROUP_OUT",
                "scenario_id": f"LEAVE_CLUSTER_{number:02d}",
                "retained": sorted(
                    ticker for ticker in tickers if ticker not in group
                ),
                "scenario_state": "DESCRIPTIVE_CLUSTER_SENSITIVITY",
            }
        )
    return scenarios


def _proxy_sensitivity_table(
    scope: str,
    tickers: list[str],
    complete: pd.DataFrame,
    components: list[list[str]],
    config: IntradayCrossAssetDependenceConfig,
) -> pd.DataFrame:
    baseline = _sensitivity_metrics(complete, tickers)
    rows: list[dict[str, object]] = []
    for scenario in _proxy_scenarios(tickers, components, config):
        retained = list(scenario["retained"])
        omitted = sorted(set(tickers).difference(retained))
        metrics = _sensitivity_metrics(complete, retained)
        effective_fraction = float(metrics["effective_asset_fraction"])
        residual_count = float(metrics["residual_effective_asset_count"])
        rows.append(
            {
                "scope": scope,
                "scenario_type": scenario["scenario_type"],
                "scenario_id": scenario["scenario_id"],
                "retained_assets": ";".join(retained),
                "omitted_assets": ";".join(omitted),
                "assets": len(retained),
                **metrics,
                "effective_fraction_change_vs_baseline": (
                    effective_fraction - float(baseline["effective_asset_fraction"])
                    if math.isfinite(effective_fraction)
                    else math.nan
                ),
                "residual_effective_count_change_vs_baseline": (
                    residual_count
                    - float(baseline["residual_effective_asset_count"])
                    if math.isfinite(residual_count)
                    else math.nan
                ),
                "scenario_state": scenario["scenario_state"],
            }
        )
    return pd.DataFrame(rows, columns=PROXY_SENSITIVITY_COLUMNS)


def _unavailable_sensitivity_metrics(observations: int) -> dict[str, float | int]:
    return {
        "observations": int(observations),
        "effective_asset_count": math.nan,
        "effective_asset_fraction": math.nan,
        "common_factor_share": math.nan,
        "median_pair_absolute_correlation": math.nan,
        "residual_effective_asset_count": math.nan,
        "residual_median_pair_absolute_correlation": math.nan,
    }


def _finite_difference(value: object, baseline: object) -> float:
    current = float(value)
    reference = float(baseline)
    if not math.isfinite(current) or not math.isfinite(reference):
        return math.nan
    return current - reference


def _proxy_block_sensitivity_table(
    scope: str,
    tickers: list[str],
    complete: pd.DataFrame,
    components: list[list[str]],
    block_metadata: pd.DataFrame,
    config: IntradayCrossAssetDependenceConfig,
) -> pd.DataFrame:
    scenarios = _proxy_scenarios(tickers, components, config)
    dates = (
        pd.Index(complete.index.tz_convert(config.market_timezone).date)
        if not complete.empty
        else pd.Index([])
    )
    rows: list[dict[str, object]] = []
    for meta in block_metadata.itertuples(index=False):
        start = pd.Timestamp(meta.block_start_session).date()
        end = pd.Timestamp(meta.block_end_session).date()
        selected = complete.loc[(dates >= start) & (dates <= end)]
        if not bool(meta.block_complete):
            base_state = "INCOMPLETE_BLOCK"
        elif len(selected) < int(config.minimum_block_observations):
            base_state = "INSUFFICIENT_OBSERVATIONS"
        else:
            base_state = "BLOCK_AVAILABLE"
        baseline = (
            _sensitivity_metrics(selected, tickers)
            if base_state == "BLOCK_AVAILABLE"
            else _unavailable_sensitivity_metrics(len(selected))
        )
        for scenario in scenarios:
            retained = list(scenario["retained"])
            omitted = sorted(set(tickers).difference(retained))
            block_state = base_state
            if base_state != "BLOCK_AVAILABLE":
                block_state = base_state
            elif scenario["scenario_state"] == "COMBINATION_LIMIT_EXCEEDED":
                block_state = "COMBINATION_LIMIT_EXCEEDED"
            elif len(retained) < 2:
                block_state = "INSUFFICIENT_ASSETS"
            metrics = (
                _sensitivity_metrics(selected, retained)
                if block_state == "BLOCK_AVAILABLE"
                else _unavailable_sensitivity_metrics(len(selected))
            )
            rows.append(
                {
                    "scope": scope,
                    "block_id": int(meta.block_id),
                    "block_start_session": meta.block_start_session,
                    "block_end_session": meta.block_end_session,
                    "block_sessions": int(meta.block_sessions),
                    "block_complete": bool(meta.block_complete),
                    "scenario_type": scenario["scenario_type"],
                    "scenario_id": scenario["scenario_id"],
                    "retained_assets": ";".join(retained),
                    "omitted_assets": ";".join(omitted),
                    "assets": len(retained),
                    **metrics,
                    "effective_fraction_change_vs_block_baseline": (
                        _finite_difference(
                            metrics["effective_asset_fraction"],
                            baseline["effective_asset_fraction"],
                        )
                    ),
                    "residual_effective_count_change_vs_block_baseline": (
                        _finite_difference(
                            metrics["residual_effective_asset_count"],
                            baseline["residual_effective_asset_count"],
                        )
                    ),
                    "scenario_state": scenario["scenario_state"],
                    "block_state": block_state,
                }
            )
    return pd.DataFrame(rows, columns=PROXY_BLOCK_SENSITIVITY_COLUMNS)


def _finite_summary(values: pd.Series) -> tuple[float, float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.loc[np.isfinite(numeric.to_numpy(dtype=float))]
    if numeric.empty:
        return math.nan, math.nan, math.nan, math.nan
    minimum = float(numeric.min())
    median = float(numeric.median())
    maximum = float(numeric.max())
    return minimum, median, maximum, maximum - minimum


def _proxy_block_stability_table(
    blocks: pd.DataFrame,
    config: IntradayCrossAssetDependenceConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["scope", "scenario_type", "scenario_id"]
    for _, group in blocks.groupby(group_columns, sort=True):
        group = group.sort_values("block_id", kind="mergesort")
        eligible = group.loc[group["block_state"].eq("BLOCK_AVAILABLE")]
        first = group.iloc[0]
        effective = _finite_summary(eligible["effective_asset_count"])
        common = _finite_summary(eligible["common_factor_share"])
        residual = _finite_summary(
            eligible["residual_effective_asset_count"]
        )
        raw_correlation = _finite_summary(
            eligible["median_pair_absolute_correlation"]
        )
        residual_correlation = _finite_summary(
            eligible["residual_median_pair_absolute_correlation"]
        )
        rows.append(
            {
                "scope": first["scope"],
                "scenario_type": first["scenario_type"],
                "scenario_id": first["scenario_id"],
                "retained_assets": first["retained_assets"],
                "omitted_assets": first["omitted_assets"],
                "assets": int(first["assets"]),
                "eligible_blocks": len(eligible),
                "minimum_block_observations": (
                    int(eligible["observations"].min())
                    if not eligible.empty
                    else 0
                ),
                "minimum_effective_asset_count": effective[0],
                "median_effective_asset_count": effective[1],
                "maximum_effective_asset_count": effective[2],
                "effective_asset_count_range": effective[3],
                "minimum_common_factor_share": common[0],
                "median_common_factor_share": common[1],
                "maximum_common_factor_share": common[2],
                "common_factor_share_range": common[3],
                "minimum_residual_effective_asset_count": residual[0],
                "median_residual_effective_asset_count": residual[1],
                "maximum_residual_effective_asset_count": residual[2],
                "residual_effective_asset_count_range": residual[3],
                "minimum_median_pair_absolute_correlation": raw_correlation[0],
                "median_median_pair_absolute_correlation": raw_correlation[1],
                "maximum_median_pair_absolute_correlation": raw_correlation[2],
                "median_pair_absolute_correlation_range": raw_correlation[3],
                "minimum_residual_median_pair_absolute_correlation": (
                    residual_correlation[0]
                ),
                "median_residual_median_pair_absolute_correlation": (
                    residual_correlation[1]
                ),
                "maximum_residual_median_pair_absolute_correlation": (
                    residual_correlation[2]
                ),
                "residual_median_pair_absolute_correlation_range": (
                    residual_correlation[3]
                ),
                "scenario_state": first["scenario_state"],
                "block_stability_state": (
                    "SUFFICIENT_COMPLETE_BLOCKS"
                    if len(eligible) >= int(config.minimum_complete_blocks)
                    else "INSUFFICIENT_COMPLETE_BLOCKS"
                ),
            }
        )
    return pd.DataFrame(rows, columns=PROXY_BLOCK_STABILITY_COLUMNS)


def _proxy_representative_invariance_table(
    scope: str,
    blocks: pd.DataFrame,
    components: list[list[str]],
    config: IntradayCrossAssetDependenceConfig,
) -> pd.DataFrame:
    proxy_groups = [group for group in components if len(group) > 1]
    expected = math.prod(len(group) for group in proxy_groups)
    rows: list[dict[str, object]] = []
    for block_id, group in blocks.groupby("block_id", sort=True):
        baseline = group.loc[group["scenario_type"].eq("BASELINE")].iloc[0]
        collapse = group.loc[
            group["scenario_type"].eq("COLLAPSE_STABLE_PROXY_GROUPS")
            & group["scenario_state"].eq(
                "ENUMERATED_WITHOUT_REPRESENTATIVE_SELECTION"
            )
            & group["block_state"].eq("BLOCK_AVAILABLE")
        ]
        if not proxy_groups:
            state = "NO_STABLE_PROXY_GROUPS"
        elif expected > int(config.maximum_proxy_combinations):
            state = "COMBINATION_LIMIT_EXCEEDED"
        elif baseline["block_state"] != "BLOCK_AVAILABLE":
            state = str(baseline["block_state"])
        elif collapse["scenario_id"].nunique() != expected:
            state = "INCOMPLETE_REPRESENTATIVE_GRID"
        else:
            state = "COMPLETE_REPRESENTATIVE_GRID"
        complete_grid = state == "COMPLETE_REPRESENTATIVE_GRID"

        def spread(column: str) -> float:
            return _finite_summary(collapse[column])[3] if complete_grid else math.nan

        rows.append(
            {
                "scope": scope,
                "block_id": int(block_id),
                "block_start_session": baseline["block_start_session"],
                "block_end_session": baseline["block_end_session"],
                "block_sessions": int(baseline["block_sessions"]),
                "block_complete": bool(baseline["block_complete"]),
                "expected_scenarios": expected if proxy_groups else 0,
                "observed_scenarios": collapse["scenario_id"].nunique(),
                "effective_asset_count_range": spread(
                    "effective_asset_count"
                ),
                "effective_asset_fraction_range": spread(
                    "effective_asset_fraction"
                ),
                "common_factor_share_range": spread("common_factor_share"),
                "median_pair_absolute_correlation_range": spread(
                    "median_pair_absolute_correlation"
                ),
                "residual_effective_asset_count_range": spread(
                    "residual_effective_asset_count"
                ),
                "residual_median_pair_absolute_correlation_range": spread(
                    "residual_median_pair_absolute_correlation"
                ),
                "invariance_state": state,
            }
        )
    return pd.DataFrame(
        rows, columns=PROXY_REPRESENTATIVE_INVARIANCE_COLUMNS
    )


def analyze_intraday_cross_asset_dependence(
    markets: Mapping[str, pd.DataFrame],
    feature_frames: Mapping[str, pd.DataFrame],
    source_audits: Mapping[str, object],
    config: IntradayCrossAssetDependenceConfig | None = None,
) -> IntradayCrossAssetDependenceReport:
    """Misura dipendenza contemporanea e ampiezza effettiva del campione."""

    active = config or IntradayCrossAssetDependenceConfig()
    common = sorted(
        set(markets).intersection(feature_frames).intersection(source_audits)
    )
    if len(common) < 2:
        raise ValueError("Servono almeno due asset completi per il confronto.")
    prepared = {
        ticker: _prepare_asset(
            ticker,
            markets[ticker],
            feature_frames[ticker],
            source_audits[ticker],
            active,
        )
        for ticker in common
    }
    session_map, block_metadata = _session_blocks(prepared, active)
    for frame in prepared.values():
        frame["block_id"] = frame["session_date"].map(session_map).astype(int)

    pair_rows = []
    for position, ticker_a in enumerate(common):
        for ticker_b in common[position + 1 :]:
            pair_rows.append(
                _pair_row(
                    ticker_a,
                    ticker_b,
                    prepared[ticker_a],
                    prepared[ticker_b],
                    block_metadata,
                    active,
                )
            )
    pairs = pd.DataFrame(pair_rows, columns=PAIR_COLUMNS).sort_values(
        ["ticker_a", "ticker_b"], kind="mergesort"
    ).reset_index(drop=True)

    special = {
        str(ticker).upper().strip()
        for ticker in active.special_context_tickers
        if str(ticker).strip()
    }
    ordinary = [ticker for ticker in common if ticker not in special]
    scopes: list[tuple[str, list[str]]] = []
    if len(ordinary) >= 2 and ordinary != common:
        scopes.append(("ORDINARY_ASSETS", ordinary))
    scopes.append(("ALL_ASSETS_DESCRIPTIVE", common))

    block_tables = []
    summaries = []
    factor_loading_tables = []
    factor_block_tables = []
    factor_summaries = []
    residual_pair_block_tables = []
    residual_pair_tables = []
    proxy_cluster_tables = []
    proxy_sensitivity_tables = []
    proxy_block_sensitivity_tables = []
    proxy_block_stability_tables = []
    proxy_representative_invariance_tables = []
    phase_factor_loading_tables = []
    phase_factor_block_tables = []
    phase_factor_summary_tables = []
    phase_factor_contrasts = []
    phase_residual_pair_block_tables = []
    phase_residual_pair_tables = []
    phase_residual_contrast_tables = []
    for scope, tickers in scopes:
        blocks, summary = _scope_tables(
            scope, tickers, prepared, block_metadata, active
        )
        block_tables.append(blocks)
        summaries.append(summary)
        factor_loadings, factor_blocks, factor_summary = _factor_scope_tables(
            scope, tickers, prepared, block_metadata, active
        )
        factor_loading_tables.append(factor_loadings)
        factor_block_tables.append(factor_blocks)
        factor_summaries.append(factor_summary)
        (
            phase_factor_loadings,
            phase_factor_blocks,
            phase_factor_summary,
            phase_factor_contrast,
        ) = _phase_factor_scope_tables(
            scope,
            tickers,
            prepared,
            block_metadata,
            active,
        )
        phase_factor_loading_tables.append(phase_factor_loadings)
        phase_factor_block_tables.append(phase_factor_blocks)
        phase_factor_summary_tables.append(phase_factor_summary)
        phase_factor_contrasts.append(phase_factor_contrast)
        (
            phase_residual_pair_blocks,
            phase_residual_pairs,
            phase_residual_contrast,
        ) = _phase_residual_scope_tables(
            scope,
            tickers,
            prepared,
            block_metadata,
            active,
        )
        phase_residual_pair_block_tables.append(
            phase_residual_pair_blocks
        )
        phase_residual_pair_tables.append(phase_residual_pairs)
        phase_residual_contrast_tables.append(phase_residual_contrast)
        residual_blocks, residual_pairs, complete_changes = (
            _residual_scope_tables(
                scope,
                tickers,
                prepared,
                pairs,
                block_metadata,
                active,
            )
        )
        proxy_clusters, proxy_components = _proxy_membership_table(
            scope,
            tickers,
            prepared,
            residual_pairs,
        )
        proxy_sensitivity = _proxy_sensitivity_table(
            scope,
            tickers,
            complete_changes,
            proxy_components,
            active,
        )
        proxy_block_sensitivity = _proxy_block_sensitivity_table(
            scope,
            tickers,
            complete_changes,
            proxy_components,
            block_metadata,
            active,
        )
        proxy_block_stability = _proxy_block_stability_table(
            proxy_block_sensitivity,
            active,
        )
        proxy_representative_invariance = (
            _proxy_representative_invariance_table(
                scope,
                proxy_block_sensitivity,
                proxy_components,
                active,
            )
        )
        residual_pair_block_tables.append(residual_blocks)
        residual_pair_tables.append(residual_pairs)
        proxy_cluster_tables.append(proxy_clusters)
        proxy_sensitivity_tables.append(proxy_sensitivity)
        proxy_block_sensitivity_tables.append(proxy_block_sensitivity)
        proxy_block_stability_tables.append(proxy_block_stability)
        proxy_representative_invariance_tables.append(
            proxy_representative_invariance
        )
    block_breadth = pd.concat(block_tables, ignore_index=True)
    breadth = pd.DataFrame(summaries, columns=BREADTH_COLUMNS)
    factor_loadings = pd.concat(factor_loading_tables, ignore_index=True)
    factor_blocks = pd.concat(factor_block_tables, ignore_index=True)
    factor_summary = pd.DataFrame(
        factor_summaries, columns=FACTOR_SUMMARY_COLUMNS
    )
    residual_pair_blocks = pd.concat(
        residual_pair_block_tables, ignore_index=True
    )
    residual_pairs = pd.concat(residual_pair_tables, ignore_index=True)
    proxy_clusters = pd.concat(proxy_cluster_tables, ignore_index=True)
    proxy_sensitivity = pd.concat(
        proxy_sensitivity_tables, ignore_index=True
    )
    proxy_block_sensitivity = pd.concat(
        proxy_block_sensitivity_tables, ignore_index=True
    )
    proxy_block_stability = pd.concat(
        proxy_block_stability_tables, ignore_index=True
    )
    proxy_representative_invariance = pd.concat(
        proxy_representative_invariance_tables, ignore_index=True
    )
    phase_factor_loadings = pd.concat(
        phase_factor_loading_tables, ignore_index=True
    )
    phase_factor_blocks = pd.concat(
        phase_factor_block_tables, ignore_index=True
    )
    phase_factor_summary = pd.concat(
        phase_factor_summary_tables, ignore_index=True
    )
    phase_factor_contrast = pd.DataFrame(
        phase_factor_contrasts,
        columns=PHASE_FACTOR_CONTRAST_COLUMNS,
    )
    phase_residual_pair_blocks = pd.concat(
        phase_residual_pair_block_tables,
        ignore_index=True,
    )
    phase_residual_pairs = pd.concat(
        phase_residual_pair_tables,
        ignore_index=True,
    )
    phase_residual_contrast = pd.concat(
        phase_residual_contrast_tables,
        ignore_index=True,
    )
    caveats_list = [
        "Le variazioni di close sono contemporanee, intraseduta e gia' concluse.",
        "La conta effettiva deriva dallo spettro della matrice di correlazione.",
        "Correlazione e contesto condiviso non misurano capacita' predittiva.",
        (
            "La sensibilita' ai proxy e' ripetuta in blocchi fissi senza "
            "scegliere rappresentanti."
        ),
        (
            "Le fasi OPEN, MID_SESSION e CLOSE usano soltanto variazioni "
            "il cui inizio e fine appartengono alla stessa fase."
        ),
        (
            "Le dipendenze residue per fase ricalcolano la componente comune "
            "separatamente in ogni fase e in ogni blocco completo."
        ),
    ]
    if ordinary != common:
        caveats_list.insert(
            2,
            (
                "VXX resta escluso dallo scope ordinario e compare solo "
                "nello scope completo."
            ),
        )
    return IntradayCrossAssetDependenceReport(
        pairs=pairs,
        block_breadth=block_breadth,
        breadth=breadth,
        factor_loadings=factor_loadings,
        factor_blocks=factor_blocks,
        factor_summary=factor_summary,
        residual_pair_blocks=residual_pair_blocks,
        residual_pairs=residual_pairs,
        proxy_clusters=proxy_clusters,
        proxy_sensitivity=proxy_sensitivity,
        proxy_block_sensitivity=proxy_block_sensitivity,
        proxy_block_stability=proxy_block_stability,
        proxy_representative_invariance=proxy_representative_invariance,
        phase_factor_loadings=phase_factor_loadings,
        phase_factor_blocks=phase_factor_blocks,
        phase_factor_summary=phase_factor_summary,
        phase_factor_contrast=phase_factor_contrast,
        phase_residual_pair_blocks=phase_residual_pair_blocks,
        phase_residual_pairs=phase_residual_pairs,
        phase_residual_contrast=phase_residual_contrast,
        caveats=tuple(caveats_list),
    )


def _write(frame: pd.DataFrame, path: str | Path, columns: tuple[str, ...]) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(columns)].to_csv(destination, index=False)
    return destination


def write_cross_asset_dependence_pairs(
    pairs: pd.DataFrame, path: str | Path
) -> Path:
    return _write(pairs, path, PAIR_COLUMNS)


def write_cross_asset_block_breadth(
    blocks: pd.DataFrame, path: str | Path
) -> Path:
    return _write(blocks, path, BLOCK_BREADTH_COLUMNS)


def write_cross_asset_effective_breadth(
    breadth: pd.DataFrame, path: str | Path
) -> Path:
    return _write(breadth, path, BREADTH_COLUMNS)


def write_cross_asset_factor_loadings(
    loadings: pd.DataFrame, path: str | Path
) -> Path:
    return _write(loadings, path, FACTOR_LOADING_COLUMNS)


def write_cross_asset_factor_blocks(
    blocks: pd.DataFrame, path: str | Path
) -> Path:
    return _write(blocks, path, FACTOR_BLOCK_COLUMNS)


def write_cross_asset_factor_summary(
    summary: pd.DataFrame, path: str | Path
) -> Path:
    return _write(summary, path, FACTOR_SUMMARY_COLUMNS)


def write_cross_asset_residual_pair_blocks(
    blocks: pd.DataFrame, path: str | Path
) -> Path:
    return _write(blocks, path, RESIDUAL_PAIR_BLOCK_COLUMNS)


def write_cross_asset_residual_pairs(
    pairs: pd.DataFrame, path: str | Path
) -> Path:
    return _write(pairs, path, RESIDUAL_PAIR_COLUMNS)


def write_cross_asset_proxy_clusters(
    clusters: pd.DataFrame, path: str | Path
) -> Path:
    return _write(clusters, path, PROXY_CLUSTER_COLUMNS)


def write_cross_asset_proxy_sensitivity(
    sensitivity: pd.DataFrame, path: str | Path
) -> Path:
    return _write(sensitivity, path, PROXY_SENSITIVITY_COLUMNS)


def write_cross_asset_proxy_block_sensitivity(
    sensitivity: pd.DataFrame, path: str | Path
) -> Path:
    return _write(sensitivity, path, PROXY_BLOCK_SENSITIVITY_COLUMNS)


def write_cross_asset_proxy_block_stability(
    stability: pd.DataFrame, path: str | Path
) -> Path:
    return _write(stability, path, PROXY_BLOCK_STABILITY_COLUMNS)


def write_cross_asset_proxy_representative_invariance(
    invariance: pd.DataFrame, path: str | Path
) -> Path:
    return _write(
        invariance,
        path,
        PROXY_REPRESENTATIVE_INVARIANCE_COLUMNS,
    )


def write_cross_asset_phase_factor_loadings(
    loadings: pd.DataFrame, path: str | Path
) -> Path:
    return _write(loadings, path, PHASE_FACTOR_LOADING_COLUMNS)


def write_cross_asset_phase_factor_blocks(
    blocks: pd.DataFrame, path: str | Path
) -> Path:
    return _write(blocks, path, PHASE_FACTOR_BLOCK_COLUMNS)


def write_cross_asset_phase_factor_summary(
    summary: pd.DataFrame, path: str | Path
) -> Path:
    return _write(summary, path, PHASE_FACTOR_SUMMARY_COLUMNS)


def write_cross_asset_phase_factor_contrast(
    contrast: pd.DataFrame, path: str | Path
) -> Path:
    return _write(contrast, path, PHASE_FACTOR_CONTRAST_COLUMNS)


def write_cross_asset_phase_residual_pair_blocks(
    blocks: pd.DataFrame, path: str | Path
) -> Path:
    return _write(blocks, path, PHASE_RESIDUAL_PAIR_BLOCK_COLUMNS)


def write_cross_asset_phase_residual_pairs(
    pairs: pd.DataFrame, path: str | Path
) -> Path:
    return _write(pairs, path, PHASE_RESIDUAL_PAIR_COLUMNS)


def write_cross_asset_phase_residual_contrast(
    contrast: pd.DataFrame, path: str | Path
) -> Path:
    return _write(contrast, path, PHASE_RESIDUAL_CONTRAST_COLUMNS)
