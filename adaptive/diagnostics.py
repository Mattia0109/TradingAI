"""Diagnostica predittiva post-trade per specialisti e meta-modello."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = (
    "signals",
    "hit_rate",
    "mean_signed_return",
    "mean_net_return",
    "mean_absolute_error",
    "forecast_correlation",
    "average_confidence",
    "average_horizon_bars",
)


def _correlation(group: pd.DataFrame) -> float:
    if len(group) < 2:
        return 0.0
    expected = group["expected_return"].astype(float)
    realized = group["realized_return"].astype(float)
    if expected.nunique() < 2 or realized.nunique() < 2:
        return 0.0
    value = float(expected.corr(realized))
    return value if np.isfinite(value) else 0.0


def _summarize(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "signals": int(len(group)),
            "hit_rate": float((group["signed_return"] > 0.0).mean()),
            "mean_signed_return": float(group["signed_return"].mean()),
            "mean_net_return": float(group["net_signed_return"].mean()),
            "mean_absolute_error": float(
                (group["expected_return"] - group["realized_return"]).abs().mean()
            ),
            "forecast_correlation": _correlation(group),
            "average_confidence": float(group["confidence"].mean()),
            "average_horizon_bars": float(group["horizon_bars"].mean()),
        }
    )


def summarize_forecasts(
    observations: Iterable[Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restituisce osservazioni mature e sintesi per modello e modello/regime."""

    frame = pd.DataFrame(tuple(observations))
    if frame.empty:
        empty = pd.DataFrame(columns=SUMMARY_COLUMNS)
        return frame, empty.copy(), empty.copy()
    active = frame.loc[
        (frame["direction"] != "FLAT")
        & frame["realized_return"].notna()
    ].copy()
    if active.empty:
        empty = pd.DataFrame(columns=SUMMARY_COLUMNS)
        return active, empty.copy(), empty.copy()
    active["signed_return"] = (
        active["direction_sign"].astype(float)
        * active["realized_return"].astype(float)
    )
    active["net_signed_return"] = (
        active["signed_return"]
        - active["estimated_cost_bps"].astype(float) / 10_000.0
    )
    by_strategy = active.groupby("strategy_id", sort=True).apply(
        _summarize, include_groups=False
    )
    by_strategy_regime = active.groupby(
        ["strategy_id", "regime"], sort=True
    ).apply(_summarize, include_groups=False)
    return active, by_strategy, by_strategy_regime
