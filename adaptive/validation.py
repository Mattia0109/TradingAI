"""Promotion gate temporale per confronti Champion/Challenger."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PromotionGateResult:
    eligible_for_review: bool
    fold_results: pd.DataFrame
    winning_folds: int
    required_winning_folds: int
    reasons: tuple[str, ...]


def _metrics(returns: pd.Series, annualization: int) -> tuple[float, float, float]:
    clean = returns.astype(float).dropna()
    growth = (1.0 + clean).cumprod()
    total_return = float(growth.iloc[-1] - 1.0)
    deviation = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    sharpe = (
        float(clean.mean()) / deviation * math.sqrt(annualization)
        if deviation > 0.0
        else 0.0
    )
    drawdown = float((1.0 - growth / growth.cummax()).max())
    return total_return, sharpe, drawdown


def evaluate_challenger(
    champion_returns: pd.Series,
    challenger_returns: pd.Series,
    *,
    folds: int = 3,
    annualization_factor: int = 252,
    drawdown_relative_tolerance: float = 0.10,
    drawdown_absolute_tolerance: float = 0.005,
) -> PromotionGateResult:
    aligned = pd.concat(
        [
            champion_returns.rename("champion"),
            challenger_returns.rename("challenger"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if folds < 2:
        raise ValueError("Servono almeno due fold temporali.")
    if len(aligned) < folds * 20:
        raise ValueError("Storico insufficiente per il promotion gate.")
    rows: list[dict[str, object]] = []
    for fold_number, positions in enumerate(
        np.array_split(np.arange(len(aligned)), folds), start=1
    ):
        segment = aligned.iloc[positions]
        champion = _metrics(segment["champion"], annualization_factor)
        challenger = _metrics(segment["challenger"], annualization_factor)
        win = challenger[0] > champion[0] and challenger[1] > champion[1]
        rows.append(
            {
                "fold": fold_number,
                "start": segment.index[0],
                "end": segment.index[-1],
                "champion_return": champion[0],
                "challenger_return": challenger[0],
                "champion_sharpe": champion[1],
                "challenger_sharpe": challenger[1],
                "champion_drawdown": champion[2],
                "challenger_drawdown": challenger[2],
                "challenger_wins": win,
            }
        )
    frame = pd.DataFrame(rows).set_index("fold")
    winning_folds = int(frame["challenger_wins"].sum())
    required = math.ceil(folds * 2.0 / 3.0)
    champion_all = _metrics(aligned["champion"], annualization_factor)
    challenger_all = _metrics(aligned["challenger"], annualization_factor)
    drawdown_limit = (
        champion_all[2] * (1.0 + drawdown_relative_tolerance)
        + drawdown_absolute_tolerance
    )
    reasons: list[str] = []
    if winning_folds < required:
        reasons.append(
            f"Il Challenger vince {winning_folds}/{folds} fold; ne servono {required}."
        )
    if challenger_all[1] <= champion_all[1]:
        reasons.append("Sharpe aggregato del Challenger non superiore al Champion.")
    if challenger_all[2] > drawdown_limit:
        reasons.append("Drawdown del Challenger oltre la tolleranza prudenziale.")
    eligible = not reasons
    if eligible:
        reasons.append(
            "Gate superato: candidato a revisione, senza promozione automatica."
        )
    return PromotionGateResult(
        eligible_for_review=eligible,
        fold_results=frame,
        winning_folds=winning_folds,
        required_winning_folds=required,
        reasons=tuple(reasons),
    )
