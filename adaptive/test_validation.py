import numpy as np
import pandas as pd
import pytest

from adaptive.validation import evaluate_challenger


def returns(value: float, rows: int = 120) -> pd.Series:
    variation = np.where(np.arange(rows) % 2 == 0, 0.0001, -0.0001)
    return pd.Series(
        value + variation,
        index=pd.date_range("2020-01-01", periods=rows, freq="B"),
    )


def test_gate_marks_stable_challenger_only_for_review() -> None:
    result = evaluate_challenger(returns(0.0001), returns(0.0003))
    assert result.eligible_for_review is True
    assert result.winning_folds == 3
    assert "senza promozione automatica" in result.reasons[0]


def test_gate_rejects_unstable_challenger() -> None:
    champion = returns(0.0002)
    challenger = returns(0.0004)
    challenger.iloc[40:80] = -0.002
    result = evaluate_challenger(champion, challenger)
    assert result.eligible_for_review is False
    assert any("Drawdown" in reason for reason in result.reasons)


def test_gate_requires_enough_history() -> None:
    with pytest.raises(ValueError, match="Storico insufficiente"):
        evaluate_challenger(returns(0.0, 30), returns(0.0, 30))
