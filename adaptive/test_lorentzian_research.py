from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from adaptive.lorentzian_research import (
    CausalLorentzianResearchEngine,
    LORENTZIAN_DESCRIPTOR_COLUMNS,
    LorentzianParity,
    LorentzianResearchConfig,
    lorentzian_distance,
    session_phase_context,
    session_slot_context,
    summarize_lorentzian_distribution,
)


def research_features(sessions: int = 36) -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    for day in pd.bdate_range("2025-01-06", periods=sessions):
        start = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(
            hours=9,
            minutes=30,
        )
        timestamps.extend(pd.date_range(start, periods=26, freq="15min"))
    index = pd.DatetimeIndex(timestamps, name="date")
    x = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "squeeze_momentum_pct_close": (
                0.0020 * np.sin(x / 9.0) + 0.0004 * np.cos(x / 31.0)
            ),
            "squeeze_momentum_change_pct_close": (
                0.0003 * np.cos(x / 7.0) - 0.0001 * np.sin(x / 23.0)
            ),
            "choppiness": 50.0 + 12.0 * np.sin(x / 17.0),
            "cmf": 0.18 * np.cos(x / 13.0) + 0.03 * np.sin(x / 5.0),
        },
        index=index,
    )


def compact_config(**overrides) -> LorentzianResearchConfig:
    values = {
        "normalization_window": 52,
        "normalization_min_periods": 26,
        "neighbors": 4,
        "minimum_candidates": 8,
        "embargo_bars": 3,
        "sample_stride": 2,
        "history_limit": 400,
    }
    values.update(overrides)
    return LorentzianResearchConfig(**values)


class LorentzianResearchTests(unittest.TestCase):
    def test_distance_matches_reference_and_has_no_square_root(self) -> None:
        left = [10.0, 20.0, 30.0, 40.0]
        right = [9.0, 23.0, 30.0, 36.0]
        expected = sum(
            math.log1p(abs(a - b)) for a, b in zip(left, right)
        )

        observed = lorentzian_distance(left, right)

        self.assertAlmostEqual(observed, expected)
        self.assertNotAlmostEqual(observed, math.sqrt(expected))
        self.assertEqual(lorentzian_distance(left, left), 0.0)

    def test_future_mutation_cannot_change_past_descriptors(self) -> None:
        values = research_features()
        contexts = session_phase_context(values.index)
        engine = CausalLorentzianResearchEngine(compact_config())
        baseline = engine.compute(values, contexts)
        cutoff = 620
        changed = values.copy()
        changed.iloc[cutoff + 1 :, 0] += 0.25
        changed.iloc[cutoff + 1 :, 1] *= -80.0
        changed.iloc[cutoff + 1 :, 2] += 400.0
        changed.iloc[cutoff + 1 :, 3] -= 6.0

        mutated = engine.compute(changed, contexts)

        pd.testing.assert_frame_equal(
            baseline.normalized_features.iloc[: cutoff + 1],
            mutated.normalized_features.iloc[: cutoff + 1],
        )
        pd.testing.assert_frame_equal(
            baseline.descriptors.iloc[: cutoff + 1],
            mutated.descriptors.iloc[: cutoff + 1],
        )

    def test_slot_normalization_is_causal_and_context_isolated(self) -> None:
        values = research_features(80)
        phases = session_phase_context(values.index)
        slots = session_slot_context(values.index)
        config = compact_config(
            context_normalization_window=30,
            context_normalization_min_periods=15,
        )
        engine = CausalLorentzianResearchEngine(config)
        baseline = engine.compute(values, phases, slots)
        changed = values.copy()
        changed.loc[slots.eq("SLOT_00"), "choppiness"] = (
            changed.loc[slots.eq("SLOT_00"), "choppiness"] * 7.0 + 500.0
        )

        observed = engine.compute(changed, phases, slots)

        unaffected = slots.ne("SLOT_00")
        pd.testing.assert_series_equal(
            baseline.normalized_features.loc[unaffected, "choppiness"],
            observed.normalized_features.loc[unaffected, "choppiness"],
        )
        comparable = slots.eq("SLOT_00") & baseline.normalized_features[
            "choppiness"
        ].notna()
        np.testing.assert_allclose(
            baseline.normalized_features.loc[comparable, "choppiness"],
            observed.normalized_features.loc[comparable, "choppiness"],
            rtol=1e-9,
            atol=1e-9,
        )

    def test_positive_affine_rescaling_preserves_descriptors(self) -> None:
        values = research_features()
        contexts = session_phase_context(values.index)
        engine = CausalLorentzianResearchEngine(compact_config())
        baseline = engine.compute(values, contexts).descriptors
        changed = values.copy()
        changed.iloc[:, 0] = changed.iloc[:, 0] * 300.0 + 7.0
        changed.iloc[:, 1] = changed.iloc[:, 1] * 0.25 - 2.0
        changed.iloc[:, 2] = changed.iloc[:, 2] * 3.0 + 100.0
        changed.iloc[:, 3] = changed.iloc[:, 3] * 11.0 - 4.0

        scaled = engine.compute(changed, contexts).descriptors

        np.testing.assert_allclose(
            baseline.to_numpy(dtype=float),
            scaled.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )

    def test_embargo_neighbor_count_and_context_are_enforced(self) -> None:
        values = research_features()
        config = compact_config(embargo_bars=6, neighbors=5, minimum_candidates=10)
        report = CausalLorentzianResearchEngine(config).compute(
            values,
            session_phase_context(values.index),
        )
        complete = report.descriptors.loc[
            report.descriptors["lorentzian_neighbor_count"] > 0
        ]

        self.assertFalse(complete.empty)
        self.assertTrue(
            (complete["lorentzian_neighbor_count"] == config.neighbors).all()
        )
        self.assertTrue(
            (
                complete["lorentzian_neighbor_age_min_bars"]
                >= config.embargo_bars + 1
            ).all()
        )

    def test_constant_features_are_finite_after_warmup(self) -> None:
        values = research_features(20)
        values.loc[:, :] = [0.0, 0.0, 50.0, 0.0]
        report = CausalLorentzianResearchEngine(compact_config()).compute(values)
        complete = report.descriptors.loc[
            report.descriptors["lorentzian_neighbor_count"] > 0
        ]

        self.assertFalse(complete.empty)
        self.assertTrue(np.isfinite(complete.to_numpy(dtype=float)).all())
        np.testing.assert_allclose(
            complete["lorentzian_distance_median"].to_numpy(),
            0.0,
        )
        np.testing.assert_allclose(
            complete["lorentzian_local_density"].to_numpy(),
            1.0,
        )

    def test_insufficient_history_is_explicit(self) -> None:
        values = research_features(3)
        config = compact_config(
            normalization_window=52,
            normalization_min_periods=40,
            minimum_candidates=20,
        )
        report = CausalLorentzianResearchEngine(config).compute(values)

        self.assertTrue(
            (report.descriptors["lorentzian_neighbor_count"] == 0).all()
        )
        self.assertTrue(
            report.descriptors["lorentzian_distance_median"].isna().all()
        )

    def test_output_contract_is_descriptive_only(self) -> None:
        values = research_features()
        report = CausalLorentzianResearchEngine(compact_config()).compute(
            values,
            session_phase_context(values.index),
        )
        forbidden = {
            "signal",
            "direction",
            "prediction",
            "order",
            "position",
            "entry",
            "exit",
            "pnl",
            "outcome",
            "leverage",
        }

        self.assertEqual(
            tuple(report.descriptors.columns),
            LORENTZIAN_DESCRIPTOR_COLUMNS,
        )
        self.assertTrue(
            all(
                not any(token in column.lower() for token in forbidden)
                for column in report.descriptors.columns
            )
        )
        self.assertTrue(report.research_only)
        self.assertEqual(
            report.parity["distance_formula"],
            LorentzianParity.EXACT_REFERENCE_DISTANCE,
        )
        self.assertEqual(
            report.parity["directional_classifier"],
            LorentzianParity.CLASSIFIER_NOT_REPRODUCED,
        )

    def test_distribution_summary_uses_complete_history(self) -> None:
        values = research_features()
        report = CausalLorentzianResearchEngine(compact_config()).compute(
            values,
            session_phase_context(values.index),
        )

        summary = summarize_lorentzian_distribution(report)

        self.assertEqual(summary.total_rows, len(values))
        self.assertGreater(summary.complete_rows, 0)
        self.assertGreater(summary.coverage, 0.0)
        self.assertLessEqual(summary.coverage, 1.0)
        self.assertTrue(math.isfinite(summary.median_distance))
        self.assertTrue(math.isfinite(summary.distance_p90))
        self.assertTrue(math.isfinite(summary.median_density))

    def test_session_phase_context_uses_new_york_boundaries(self) -> None:
        index = pd.DatetimeIndex(
            [
                "2025-01-06 09:30",
                "2025-01-06 10:15",
                "2025-01-06 10:30",
                "2025-01-06 14:45",
                "2025-01-06 15:00",
                "2025-01-06 15:45",
            ],
            tz="America/New_York",
        )

        context = session_phase_context(index)

        self.assertEqual(
            context.tolist(),
            ["OPEN", "OPEN", "MID_SESSION", "MID_SESSION", "CLOSE", "CLOSE"],
        )

    def test_session_slot_context_uses_exact_rth_grid(self) -> None:
        index = pd.DatetimeIndex(
            ["2025-01-06 09:30", "2025-01-06 10:00", "2025-01-06 15:45"],
            tz="America/New_York",
        )

        slots = session_slot_context(index)

        self.assertEqual(slots.tolist(), ["SLOT_00", "SLOT_02", "SLOT_25"])
        with self.assertRaises(ValueError):
            session_slot_context(
                pd.DatetimeIndex(["2025-01-06 09:31"], tz="America/New_York")
            )


if __name__ == "__main__":
    unittest.main()
