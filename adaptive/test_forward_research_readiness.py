from __future__ import annotations

import copy
import json

from adaptive.forward_research_readiness import (
    ForwardReadinessConfig,
    ForwardReadinessState,
    compare_research_freezes,
    write_forward_readiness_details,
    write_forward_readiness_summary,
)
from adaptive.research_freeze import IntradayResearchSpecification
from adaptive.run_forward_research_readiness import parse_arguments
from adaptive.test_research_freeze import build_freeze


def compact_config(**overrides) -> ForwardReadinessConfig:
    values = {
        "minimum_common_forward_sessions": 2,
        "sessions_per_checkpoint": 1,
        "minimum_checkpoints": 2,
        "minimum_eligible_sources": 1,
    }
    values.update(overrides)
    return ForwardReadinessConfig(**values)


def test_cumulative_intact_extension_passes_readiness(tmp_path) -> None:
    baseline = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-04", "2023-01-05"],
        reports_name="baseline_reports",
    )
    candidate = build_freeze(
        tmp_path,
        [
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
            "2023-01-09",
        ],
        reports_name="candidate_reports",
    )

    audit = compare_research_freezes(baseline, candidate, compact_config())

    assert audit.state is ForwardReadinessState.PASS
    assert audit.common_forward_sessions == 2
    assert audit.complete_checkpoints == 2
    assert audit.details.loc[0, "comparison_mode"] == "CUMULATIVE"
    assert audit.details.loc[0, "changed_overlap_sessions"] == 0


def test_same_checkpoint_is_inconclusive_not_failure(tmp_path) -> None:
    baseline = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-04"],
        reports_name="reports",
    )

    audit = compare_research_freezes(baseline, baseline, compact_config())

    assert audit.state is ForwardReadinessState.INCONCLUSIVE
    assert audit.common_forward_sessions == 0
    assert any("0/2" in reason for reason in audit.reasons)


def test_changed_historical_session_fails(tmp_path) -> None:
    baseline = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-04"],
        reports_name="baseline_reports",
    )
    candidate = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
        adjustment=0.5,
        reports_name="candidate_reports",
    )

    audit = compare_research_freezes(baseline, candidate, compact_config())

    assert audit.state is ForwardReadinessState.FAIL
    assert audit.details.loc[0, "changed_overlap_sessions"] == 1
    assert "HISTORICAL_DATA_CHANGED" in audit.details.loc[0, "source_state"]


def test_forward_only_non_overlapping_checkpoint_is_supported(tmp_path) -> None:
    baseline = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-04"],
        reports_name="baseline_reports",
    )
    candidate = build_freeze(
        tmp_path,
        ["2023-01-05", "2023-01-06"],
        reports_name="candidate_reports",
    )

    audit = compare_research_freezes(baseline, candidate, compact_config())

    assert audit.state is ForwardReadinessState.PASS
    assert audit.details.loc[0, "comparison_mode"] == "FORWARD_ONLY"
    assert audit.details.loc[0, "forward_sessions"] == 2


def test_specification_change_fails_even_with_new_data(tmp_path) -> None:
    baseline = build_freeze(
        tmp_path,
        ["2023-01-03"],
        reports_name="baseline_reports",
    )
    candidate = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-04", "2023-01-05"],
        specification=IntradayResearchSpecification(neighbors=12),
        reports_name="candidate_reports",
    )

    audit = compare_research_freezes(baseline, candidate, compact_config())

    assert audit.state is ForwardReadinessState.FAIL
    assert audit.specification_unchanged is False


def test_missing_baseline_session_in_cumulative_candidate_fails(tmp_path) -> None:
    baseline = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-04", "2023-01-05"],
        reports_name="baseline_reports",
    )
    candidate = build_freeze(
        tmp_path,
        ["2023-01-03", "2023-01-05", "2023-01-06", "2023-01-09"],
        reports_name="candidate_reports",
    )

    audit = compare_research_freezes(baseline, candidate, compact_config())

    assert audit.state is ForwardReadinessState.FAIL
    assert audit.details.loc[0, "missing_baseline_sessions"] == 1


def test_recomputed_freeze_with_changed_implementation_fails(tmp_path) -> None:
    baseline = build_freeze(
        tmp_path,
        ["2023-01-03"],
        reports_name="reports",
    )
    candidate = copy.deepcopy(baseline)
    from adaptive.research_freeze import canonical_json_bytes
    import hashlib

    first_file = next(iter(candidate["implementation"]["files"]))
    candidate["implementation"]["files"][first_file] = "0" * 64
    candidate["implementation"]["sha256"] = hashlib.sha256(
        canonical_json_bytes(candidate["implementation"]["files"])
    ).hexdigest()
    candidate.pop("freeze_id")
    candidate["freeze_id"] = hashlib.sha256(
        canonical_json_bytes(candidate)
    ).hexdigest()

    audit = compare_research_freezes(baseline, candidate, compact_config())

    assert audit.state is ForwardReadinessState.FAIL
    assert audit.implementation_unchanged is False


def test_summary_and_details_are_research_only(tmp_path) -> None:
    baseline = build_freeze(
        tmp_path,
        ["2023-01-03"],
        reports_name="reports",
    )
    audit = compare_research_freezes(baseline, baseline, compact_config())

    details = write_forward_readiness_details(audit, tmp_path / "details.csv")
    summary = write_forward_readiness_summary(audit, tmp_path / "summary.json")
    payload = json.loads(summary.read_text(encoding="utf-8"))

    assert details.exists()
    assert payload["research_only"] is True
    assert payload["state"] == "INCONCLUSIVE"
    assert "non prova" in payload["interpretation"]


def test_forward_cli_resolves_outputs_next_to_candidate(tmp_path) -> None:
    candidate = tmp_path / "checkpoint" / "freeze.json"
    arguments = parse_arguments(
        [
            "--baseline-freeze",
            str(tmp_path / "baseline.json"),
            "--candidate-freeze",
            str(candidate),
        ]
    )

    assert arguments.details_output.endswith(
        "checkpoint/intraday_forward_readiness_details.csv"
    )
    assert arguments.summary_output.endswith(
        "checkpoint/intraday_forward_readiness_summary.json"
    )
