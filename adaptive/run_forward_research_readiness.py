"""CLI dell'audit forward-only tra due checkpoint congelati."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adaptive.forward_research_readiness import (
    FORWARD_READINESS_DETAILS_FILENAME,
    FORWARD_READINESS_SUMMARY_FILENAME,
    ForwardReadinessConfig,
    ForwardReadinessState,
    compare_research_freeze_files,
    write_forward_readiness_details,
    write_forward_readiness_summary,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Verifica che un checkpoint contenga soltanto nuove sedute "
            "intatte e una specifica immutata. Non valuta performance."
        )
    )
    parser.add_argument("--baseline-freeze", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--details-output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--minimum-common-forward-sessions", type=int, default=60)
    parser.add_argument("--sessions-per-checkpoint", type=int, default=20)
    parser.add_argument("--minimum-checkpoints", type=int, default=3)
    parser.add_argument("--minimum-eligible-sources", type=int, default=5)
    arguments = parser.parse_args(argv)
    root = Path(arguments.candidate_freeze).resolve().parent / "forward_readiness"
    if arguments.details_output is None:
        arguments.details_output = str(root / FORWARD_READINESS_DETAILS_FILENAME)
    if arguments.summary_output is None:
        arguments.summary_output = str(root / FORWARD_READINESS_SUMMARY_FILENAME)
    return arguments


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    config = ForwardReadinessConfig(
        minimum_common_forward_sessions=(
            arguments.minimum_common_forward_sessions
        ),
        sessions_per_checkpoint=arguments.sessions_per_checkpoint,
        minimum_checkpoints=arguments.minimum_checkpoints,
        minimum_eligible_sources=arguments.minimum_eligible_sources,
    )
    audit = compare_research_freeze_files(
        arguments.baseline_freeze,
        arguments.candidate_freeze,
        config,
    )
    details = write_forward_readiness_details(audit, arguments.details_output)
    summary = write_forward_readiness_summary(audit, arguments.summary_output)

    print("\nTRADINGAI FORWARD EVIDENCE READINESS — RESEARCH-ONLY")
    print("Nessuna previsione, operazione, size, stop, leva o P&L.")
    print(f"Specifica invariata:       {audit.specification_unchanged}")
    print(f"Implementazione invariata: {audit.implementation_unchanged}")
    print(f"Sorgenti idonee:           {audit.eligible_sources}")
    print(f"Sedute forward comuni:     {audit.common_forward_sessions}")
    print(f"Checkpoint completi:       {audit.complete_checkpoints}")
    print(f"Esito:                     {audit.state.value}")
    for reason in audit.reasons:
        print(f"- {reason}")
    print(f"Dettaglio CSV:              {details}")
    print(f"Riepilogo JSON:             {summary}")
    print(
        "PASS significa soltanto che i dati sono pronti per una revisione "
        "scientifica cieca; non dimostra che la strategia funzioni."
    )
    return 1 if audit.state is ForwardReadinessState.FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
