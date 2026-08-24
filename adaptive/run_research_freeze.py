"""CLI per congelare un checkpoint della ricerca intraday descrittiva."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adaptive.local_intraday_data import load_local_intraday_markets
from adaptive.research_freeze import (
    RESEARCH_FREEZE_FILENAME,
    IntradayResearchSpecification,
    build_research_freeze,
    write_research_freeze,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Congela dati, parametri, codice e report descrittivi. "
            "Non calcola segnali, outcome futuri o P&L."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--interval", default="15m", choices=("15m",))
    parser.add_argument("--sessions-per-block", type=int, default=30)
    parser.add_argument("--minimum-complete-blocks", type=int, default=4)
    parser.add_argument("--normalization-window", type=int, default=520)
    parser.add_argument("--normalization-min-periods", type=int, default=260)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--minimum-candidates", type=int, default=16)
    parser.add_argument("--embargo-bars", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--history-limit", type=int, default=4_000)
    parser.add_argument("--context-minimum-observations", type=int, default=60)
    parser.add_argument("--context-minimum-sessions", type=int, default=10)
    parser.add_argument(
        "--context-minimum-complete-blocks", type=int, default=3
    )
    parser.add_argument("--repository-root", default=None)
    arguments = parser.parse_args(argv)
    if arguments.output is None:
        arguments.output = str(Path(arguments.report_dir) / RESEARCH_FREEZE_FILENAME)
    return arguments


def _specification(arguments) -> IntradayResearchSpecification:
    return IntradayResearchSpecification(
        interval=arguments.interval,
        sessions_per_block=arguments.sessions_per_block,
        minimum_complete_blocks=arguments.minimum_complete_blocks,
        normalization_window=arguments.normalization_window,
        normalization_min_periods=arguments.normalization_min_periods,
        neighbors=arguments.neighbors,
        minimum_candidates=arguments.minimum_candidates,
        embargo_bars=arguments.embargo_bars,
        sample_stride=arguments.sample_stride,
        history_limit=arguments.history_limit,
        context_minimum_observations=arguments.context_minimum_observations,
        context_minimum_sessions=arguments.context_minimum_sessions,
        context_minimum_complete_blocks=(
            arguments.context_minimum_complete_blocks
        ),
    )


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    markets, audits, errors = load_local_intraday_markets(
        arguments.data_dir,
        arguments.tickers,
    )
    freeze = build_research_freeze(
        markets=markets,
        source_audits=audits,
        source_errors=errors,
        report_directory=arguments.report_dir,
        specification=_specification(arguments),
        repository_root=arguments.repository_root,
    )
    destination = write_research_freeze(freeze, arguments.output)
    sources = freeze["sources"]
    first = min(record["first_timestamp"] for record in sources)
    last = max(record["last_timestamp"] for record in sources)

    print("\nTRADINGAI INTRADAY RESEARCH FREEZE — RESEARCH-ONLY")
    print("Nessun segnale, ordine, size, stop, leva, outcome futuro o P&L.")
    print(f"Freeze ID:              {freeze['freeze_id']}")
    print(f"Specifica SHA-256:      {freeze['specification_sha256']}")
    print(f"Implementazione SHA:    {freeze['implementation']['sha256']}")
    print(f"Sorgenti congelate:     {len(sources)}")
    print(f"Intervallo osservato:   {first} -> {last}")
    print(f"Report audit:           {freeze['report_audit']['state']}")
    print(f"File freeze:            {destination}")
    print(
        "Il file e' immutabile: per un checkpoint futuro usa una nuova "
        "directory di report."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
