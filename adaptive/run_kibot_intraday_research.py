"""CLI integrata per ricerca descrittiva 15m su tick Kibot."""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

from adaptive.kibot_intraday_research import (
    KibotIntradayResearchConfig,
    KibotIntradayResearchEngine,
)
from adaptive.lorentzian_research import summarize_lorentzian_distribution


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Aggrega tick Kibot a 15m e misura feature, stabilita', "
            "microstruttura e geometria Lorentziana senza segnali o P&L."
        )
    )
    parser.add_argument("--file", required=True)
    parser.add_argument("--ticker", default="IVE")
    parser.add_argument("--interval", default="15m", choices=("15m",))
    parser.add_argument("--sessions-per-block", type=int, default=5)
    parser.add_argument("--minimum-complete-blocks", type=int, default=3)
    parser.add_argument("--context-normalization-window", type=int, default=20)
    parser.add_argument("--context-normalization-min-periods", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--minimum-candidates", type=int, default=16)
    parser.add_argument("--embargo-bars", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--history-limit", type=int, default=2_000)
    return parser.parse_args(argv)


def _percentage(series: pd.Series, value: str) -> float:
    valid = series.loc[series != "INSUFFICIENT"]
    return float((valid == value).mean()) if not valid.empty else math.nan


def _persistent_counts(table: pd.DataFrame) -> tuple[int, int]:
    if table.empty:
        return 0, 0
    persistent = table["pattern"].isin(
        {"PERSISTENT_ELEVATED", "PERSISTENT_HIGH"}
    )
    return int(persistent.sum()), int(
        table["pattern"].eq("PERSISTENT_HIGH").sum()
    )


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    config = KibotIntradayResearchConfig(
        sessions_per_block=arguments.sessions_per_block,
        minimum_complete_blocks=arguments.minimum_complete_blocks,
        context_normalization_window=arguments.context_normalization_window,
        context_normalization_min_periods=(
            arguments.context_normalization_min_periods
        ),
        neighbors=arguments.neighbors,
        minimum_candidates=arguments.minimum_candidates,
        embargo_bars=arguments.embargo_bars,
        sample_stride=arguments.sample_stride,
        history_limit=arguments.history_limit,
    )
    report = KibotIntradayResearchEngine(config).analyze(
        arguments.file,
        arguments.ticker,
    )
    audit = report.tick_result.audit
    features = report.features
    stability = report.stability
    lorentzian = summarize_lorentzian_distribution(report.lorentzian)
    numeric_persistent, numeric_high = _persistent_counts(
        stability.numeric_persistence
    )
    state_persistent, state_high = _persistent_counts(
        stability.state_persistence
    )
    phase_persistent, phase_high = _persistent_counts(
        stability.phase_persistence
    )

    print("\nTRADINGAI KIBOT 15M INTEGRATED RESEARCH — DESCRIPTIVE ONLY")
    print("Tick -> barre 15m -> feature -> stabilita' -> Lorentzian causale.")
    print(
        "Nessuna direzione, previsione, operazione, size, stop, leva, "
        "outcome futuro o P&L."
    )
    print("\nQUALITA' E COPERTURA")
    print(f"Ticker:                         {audit.ticker}")
    print(
        f"Periodo:                        {audit.first_timestamp} -> "
        f"{audit.last_timestamp}"
    )
    print(
        f"Sessioni / barre 15m:           {audit.observed_sessions} / "
        f"{audit.output_bars}"
    )
    print(f"Copertura bucket:               {audit.bucket_coverage:.1%}")
    print(f"Stato sorgente:                 {audit.status.value}")

    print("\nDISTRIBUZIONI FEATURE 15M")
    print(f"Righe:                          {len(features)}")
    print(
        "Squeeze valide:                 "
        f"{int(features['squeeze_momentum'].notna().sum())}"
    )
    print(
        "CHOP valide:                    "
        f"{int(features['choppiness'].notna().sum())}"
    )
    print(
        "CMF valide:                     "
        f"{int(features['cmf'].notna().sum())}"
    )
    print(
        "Frazione CHOPPY:                "
        f"{_percentage(features['chop_segment'], 'CHOPPY'):.1%}"
    )
    print(
        "Frazione TRENDING:              "
        f"{_percentage(features['chop_segment'], 'TRENDING'):.1%}"
    )
    print(
        "Frazione SQUEEZE_ON:            "
        f"{_percentage(features['squeeze_state'], 'SQUEEZE_ON'):.1%}"
    )

    print("\nMICROSTRUTTURA PER FASE")
    print(
        f"{'FASE':14} {'BARRE':>6} {'TICK_MED':>9} {'SPREAD_MED':>11} "
        f"{'SPREAD_P90':>11} {'OUT_NBBO':>10}"
    )
    print("-" * 70)
    for row in report.phase_microstructure.itertuples(index=False):
        print(
            f"{row.session_phase:14} {row.bars:6d} "
            f"{row.median_tick_count:9.1f} {row.median_spread_bps:11.3f} "
            f"{row.median_p90_spread_bps:11.3f} "
            f"{row.mean_outside_nbbo_fraction:10.3%}"
        )

    print("\nSTABILITA' TRA BLOCCHI")
    print(
        "Sessioni / blocchi completi:    "
        f"{stability.observed_sessions} / {stability.complete_blocks}"
    )
    print(
        "Feature numeriche persistenti:  "
        f"{numeric_persistent} (high: {numeric_high})"
    )
    print(
        "Stati persistenti:              "
        f"{state_persistent} (high: {state_high})"
    )
    print(
        "Fasi persistenti:               "
        f"{phase_persistent} (high: {phase_high})"
    )

    print("\nGEOMETRIA LORENTZIANA CAUSALE")
    print(
        "Righe complete:                 "
        f"{lorentzian.complete_rows}/{lorentzian.total_rows}"
    )
    print(f"Copertura:                      {lorentzian.coverage:.1%}")
    print(
        "Distanza mediana / p90:         "
        f"{lorentzian.median_distance:.4f} / {lorentzian.distance_p90:.4f}"
    )
    print(f"Densita' mediana:               {lorentzian.median_density:.4f}")
    print(
        "Eta' mediana vicini (barre):    "
        f"{lorentzian.median_neighbor_age_bars:.1f}"
    )

    print("\nLIMITI")
    for caveat in report.caveats:
        print(f"- {caveat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
