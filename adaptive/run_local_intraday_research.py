"""Report unico sui campioni locali 1m, aggregati causalmente a 15m."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pandas as pd

from adaptive.intraday_features import IntradayReferenceFeatureEngine
from adaptive.intraday_context_stability import (
    IntradayContextStabilityConfig,
    analyze_intraday_context_stability,
    summarize_context_stability,
    write_context_stability,
)
from adaptive.intraday_cross_asset_dependence import (
    IntradayCrossAssetDependenceConfig,
    analyze_intraday_cross_asset_dependence,
    write_cross_asset_block_breadth,
    write_cross_asset_dependence_pairs,
    write_cross_asset_effective_breadth,
    write_cross_asset_factor_blocks,
    write_cross_asset_factor_loadings,
    write_cross_asset_factor_summary,
    write_cross_asset_phase_factor_blocks,
    write_cross_asset_phase_factor_contrast,
    write_cross_asset_phase_factor_loadings,
    write_cross_asset_phase_factor_summary,
    write_cross_asset_phase_residual_contrast,
    write_cross_asset_phase_residual_pair_blocks,
    write_cross_asset_phase_residual_pairs,
    write_cross_asset_proxy_block_sensitivity,
    write_cross_asset_proxy_block_stability,
    write_cross_asset_proxy_clusters,
    write_cross_asset_proxy_representative_invariance,
    write_cross_asset_proxy_sensitivity,
    write_cross_asset_residual_pair_blocks,
    write_cross_asset_residual_pairs,
)
from adaptive.intraday_report_audit import (
    REPORT_MANIFEST_FILENAME,
    STANDARD_INTRADAY_REPORT_FILENAMES,
    audit_intraday_report_directory,
    write_intraday_report_manifest,
)


STANDARD_REPORT_OUTPUTS = {
    "matrix_output": "intraday_feature_stability_matrix.csv",
    "regime_atlas_output": "intraday_regime_atlas.csv",
    "regime_transition_output": "intraday_regime_transitions.csv",
    "context_stability_output": "intraday_context_stability.csv",
    "lorentzian_stability_output": "lorentzian_neighborhood_stability.csv",
    "lorentzian_soft_surface_output": "lorentzian_soft_surface.csv",
    "lorentzian_soft_blocks_output": "lorentzian_soft_block_details.csv",
    "lorentzian_soft_stability_output": "lorentzian_soft_block_stability.csv",
    "cross_asset_dependence_output": "intraday_cross_asset_dependence.csv",
    "cross_asset_block_breadth_output": "intraday_cross_asset_block_breadth.csv",
    "cross_asset_breadth_output": "intraday_cross_asset_effective_breadth.csv",
    "cross_asset_factor_loadings_output": "intraday_common_factor_loadings.csv",
    "cross_asset_factor_blocks_output": "intraday_common_factor_blocks.csv",
    "cross_asset_factor_summary_output": "intraday_common_factor_summary.csv",
    "cross_asset_residual_blocks_output": "intraday_residual_pair_blocks.csv",
    "cross_asset_residual_pairs_output": "intraday_residual_pairs.csv",
    "cross_asset_proxy_clusters_output": "intraday_stable_proxy_clusters.csv",
    "cross_asset_proxy_sensitivity_output": "intraday_proxy_sensitivity.csv",
    "cross_asset_proxy_blocks_output": "intraday_proxy_block_sensitivity.csv",
    "cross_asset_proxy_block_stability_output": "intraday_proxy_block_stability.csv",
    "cross_asset_proxy_invariance_output": "intraday_proxy_representative_invariance.csv",
    "cross_asset_phase_factor_loadings_output": "intraday_phase_factor_loadings.csv",
    "cross_asset_phase_factor_blocks_output": "intraday_phase_factor_blocks.csv",
    "cross_asset_phase_factor_summary_output": "intraday_phase_factor_summary.csv",
    "cross_asset_phase_factor_contrast_output": "intraday_phase_factor_contrast.csv",
    "cross_asset_phase_residual_blocks_output": "intraday_phase_residual_pair_blocks.csv",
    "cross_asset_phase_residual_pairs_output": "intraday_phase_residual_pairs.csv",
    "cross_asset_phase_residual_contrast_output": "intraday_phase_residual_contrast.csv",
}
from adaptive.intraday_feature_readiness import (
    build_descriptive_feature_matrix,
    non_stable_evidence,
    summarize_asset_matrix,
    write_descriptive_feature_matrix,
)
from adaptive.intraday_regime_atlas import (
    IntradayRegimeAtlasConfig,
    build_intraday_regime_atlas,
    summarize_regime_atlas,
    write_regime_atlas,
    write_regime_transitions,
)
from adaptive.intraday_stability import (
    IntradayFeatureStabilityAnalyzer,
    IntradayStabilityConfig,
    add_dimensionless_squeeze_features,
    regular_session_frame,
    summarize_cross_asset_phase_consensus,
)
from adaptive.local_intraday_data import (
    LocalSourceReadiness,
    load_local_intraday_markets,
)
from adaptive.lorentzian_research import (
    CausalLorentzianResearchEngine,
    LorentzianResearchConfig,
    session_phase_context,
    session_slot_context,
    summarize_lorentzian_distribution,
)
from adaptive.lorentzian_neighborhood_stability import (
    LorentzianNeighborhoodStabilityConfig,
    analyze_lorentzian_neighborhood_stability,
    write_lorentzian_neighborhood_stability,
)
from adaptive.lorentzian_soft_surface import (
    LorentzianSoftSurfaceConfig,
    analyze_lorentzian_soft_surface,
    summarize_soft_surface_across_assets,
    write_lorentzian_soft_surface,
)
from adaptive.lorentzian_soft_block_stability import (
    LorentzianSoftBlockStabilityConfig,
    analyze_lorentzian_soft_block_stability,
    summarize_soft_block_stability_across_assets,
    write_lorentzian_soft_block_details,
    write_lorentzian_soft_block_stability,
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Valida ZIP 1m locali, crea barre 15m complete e misura feature, "
            "drift e geometria Lorentziana. Non calcola segnali o P&L."
        )
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory che contiene gli archivi TICKER_1min_*.zip.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker opzionali; se omessi vengono letti tutti gli ZIP compatibili.",
    )
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory opzionale che abilita automaticamente tutti i CSV "
            "descrittivi con nomi standard. I percorsi espliciti prevalgono."
        ),
    )
    parser.add_argument(
        "--matrix-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la matrice descrittiva "
            "asset x feature x fase."
        ),
    )
    parser.add_argument(
        "--regime-atlas-output",
        default=None,
        help=(
            "Percorso CSV opzionale per le frequenze descrittive dei "
            "contesti intraday osservati."
        ),
    )
    parser.add_argument(
        "--regime-transition-output",
        default=None,
        help=(
            "Percorso CSV opzionale per le transizioni intraseduta tra "
            "contesti consecutivi."
        ),
    )
    parser.add_argument(
        "--context-stability-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la persistenza descrittiva dei "
            "contesti e delle transizioni tra blocchi storici."
        ),
    )
    parser.add_argument(
        "--lorentzian-stability-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il confronto descrittivo del "
            "vicinato Lorentziano tra profondita' storiche diverse."
        ),
    )
    parser.add_argument(
        "--lorentzian-soft-surface-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la superficie descrittiva di "
            "decadimento temporale e penalita' morbida del contesto."
        ),
    )
    parser.add_argument(
        "--lorentzian-soft-blocks-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il dettaglio cronologico del "
            "profilo soft congelato tra blocchi di sessioni."
        ),
    )
    parser.add_argument(
        "--lorentzian-soft-stability-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la stabilita' descrittiva per asset "
            "del profilo soft congelato."
        ),
    )
    parser.add_argument(
        "--cross-asset-dependence-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la dipendenza descrittiva tra "
            "coppie di asset contemporanei."
        ),
    )
    parser.add_argument(
        "--cross-asset-block-breadth-output",
        default=None,
        help=(
            "Percorso CSV opzionale per l'ampiezza effettiva cross-asset "
            "nei blocchi cronologici fissi."
        ),
    )
    parser.add_argument(
        "--cross-asset-breadth-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il riepilogo dell'evidenza "
            "cross-asset effettivamente distinta."
        ),
    )
    parser.add_argument(
        "--cross-asset-factor-loadings-output",
        default=None,
        help=(
            "Percorso CSV opzionale per i contributi descrittivi degli "
            "asset alla prima componente comune."
        ),
    )
    parser.add_argument(
        "--cross-asset-factor-blocks-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la stabilita' della componente "
            "comune nei blocchi cronologici fissi."
        ),
    )
    parser.add_argument(
        "--cross-asset-factor-summary-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il riepilogo della componente "
            "comune e dell'ampiezza residua."
        ),
    )
    parser.add_argument(
        "--cross-asset-residual-blocks-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la dipendenza residua descrittiva "
            "tra coppie nei blocchi cronologici fissi."
        ),
    )
    parser.add_argument(
        "--cross-asset-residual-pairs-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il riepilogo descrittivo della "
            "dipendenza residua tra coppie."
        ),
    )
    parser.add_argument(
        "--cross-asset-proxy-clusters-output",
        default=None,
        help=(
            "Percorso CSV opzionale per i gruppi descrittivi di proxy "
            "stabili, senza scelta di un rappresentante."
        ),
    )
    parser.add_argument(
        "--cross-asset-proxy-sensitivity-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la sensibilita' descrittiva a tutte "
            "le combinazioni dei proxy stabili."
        ),
    )
    parser.add_argument(
        "--cross-asset-proxy-blocks-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la sensibilita' descrittiva dei "
            "proxy in ogni blocco cronologico fisso."
        ),
    )
    parser.add_argument(
        "--cross-asset-proxy-block-stability-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il riepilogo tra blocchi di ogni "
            "scenario proxy dichiarato."
        ),
    )
    parser.add_argument(
        "--cross-asset-proxy-invariance-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la dispersione tra tutte le scelte "
            "possibili dei rappresentanti proxy, senza selezione."
        ),
    )
    parser.add_argument(
        "--cross-asset-phase-factor-loadings-output",
        default=None,
        help=(
            "Percorso CSV opzionale per i contributi alla componente comune "
            "separati tra OPEN, MID_SESSION e CLOSE."
        ),
    )
    parser.add_argument(
        "--cross-asset-phase-factor-blocks-output",
        default=None,
        help=(
            "Percorso CSV opzionale per la componente comune per fase nei "
            "blocchi cronologici fissi."
        ),
    )
    parser.add_argument(
        "--cross-asset-phase-factor-summary-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il riepilogo cross-asset separato "
            "per fase della seduta."
        ),
    )
    parser.add_argument(
        "--cross-asset-phase-factor-contrast-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il contrasto descrittivo tra le "
            "tre fasi della seduta."
        ),
    )
    parser.add_argument(
        "--cross-asset-phase-residual-blocks-output",
        default=None,
        help=(
            "Percorso CSV opzionale per le correlazioni raw e residue "
            "separate per fase e blocco cronologico."
        ),
    )
    parser.add_argument(
        "--cross-asset-phase-residual-pairs-output",
        default=None,
        help=(
            "Percorso CSV opzionale per le dipendenze residue tra coppie "
            "separate tra OPEN, MID_SESSION e CLOSE."
        ),
    )
    parser.add_argument(
        "--cross-asset-phase-residual-contrast-output",
        default=None,
        help=(
            "Percorso CSV opzionale per il contrasto della dipendenza "
            "residua tra le tre fasi della seduta."
        ),
    )
    parser.add_argument(
        "--context-minimum-observations",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--context-minimum-sessions",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--context-minimum-complete-blocks",
        type=int,
        default=3,
    )
    return _resolve_standard_outputs(parser.parse_args(argv))


def _resolve_standard_outputs(arguments):
    if not arguments.output_dir:
        return arguments
    # Mantiene una rappresentazione stabile e portabile nei valori CLI e nei
    # manifest, indipendentemente dal separatore nativo del sistema operativo.
    # Windows accetta i forward slash per questi percorsi e Path li risolve
    # correttamente quando i file vengono effettivamente scritti.
    root = Path(str(arguments.output_dir).replace("\\", "/")).expanduser().as_posix()
    arguments.output_dir = root
    for attribute, filename in STANDARD_REPORT_OUTPUTS.items():
        if getattr(arguments, attribute) is None:
            setattr(arguments, attribute, str(PurePosixPath(root) / filename))
    return arguments


def _standard_outputs_share_directory(arguments) -> bool:
    """True quando l'output-dir contiene davvero tutti i report standard."""

    if not arguments.output_dir:
        return False
    root = Path(arguments.output_dir).resolve()
    return all(
        Path(getattr(arguments, attribute)).resolve().parent == root
        for attribute in STANDARD_REPORT_OUTPUTS
    )


def _percentage(series: pd.Series, value: str) -> float:
    valid = series.loc[series != "INSUFFICIENT"]
    return float((valid == value).mean()) if not valid.empty else math.nan


def _persistent_counts(table: pd.DataFrame) -> tuple[int, int]:
    if table.empty:
        return 0, 0
    persistent = table["pattern"].isin(
        {"PERSISTENT_ELEVATED", "PERSISTENT_HIGH"}
    )
    high = table["pattern"].eq("PERSISTENT_HIGH")
    return int(persistent.sum()), int(high.sum())


def _lorentzian_history_limits(maximum: int) -> tuple[int, ...]:
    """Costruisce finestre annidate mantenendo il limite CLI come riferimento."""

    maximum = int(maximum)
    canonical = (520, 1_040, 2_080)
    return tuple(value for value in canonical if value < maximum) + (maximum,)


def _lorentzian_half_lives(maximum: int) -> tuple[int, ...]:
    """Mantiene la griglia canonica entro la memoria storica dichiarata."""

    maximum = int(maximum)
    values = tuple(value for value in (520, 1_040, 2_080) if value <= maximum)
    return values or (maximum,)


def _display_number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if math.isfinite(float(value)) else "N/A"


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    stability_config = IntradayStabilityConfig(
        sessions_per_block=arguments.sessions_per_block,
        minimum_complete_blocks=arguments.minimum_complete_blocks,
    )
    lorentzian_config = LorentzianResearchConfig(
        normalization_window=arguments.normalization_window,
        normalization_min_periods=arguments.normalization_min_periods,
        neighbors=arguments.neighbors,
        minimum_candidates=arguments.minimum_candidates,
        embargo_bars=arguments.embargo_bars,
        sample_stride=arguments.sample_stride,
        history_limit=arguments.history_limit,
    )
    regime_atlas_config = IntradayRegimeAtlasConfig(
        sessions_per_block=arguments.sessions_per_block,
        minimum_observations=arguments.context_minimum_observations,
        minimum_sessions=arguments.context_minimum_sessions,
        minimum_complete_blocks=arguments.context_minimum_complete_blocks,
    )
    context_stability_config = IntradayContextStabilityConfig(
        minimum_complete_blocks=arguments.minimum_complete_blocks,
    )
    lorentzian_stability_config = LorentzianNeighborhoodStabilityConfig(
        history_limits=_lorentzian_history_limits(arguments.history_limit),
        neighbors=arguments.neighbors,
        minimum_candidates=arguments.minimum_candidates,
        embargo_bars=arguments.embargo_bars,
        sample_stride=arguments.sample_stride,
    )
    lorentzian_soft_config = LorentzianSoftSurfaceConfig(
        history_limit=arguments.history_limit,
        recency_half_lives=_lorentzian_half_lives(arguments.history_limit),
        neighbors=arguments.neighbors,
        minimum_candidates=arguments.minimum_candidates,
        embargo_bars=arguments.embargo_bars,
        sample_stride=arguments.sample_stride,
    )
    lorentzian_soft_block_config = LorentzianSoftBlockStabilityConfig(
        sessions_per_block=arguments.sessions_per_block,
        minimum_complete_blocks=arguments.context_minimum_complete_blocks,
        neighbors=arguments.neighbors,
    )
    cross_asset_dependence_config = IntradayCrossAssetDependenceConfig(
        sessions_per_block=arguments.sessions_per_block,
        minimum_complete_blocks=arguments.minimum_complete_blocks,
    )
    markets, source_audits, source_errors = load_local_intraday_markets(
        arguments.data_dir,
        arguments.tickers,
    )
    if not markets:
        raise ValueError("Nessun archivio locale ha superato la validazione.")

    print("\nTRADINGAI LOCAL 1M -> 15M RESEARCH PACK — DESCRIPTIVE ONLY")
    print("Timezone sorgente: America/New_York; sessione regolare soltanto.")
    print("I bucket incompleti vengono esclusi, mai riempiti o interpolati.")
    print(
        "Lorentzian: normalizzazione trailing per slot 15m e vicini della "
        "stessa fase di sessione."
    )
    print(
        "Nessuna direzione, previsione, operazione, size, stop, leva, outcome "
        "futuro o P&L."
    )
    print("\nQUALITA' SORGENTE E RESAMPLING")
    print(
        f"{'TICKER':9} {'STATUS':18} {'SESS':>5} {'1M':>8} {'15M':>6} "
        f"{'MED_COV':>8} {'MIN_COV':>8} {'FULL15':>8} {'EARLY':>5} {'MISS':>6}"
    )
    print("-" * 104)
    for ticker, audit in sorted(source_audits.items()):
        print(
            f"{ticker:9} {audit.status.value:18} "
            f"{audit.observed_sessions:5d} {audit.regular_session_rows:8d} "
            f"{audit.output_bars:6d} {audit.median_minute_coverage:8.1%} "
            f"{audit.minimum_minute_coverage:8.1%} "
            f"{audit.complete_bucket_fraction:8.1%} "
            f"{audit.early_close_sessions:5d} "
            f"{audit.estimated_missing_minutes:6d}"
        )

    feature_engine = IntradayReferenceFeatureEngine()
    feature_rows: list[tuple] = []
    stability_rows: list[tuple] = []
    lorentzian_rows: list[tuple] = []
    lorentzian_stability_reports = []
    lorentzian_soft_reports = []
    lorentzian_soft_block_reports = []
    stability_reports = []
    feature_frames: dict[str, pd.DataFrame] = {}
    analysis_errors: dict[str, str] = {}

    for ticker, market in sorted(markets.items()):
        audit = source_audits[ticker]
        try:
            numeric_features = stability_config.numeric_features
            lorentzian_features = lorentzian_config.feature_columns
            mode = "OHLCV"
            if not audit.has_volume:
                numeric_features = tuple(
                    name for name in numeric_features if name != "cmf"
                )
                lorentzian_features = tuple(
                    name for name in lorentzian_features if name != "cmf"
                )
                mode = "PRICE"
            asset_stability_config = replace(
                stability_config,
                numeric_features=numeric_features,
            )
            asset_lorentzian_config = replace(
                lorentzian_config,
                feature_columns=lorentzian_features,
            )
            regular, _ = regular_session_frame(
                market,
                asset_stability_config,
            )
            feature_report = feature_engine.compute(
                regular,
                allow_price_only=not audit.has_volume,
            )
            features = add_dimensionless_squeeze_features(
                feature_report.values,
                regular,
                asset_stability_config,
            )
            feature_frames[ticker] = features
            feature_rows.append(
                (
                    ticker,
                    mode,
                    len(features),
                    int(features["squeeze_momentum"].notna().sum()),
                    int(features["choppiness"].notna().sum()),
                    int(features["cmf"].notna().sum()),
                    _percentage(features["chop_segment"], "CHOPPY"),
                    _percentage(features["chop_segment"], "TRENDING"),
                    _percentage(features["squeeze_state"], "SQUEEZE_ON"),
                )
            )

            stability = IntradayFeatureStabilityAnalyzer(
                asset_stability_config
            ).analyze(ticker, features)
            stability_reports.append(stability)
            numeric_persistent, numeric_high = _persistent_counts(
                stability.numeric_persistence
            )
            state_persistent, state_high = _persistent_counts(
                stability.state_persistence
            )
            phase_persistent, phase_high = _persistent_counts(
                stability.phase_persistence
            )
            stability_rows.append(
                (
                    ticker,
                    stability.observed_sessions,
                    stability.complete_blocks,
                    numeric_persistent,
                    numeric_high,
                    state_persistent,
                    state_high,
                    phase_persistent,
                    phase_high,
                )
            )

            context = session_phase_context(features.index)
            normalization_context = session_slot_context(features.index)
            lorentzian = CausalLorentzianResearchEngine(
                asset_lorentzian_config
            ).compute(
                features,
                context,
                normalization_context,
            )
            summary = summarize_lorentzian_distribution(lorentzian)
            lorentzian_rows.append(
                (
                    ticker,
                    len(asset_lorentzian_config.feature_columns),
                    summary.total_rows,
                    summary.complete_rows,
                    summary.coverage,
                    summary.median_distance,
                    summary.distance_p90,
                    summary.median_density,
                    summary.median_neighbor_age_bars,
                )
            )
            lorentzian_stability_reports.append(
                analyze_lorentzian_neighborhood_stability(
                    ticker,
                    lorentzian.normalized_features,
                    features,
                    lorentzian_stability_config,
                )
            )
            lorentzian_soft = analyze_lorentzian_soft_surface(
                ticker,
                lorentzian.normalized_features,
                features,
                lorentzian_soft_config,
            )
            lorentzian_soft_reports.append(lorentzian_soft)
            if 2_080 in lorentzian_soft_config.recency_half_lives:
                lorentzian_soft_block_reports.append(
                    analyze_lorentzian_soft_block_stability(
                        ticker,
                        lorentzian_soft.details,
                        features.index,
                        lorentzian_soft_block_config,
                    )
                )
        except Exception as exc:
            analysis_errors[ticker] = f"{type(exc).__name__}: {exc}"

    print("\nDISTRIBUZIONI FEATURE 15M")
    print(
        f"{'TICKER':9} {'MODE':>6} {'ROWS':>6} {'SQZ_N':>7} "
        f"{'CHOP_N':>7} {'CMF_N':>7} "
        f"{'CHOPPY':>9} {'TREND':>9} {'SQZ_ON':>9}"
    )
    print("-" * 88)
    for row in feature_rows:
        (
            ticker,
            mode,
            total,
            squeeze_n,
            chop_n,
            cmf_n,
            choppy,
            trend,
            squeeze_on,
        ) = row
        print(
            f"{ticker:9} {mode:>6} {total:6d} {squeeze_n:7d} "
            f"{chop_n:7d} {cmf_n:7d} "
            f"{choppy:9.1%} {trend:9.1%} {squeeze_on:9.1%}"
        )

    print("\nSTABILITA' TRA BLOCCHI STORICI")
    print(
        f"{'TICKER':9} {'SESS':>5} {'BLOCK':>5} {'N_PERS':>7} {'N_HIGH':>7} "
        f"{'S_PERS':>7} {'S_HIGH':>7} {'P_PERS':>7} {'P_HIGH':>7}"
    )
    print("-" * 78)
    for row in stability_rows:
        print(
            f"{row[0]:9} {row[1]:5d} {row[2]:5d} {row[3]:7d} {row[4]:7d} "
            f"{row[5]:7d} {row[6]:7d} {row[7]:7d} {row[8]:7d}"
        )
    print("N=feature numeriche; S=stati; P=feature per fase di sessione.")

    print("\nGEOMETRIA LORENTZIANA CAUSALE")
    print(
        f"{'TICKER':9} {'F':>2} {'ROWS':>6} {'COMP':>6} {'COVER':>8} "
        f"{'D_MED':>9} "
        f"{'D_P90':>9} {'DENS':>8} {'AGE_MED':>9}"
    )
    print("-" * 85)
    for row in lorentzian_rows:
        print(
            f"{row[0]:9} {row[1]:2d} {row[2]:6d} {row[3]:6d} "
            f"{row[4]:8.1%} {row[5]:9.4f} {row[6]:9.4f} "
            f"{row[7]:8.4f} {row[8]:9.1f}"
        )

    if lorentzian_stability_reports:
        lorentzian_stability = pd.concat(
            [report.summary for report in lorentzian_stability_reports],
            ignore_index=True,
        )
        shortest = lorentzian_stability.loc[
            lorentzian_stability["history_limit"].eq(
                min(lorentzian_stability_config.history_limits)
            )
        ]
        print("\nROBUSTEZZA DEL VICINATO LORENTZIANO ALLA MEMORIA")
        print(
            "Confronto della finestra piu' breve con il riferimento massimo; "
            "nessun outcome futuro e' utilizzato."
        )
        print(
            f"{'TICKER':9} {'FILTRO':13} {'H_S':>5} {'H_R':>5} "
            f"{'COVER':>7} {'AGE_S':>8} {'AGE_R':>8} "
            f"{'OV_MED':>7} {'OV_P10':>7} {'D_RATIO':>8} {'STATO':>29}"
        )
        print("-" * 119)
        for row in shortest.itertuples(index=False):
            print(
                f"{row.ticker:9} {row.filter_mode:13} "
                f"{row.history_limit:5d} {row.reference_history_limit:5d} "
                f"{row.coverage:7.1%} "
                f"{_display_number(row.median_neighbor_age_bars, 1):>8} "
                f"{_display_number(row.reference_median_neighbor_age_bars, 1):>8} "
                f"{_display_number(row.median_overlap_reference):>7} "
                f"{_display_number(row.p10_overlap_reference):>7} "
                f"{_display_number(row.median_distance_ratio_reference):>8} "
                f"{row.overlap_state:>29}"
            )
        print(
            "PHASE_ONLY usa la stessa fase di seduta; JOINT_CONTEXT richiede "
            "anche gli stessi stati CHOP e Squeeze."
        )
        print(
            "La sovrapposizione descrive la sensibilita' della geometria alla "
            "memoria, non accuratezza o redditivita'."
        )
        if arguments.lorentzian_stability_output:
            destination = write_lorentzian_neighborhood_stability(
                lorentzian_stability,
                arguments.lorentzian_stability_output,
            )
            print(f"\nCSV ROBUSTEZZA LORENTZIANA: {destination}")
    elif arguments.lorentzian_stability_output:
        raise ValueError(
            "Nessun riepilogo Lorentziano disponibile per il CSV."
        )

    if lorentzian_soft_reports:
        lorentzian_soft = pd.concat(
            [report.summary for report in lorentzian_soft_reports],
            ignore_index=True,
        )
        cross_asset_soft = summarize_soft_surface_across_assets(
            lorentzian_soft
        )
        print("\nSUPERFICIE LORENTZIANA MORBIDA — MEDIANE CROSS-ASSET")
        print(
            "Decadimento temporale e disaccordo CHOP/Squeeze modificano "
            "soltanto la distanza geometrica."
        )
        print(
            f"{'PROFILO':21} {'HL':>5} {'CTX':>5} {'COVER':>7} "
            f"{'AGE':>7} {'CTX_OK':>7} {'MISM':>6} {'SESS':>5} "
            f"{'OV_REF':>7} {'RAW_D':>7} {'ADJ_D':>7}"
        )
        print("-" * 94)
        for row in cross_asset_soft.itertuples(index=False):
            print(
                f"{row.profile:21} {row.recency_half_life_bars:5d} "
                f"{row.context_penalty:5.2f} {row.median_coverage:7.1%} "
                f"{row.median_neighbor_age_bars:7.1f} "
                f"{row.median_joint_context_match_fraction:7.1%} "
                f"{row.median_context_mismatches:6.3f} "
                f"{row.median_neighbor_session_count:5.1f} "
                f"{row.median_overlap_raw_reference:7.3f} "
                f"{row.median_raw_distance:7.3f} "
                f"{row.median_adjusted_distance:7.3f}"
            )
        print(
            "Le mediane cross-asset sono descrittive; gli strumenti correlati "
            "non costituiscono osservazioni indipendenti."
        )
        print(
            "La superficie non seleziona un profilo e non misura accuratezza "
            "o redditivita'."
        )
        if arguments.lorentzian_soft_surface_output:
            destination = write_lorentzian_soft_surface(
                lorentzian_soft,
                arguments.lorentzian_soft_surface_output,
            )
            print(f"\nCSV SUPERFICIE LORENTZIANA: {destination}")
    elif arguments.lorentzian_soft_surface_output:
        raise ValueError(
            "Nessuna superficie Lorentziana disponibile per il CSV."
        )

    if lorentzian_soft_block_reports:
        soft_blocks = pd.concat(
            [report.blocks for report in lorentzian_soft_block_reports],
            ignore_index=True,
        )
        soft_block_summary = pd.concat(
            [report.summary for report in lorentzian_soft_block_reports],
            ignore_index=True,
        )
        cross_asset_blocks = summarize_soft_block_stability_across_assets(
            soft_block_summary
        )
        print("\nSTABILITA' DEL PROFILO LORENTZIANO SOFT PER BLOCCHI")
        print(
            "Profilo congelato HL_2080_CTX_0.5; blocchi cronologici fissi e "
            "nessun outcome futuro."
        )
        print(
            f"{'TICKER':9} {'BLOCK':>5} {'MIN_COV':>8} "
            f"{'RAW_CHG':>8} {'AGE_CHG':>8} {'CTX_CHG':>8} "
            f"{'OV_CHG':>8} {'EFF_MIN':>8} {'MAX_SHARE':>9} {'STATO':>28}"
        )
        print("-" * 112)
        for row in soft_block_summary.itertuples(index=False):
            print(
                f"{row.ticker:9} {row.eligible_blocks:5d} "
                f"{_display_number(row.minimum_block_coverage):>8} "
                f"{_display_number(row.max_raw_distance_relative_change):>8} "
                f"{_display_number(row.max_neighbor_age_relative_change):>8} "
                f"{_display_number(row.max_context_match_absolute_change):>8} "
                f"{_display_number(row.max_overlap_absolute_change):>8} "
                f"{_display_number(row.minimum_effective_session_count):>8} "
                f"{_display_number(row.maximum_neighbor_session_share):>9} "
                f"{row.stability_state:>28}"
            )
        cross = cross_asset_blocks.iloc[0]
        print(
            "Riepilogo asset: "
            f"LOW={int(cross.low_variation_assets)}, "
            f"MODERATE={int(cross.moderate_variation_assets)}, "
            f"HIGH={int(cross.high_variation_assets)}, "
            f"INSUFFICIENT={int(cross.insufficient_assets)}."
        )
        print(
            "Gli stati misurano variazione geometrica tra blocchi; non "
            "accuratezza, rendimento o idoneita' operativa."
        )
        if arguments.lorentzian_soft_blocks_output:
            destination = write_lorentzian_soft_block_details(
                soft_blocks,
                arguments.lorentzian_soft_blocks_output,
            )
            print(f"\nCSV DETTAGLIO BLOCCHI SOFT: {destination}")
        if arguments.lorentzian_soft_stability_output:
            destination = write_lorentzian_soft_block_stability(
                soft_block_summary,
                arguments.lorentzian_soft_stability_output,
            )
            print(f"CSV STABILITA' SOFT: {destination}")
    elif (
        arguments.lorentzian_soft_blocks_output
        or arguments.lorentzian_soft_stability_output
    ):
        raise ValueError(
            "Il profilo HL_2080_CTX_0.5 richiede history_limit almeno 2080."
        )

    price_only = sorted(
        ticker
        for ticker, audit in source_audits.items()
        if audit.status is LocalSourceReadiness.PRICE_ONLY
    )
    if price_only:
        print("\nASSET PRICE-ONLY SEPARATI")
        print(
            "- "
            + ", ".join(price_only)
            + ": Squeeze/CHOP inclusi; CMF assente e Lorentzian a tre feature."
        )
    if "VXX" in source_audits:
        print("\nSERIE A CONTESTO SPECIALE")
        print(
            "- VXX resta separato dagli ETF/azioni ordinari: struttura e "
            "copertura non sono direttamente confrontabili."
        )

    if stability_reports:
        consensus = summarize_cross_asset_phase_consensus(stability_reports)
        common = consensus.loc[
            consensus["cross_asset_scope"].eq("COMMON_SHIFT")
        ]
        print("\nSHIFT COMUNI TRA ASSET PER FASE")
        if common.empty:
            print("- Nessuno shift recente comune a tutti gli asset analizzati.")
        else:
            for row in common.itertuples(index=False):
                print(
                    f"- {row.feature} / {row.phase}: "
                    f"{row.latest_elevated_assets}/{row.sufficient_assets} asset."
                )
        print("Gli asset correlati non vengono trattati come prove indipendenti.")

        matrix = build_descriptive_feature_matrix(
            stability_reports,
            source_audits,
        )
        asset_matrix = summarize_asset_matrix(matrix)
        print("\nMATRICE DI STABILITA' DESCRITTIVA PER ASSET")
        print(
            f"{'TICKER':9} {'SOURCE':18} {'STABLE':>6} {'ISOL':>5} "
            f"{'CONTEXT':>7} {'INSUFF':>6} {'N/A':>4} {'OVERALL':>25}"
        )
        print("-" * 91)
        for row in asset_matrix.itertuples(index=False):
            print(
                f"{row.ticker:9} {row.source_status:18} {row.stable:6d} "
                f"{row.isolated:5d} {row.context_required:7d} "
                f"{row.insufficient:6d} {row.not_available:4d} "
                f"{row.overall:>25}"
            )

        evidence = non_stable_evidence(matrix)
        print("\nDRIFT DA CONTESTUALIZZARE")
        if evidence.empty:
            print("- Nessun drift isolato o persistente nel campione.")
        else:
            print(
                f"{'TICKER':9} {'FEATURE':38} {'SCOPE':12} "
                f"{'PATTERN':22} {'LATEST':14}"
            )
            print("-" * 101)
            for row in evidence.itertuples(index=False):
                print(
                    f"{row.ticker:9} {row.feature:38} {row.scope:12} "
                    f"{row.pattern:22} {row.latest_shift:14}"
                )

        if arguments.matrix_output:
            destination = write_descriptive_feature_matrix(
                matrix,
                arguments.matrix_output,
            )
            print(f"\nCSV MATRICE: {destination}")
    elif arguments.matrix_output:
        raise ValueError("Nessun report di stabilita' disponibile per il CSV.")

    if len(feature_frames) >= 2:
        cross_asset_dependence = analyze_intraday_cross_asset_dependence(
            markets,
            feature_frames,
            source_audits,
            cross_asset_dependence_config,
        )
        pair_summary = (
            cross_asset_dependence.pairs["dependence_state"]
            .value_counts()
            .to_dict()
        )
        ranked_pairs = cross_asset_dependence.pairs.copy()
        ranked_pairs["absolute_correlation"] = ranked_pairs[
            "close_change_correlation"
        ].abs()
        ranked_pairs = ranked_pairs.sort_values(
            ["absolute_correlation", "joint_context_nmi", "ticker_a", "ticker_b"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).head(15)

        print("\nDIPENDENZA CROSS-ASSET CONTEMPORANEA")
        print(
            "Variazioni di close gia' concluse nella stessa seduta e stati "
            "contemporanei; nessun outcome futuro."
        )
        print(
            "Coppie: "
            f"HIGH={int(pair_summary.get('HIGH_DEPENDENCE', 0))}, "
            f"MODERATE={int(pair_summary.get('MODERATE_DEPENDENCE', 0))}, "
            f"LOW={int(pair_summary.get('LOW_DEPENDENCE', 0))}, "
            f"INSUFFICIENT={int(pair_summary.get('INSUFFICIENT_OVERLAP', 0))}."
        )
        print(
            f"{'COPPIA':19} {'OBS':>6} {'CORR':>8} {'BLOCK_MED':>10} "
            f"{'CTX_NMI':>8} {'JOINT':>7} {'STATO':>23}"
        )
        print("-" * 91)
        for row in ranked_pairs.itertuples(index=False):
            print(
                f"{row.ticker_a + '/' + row.ticker_b:19} "
                f"{row.observations:6d} "
                f"{_display_number(row.close_change_correlation):>8} "
                f"{_display_number(row.median_block_close_change_correlation):>10} "
                f"{_display_number(row.joint_context_nmi):>8} "
                f"{_display_number(row.joint_state_agreement):>7} "
                f"{row.dependence_state:>23}"
            )

        print("\nAMPIEZZA EFFETTIVA DELL'EVIDENZA CROSS-ASSET")
        print(
            "La conta effettiva deriva dagli autovalori della correlazione; "
            "non equivale al numero nominale di ticker."
        )
        print(
            f"{'SCOPE':24} {'N':>3} {'OBS':>6} {'EFF_N':>7} {'EFF_%':>7} "
            f"{'DOM':>7} {'MED_ABS':>8} {'P90_ABS':>8} {'STATO':>27}"
        )
        print("-" * 107)
        for row in cross_asset_dependence.breadth.itertuples(index=False):
            print(
                f"{row.scope:24} {row.assets:3d} {row.observations:6d} "
                f"{_display_number(row.effective_asset_count):>7} "
                f"{_display_number(row.effective_asset_fraction):>7} "
                f"{_display_number(row.dominant_component_share):>7} "
                f"{_display_number(row.median_pair_absolute_correlation):>8} "
                f"{_display_number(row.p90_pair_absolute_correlation):>8} "
                f"{row.breadth_state:>27}"
            )
        if cross_asset_dependence.breadth["scope"].eq(
            "ORDINARY_ASSETS"
        ).any():
            print(
                "VXX resta separato nello scope ordinario. Dipendenza "
                "condivisa non misura accuratezza o redditivita'."
            )
        else:
            print(
                "Dipendenza condivisa non misura accuratezza o redditivita'."
            )

        print("\nDECOMPOSIZIONE DELLA COMPONENTE COMUNE")
        print(
            "La prima componente separa il movimento contemporaneo condiviso "
            "dalla variazione residua; nessun dato futuro e' usato."
        )
        print(
            f"{'SCOPE':24} {'N':>3} {'COMMON':>8} {'RAW_EFF':>8} "
            f"{'RES_EFF':>8} {'RAW_MED':>8} {'RES_MED':>8} "
            f"{'COS_MIN':>8} {'STATO':>22}"
        )
        print("-" * 111)
        for row in cross_asset_dependence.factor_summary.itertuples(index=False):
            print(
                f"{row.scope:24} {row.assets:3d} "
                f"{_display_number(row.common_factor_share):>8} "
                f"{_display_number(row.raw_effective_asset_count):>8} "
                f"{_display_number(row.residual_effective_asset_count):>8} "
                f"{_display_number(row.raw_median_pair_absolute_correlation):>8} "
                f"{_display_number(row.residual_median_pair_absolute_correlation):>8} "
                f"{_display_number(row.minimum_adjacent_loading_cosine):>8} "
                f"{row.common_mode_state:>22}"
            )
        ordinary_scope = (
            "ORDINARY_ASSETS"
            if cross_asset_dependence.factor_loadings["scope"].eq(
                "ORDINARY_ASSETS"
            ).any()
            else "ALL_ASSETS_DESCRIPTIVE"
        )
        ordinary_loadings = cross_asset_dependence.factor_loadings.loc[
            cross_asset_dependence.factor_loadings["scope"].eq(ordinary_scope)
        ].sort_values("absolute_loading_rank", kind="mergesort")
        print(f"\nCONTRIBUTI ALLA COMPONENTE COMUNE — {ordinary_scope}")
        print(
            f"{'TICKER':9} {'LOAD':>8} {'ABS':>8} {'SHARE':>8} "
            f"{'COMMON_VAR':>11} {'RANK':>5}"
        )
        print("-" * 57)
        for row in ordinary_loadings.itertuples(index=False):
            print(
                f"{row.ticker:9} {row.loading:8.4f} "
                f"{row.absolute_loading:8.4f} {row.loading_share:8.1%} "
                f"{row.common_variance_share:11.1%} "
                f"{row.absolute_loading_rank:5d}"
            )
        print(
            "La componente comune e i residui sono descrittivi: non sono "
            "segnali e non selezionano asset."
        )

        residual_scope = ordinary_scope
        residual_pairs = cross_asset_dependence.residual_pairs.loc[
            cross_asset_dependence.residual_pairs["scope"].eq(residual_scope)
        ].copy()
        residual_pairs["absolute_residual_correlation"] = residual_pairs[
            "residual_correlation"
        ].abs()
        residual_pairs = residual_pairs.sort_values(
            [
                "stable_proxy_link",
                "absolute_residual_correlation",
                "ticker_a",
                "ticker_b",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).head(15)
        residual_summary = (
            cross_asset_dependence.residual_pairs.loc[
                cross_asset_dependence.residual_pairs["scope"].eq(
                    residual_scope
                ),
                "residual_pair_state",
            ]
            .value_counts()
            .to_dict()
        )
        print(f"\nDIPENDENZA RESIDUA TRA COPPIE — {residual_scope}")
        print(
            "La prima componente e' stimata anche separatamente in ogni "
            "blocco completo; i residui sono diagnostici del sottospazio."
        )
        print(
            "Stati: "
            f"PROXY={int(residual_summary.get('STABLE_PROXY_LINK', 0))}, "
            f"HIGH={int(residual_summary.get('HIGH_RESIDUAL_DEPENDENCE', 0))}, "
            f"MODERATE={int(residual_summary.get('MODERATE_RESIDUAL_DEPENDENCE', 0))}, "
            f"LOW={int(residual_summary.get('LOW_RESIDUAL_DEPENDENCE', 0))}, "
            f"INSUFFICIENT={int(residual_summary.get('INSUFFICIENT_RESIDUAL_OVERLAP', 0))}."
        )
        print(
            f"{'COPPIA':19} {'OBS':>6} {'RAW':>8} {'RAW_MIN':>8} "
            f"{'RES':>8} {'RES_MED':>8} {'RES_MIN':>8} {'RES_MAX':>8} "
            f"{'STATO':>28}"
        )
        print("-" * 119)
        for row in residual_pairs.itertuples(index=False):
            print(
                f"{row.ticker_a + '/' + row.ticker_b:19} "
                f"{row.observations:6d} "
                f"{_display_number(row.raw_correlation):>8} "
                f"{_display_number(row.minimum_block_raw_correlation):>8} "
                f"{_display_number(row.residual_correlation):>8} "
                f"{_display_number(row.median_block_residual_correlation):>8} "
                f"{_display_number(row.minimum_block_residual_correlation):>8} "
                f"{_display_number(row.maximum_block_residual_correlation):>8} "
                f"{row.residual_pair_state:>28}"
            )
        print(
            "I gruppi proxy usano soltanto correlazione raw elevata e stabile "
            "nei blocchi; la correlazione residua non decide il gruppo."
        )

        cluster_rows = cross_asset_dependence.proxy_clusters.loc[
            cross_asset_dependence.proxy_clusters["scope"].eq(
                residual_scope
            )
        ]
        cluster_summary = (
            cluster_rows.loc[
                :,
                [
                    "cluster_id",
                    "cluster_members",
                    "member_count",
                    "cluster_state",
                ],
            ]
            .drop_duplicates()
            .sort_values(["member_count", "cluster_id"], ascending=[False, True])
        )
        print(f"\nGRUPPI DI PROXY STABILI — {residual_scope}")
        print(f"{'ID':5} {'N':>3} {'MEMBRI':45} {'STATO':>34}")
        print("-" * 91)
        for row in cluster_summary.itertuples(index=False):
            print(
                f"{row.cluster_id:5} {row.member_count:3d} "
                f"{row.cluster_members:45} {row.cluster_state:>34}"
            )

        sensitivity = cross_asset_dependence.proxy_sensitivity.loc[
            cross_asset_dependence.proxy_sensitivity["scope"].eq(
                residual_scope
            )
        ]
        print(f"\nSENSIBILITA' AI PROXY — {residual_scope}")
        print(
            "Ogni possibile membro dei gruppi stabili viene mantenuto a "
            "turno; nessun rappresentante e' scelto automaticamente."
        )
        print(
            f"{'SCENARIO':15} {'TIPO':11} {'N':>3} {'EFF_N':>8} "
            f"{'COMMON':>8} {'RES_EFF':>8} {'RAW_MED':>8} "
            f"{'RES_MED':>8} {'STATO':>40}"
        )
        print("-" * 123)
        for row in sensitivity.itertuples(index=False):
            scenario_type = {
                "BASELINE": "BASELINE",
                "COLLAPSE_STABLE_PROXY_GROUPS": "COLLAPSE",
                "LEAVE_STABLE_PROXY_GROUP_OUT": "LEAVE_OUT",
            }.get(row.scenario_type, row.scenario_type)
            print(
                f"{row.scenario_id:15} {scenario_type:11} {row.assets:3d} "
                f"{_display_number(row.effective_asset_count):>8} "
                f"{_display_number(row.common_factor_share):>8} "
                f"{_display_number(row.residual_effective_asset_count):>8} "
                f"{_display_number(row.median_pair_absolute_correlation):>8} "
                f"{_display_number(row.residual_median_pair_absolute_correlation):>8} "
                f"{row.scenario_state:>40}"
            )
        print(
            "Questa sensibilita' misura soltanto quanto cambiano le statistiche "
            "descrittive quando proxy ridondanti non vengono contati insieme."
        )

        proxy_block_stability = (
            cross_asset_dependence.proxy_block_stability.loc[
                cross_asset_dependence.proxy_block_stability["scope"].eq(
                    residual_scope
                )
                & cross_asset_dependence.proxy_block_stability[
                    "scenario_type"
                ].isin(["BASELINE", "COLLAPSE_STABLE_PROXY_GROUPS"])
            ]
        )
        print(f"\nSTABILITA' TEMPORALE DEGLI SCENARI PROXY — {residual_scope}")
        print(
            "Le stesse combinazioni sono ricalcolate indipendentemente in "
            "ogni blocco completo."
        )
        print(
            f"{'SCENARIO':15} {'BLOCK':>5} {'EFF_MIN':>8} {'EFF_MED':>8} "
            f"{'EFF_MAX':>8} {'COM_MIN':>8} {'COM_MED':>8} {'COM_MAX':>8} "
            f"{'RES_MIN':>8} {'RES_MED':>8} {'RES_MAX':>8} {'STATO':>29}"
        )
        print("-" * 137)
        for row in proxy_block_stability.itertuples(index=False):
            print(
                f"{row.scenario_id:15} {row.eligible_blocks:5d} "
                f"{_display_number(row.minimum_effective_asset_count):>8} "
                f"{_display_number(row.median_effective_asset_count):>8} "
                f"{_display_number(row.maximum_effective_asset_count):>8} "
                f"{_display_number(row.minimum_common_factor_share):>8} "
                f"{_display_number(row.median_common_factor_share):>8} "
                f"{_display_number(row.maximum_common_factor_share):>8} "
                f"{_display_number(row.minimum_residual_effective_asset_count):>8} "
                f"{_display_number(row.median_residual_effective_asset_count):>8} "
                f"{_display_number(row.maximum_residual_effective_asset_count):>8} "
                f"{row.block_stability_state:>29}"
            )

        proxy_invariance = (
            cross_asset_dependence.proxy_representative_invariance.loc[
                cross_asset_dependence.proxy_representative_invariance[
                    "scope"
                ].eq(residual_scope)
            ]
        )
        print(f"\nINVARIANZA DELLA SCELTA DEI PROXY — {residual_scope}")
        print(
            "La dispersione e' il massimo meno il minimo tra tutte le "
            "combinazioni nello stesso blocco."
        )
        print(
            f"{'BLOCK':>5} {'SESSIONI':>8} {'SCEN':>5} {'EFF_RNG':>9} "
            f"{'COMMON_RNG':>11} {'RES_EFF_RNG':>12} {'RAW_MED_RNG':>12} "
            f"{'RES_MED_RNG':>12} {'STATO':>30}"
        )
        print("-" * 114)
        for row in proxy_invariance.itertuples(index=False):
            print(
                f"{row.block_id:5d} {row.block_sessions:8d} "
                f"{row.observed_scenarios:5d} "
                f"{_display_number(row.effective_asset_count_range, 5):>9} "
                f"{_display_number(row.common_factor_share_range, 5):>11} "
                f"{_display_number(row.residual_effective_asset_count_range, 5):>12} "
                f"{_display_number(row.median_pair_absolute_correlation_range, 5):>12} "
                f"{_display_number(row.residual_median_pair_absolute_correlation_range, 5):>12} "
                f"{row.invariance_state:>30}"
            )
        print(
            "Il confronto non nomina un rappresentante e non alimenta "
            "componenti operative."
        )

        phase_summary = cross_asset_dependence.phase_factor_summary.loc[
            cross_asset_dependence.phase_factor_summary["scope"].eq(
                ordinary_scope
            )
        ]
        print(f"\nSTRUTTURA CROSS-ASSET PER FASE — {ordinary_scope}")
        print(
            "Ogni variazione resta interamente dentro OPEN, MID_SESSION o "
            "CLOSE; i passaggi tra fasi sono esclusi."
        )
        print(
            f"{'FASE':12} {'OBS':>6} {'BLOCK':>5} {'COMMON':>8} "
            f"{'RAW_EFF':>8} {'RES_EFF':>8} {'RAW_MED':>8} "
            f"{'RES_MED':>8} {'COS_MIN':>8} {'STATO':>22}"
        )
        print("-" * 111)
        for row in phase_summary.itertuples(index=False):
            print(
                f"{row.session_phase:12} {row.observations:6d} "
                f"{row.eligible_blocks:5d} "
                f"{_display_number(row.common_factor_share):>8} "
                f"{_display_number(row.raw_effective_asset_count):>8} "
                f"{_display_number(row.residual_effective_asset_count):>8} "
                f"{_display_number(row.raw_median_pair_absolute_correlation):>8} "
                f"{_display_number(row.residual_median_pair_absolute_correlation):>8} "
                f"{_display_number(row.minimum_adjacent_loading_cosine):>8} "
                f"{row.common_mode_state:>22}"
            )
        phase_contrast = cross_asset_dependence.phase_factor_contrast.loc[
            cross_asset_dependence.phase_factor_contrast["scope"].eq(
                ordinary_scope
            )
        ].iloc[0]
        print("\nCONTRASTO TRA FASI")
        print(
            f"Range componente comune: "
            f"{_display_number(phase_contrast.common_factor_share_range)}; "
            f"range ampiezza raw: "
            f"{_display_number(phase_contrast.raw_effective_asset_count_range)}; "
            f"range ampiezza residua: "
            f"{_display_number(phase_contrast.residual_effective_asset_count_range)}."
        )
        print(
            f"Componente comune massima: "
            f"{phase_contrast.maximum_common_factor_session_phase}; "
            f"ampiezza raw massima: "
            f"{phase_contrast.maximum_raw_effective_asset_session_phase}; "
            f"ampiezza residua massima: "
            f"{phase_contrast.maximum_residual_effective_asset_session_phase}."
        )
        print(f"Stato descrittivo: {phase_contrast.phase_structure_state}")
        print(
            "Le differenze tra fasi descrivono eterogeneita' contemporanea, "
            "non accuratezza o rendimento."
        )

        phase_residual = cross_asset_dependence.phase_residual_contrast.loc[
            cross_asset_dependence.phase_residual_contrast["scope"].eq(
                ordinary_scope
            )
        ].copy()
        phase_residual = phase_residual.sort_values(
            [
                "absolute_residual_correlation_range",
                "maximum_absolute_residual_correlation",
                "ticker_a",
                "ticker_b",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        phase_residual_states = (
            phase_residual["phase_residual_state"].value_counts().to_dict()
        )
        print(f"\nDIPENDENZA RESIDUA PER FASE — {ordinary_scope}")
        print(
            "La componente comune viene ricalcolata separatamente in ogni "
            "fase e in ogni blocco completo."
        )
        print(
            "Stati: "
            f"LOW={int(phase_residual_states.get('LOW_PHASE_RESIDUAL_VARIATION', 0))}, "
            f"MODERATE={int(phase_residual_states.get('MODERATE_PHASE_RESIDUAL_VARIATION', 0))}, "
            f"HIGH={int(phase_residual_states.get('HIGH_PHASE_RESIDUAL_VARIATION', 0))}, "
            f"SIGN_REVERSAL={int(phase_residual_states.get('PHASE_SIGN_REVERSAL', 0))}, "
            f"PROXY_PHASE={int(phase_residual_states.get('PHASE_SPECIFIC_PROXY_LINK', 0))}, "
            f"PROXY_STABLE={int(phase_residual_states.get('STABLE_PROXY_ACROSS_PHASES', 0))}."
        )
        print(
            f"{'COPPIA':19} {'PHASE':>5} {'RES_MIN':>8} {'RES_MAX':>8} "
            f"{'ABS_RNG':>8} {'MAX_PHASE':>11} {'PROXY':>5} "
            f"{'STATO':>34}"
        )
        print("-" * 112)
        for row in phase_residual.head(15).itertuples(index=False):
            print(
                f"{row.ticker_a + '/' + row.ticker_b:19} "
                f"{row.phases_available:5d} "
                f"{_display_number(row.minimum_residual_correlation):>8} "
                f"{_display_number(row.maximum_residual_correlation):>8} "
                f"{_display_number(row.absolute_residual_correlation_range):>8} "
                f"{row.maximum_absolute_residual_session_phase:>11} "
                f"{row.stable_proxy_phases:5d} "
                f"{row.phase_residual_state:>34}"
            )
        print(
            "Queste differenze sono contemporanee e descrittive; non sono "
            "relazioni direzionali o prospettiche."
        )

        if arguments.cross_asset_dependence_output:
            destination = write_cross_asset_dependence_pairs(
                cross_asset_dependence.pairs,
                arguments.cross_asset_dependence_output,
            )
            print(f"\nCSV DIPENDENZA CROSS-ASSET: {destination}")
        if arguments.cross_asset_block_breadth_output:
            destination = write_cross_asset_block_breadth(
                cross_asset_dependence.block_breadth,
                arguments.cross_asset_block_breadth_output,
            )
            print(f"CSV AMPIEZZA PER BLOCCHI: {destination}")
        if arguments.cross_asset_breadth_output:
            destination = write_cross_asset_effective_breadth(
                cross_asset_dependence.breadth,
                arguments.cross_asset_breadth_output,
            )
            print(f"CSV AMPIEZZA EFFETTIVA: {destination}")
        if arguments.cross_asset_factor_loadings_output:
            destination = write_cross_asset_factor_loadings(
                cross_asset_dependence.factor_loadings,
                arguments.cross_asset_factor_loadings_output,
            )
            print(f"CSV CONTRIBUTI FATTORE COMUNE: {destination}")
        if arguments.cross_asset_factor_blocks_output:
            destination = write_cross_asset_factor_blocks(
                cross_asset_dependence.factor_blocks,
                arguments.cross_asset_factor_blocks_output,
            )
            print(f"CSV FATTORE COMUNE PER BLOCCHI: {destination}")
        if arguments.cross_asset_factor_summary_output:
            destination = write_cross_asset_factor_summary(
                cross_asset_dependence.factor_summary,
                arguments.cross_asset_factor_summary_output,
            )
            print(f"CSV RIEPILOGO FATTORE COMUNE: {destination}")
        if arguments.cross_asset_residual_blocks_output:
            destination = write_cross_asset_residual_pair_blocks(
                cross_asset_dependence.residual_pair_blocks,
                arguments.cross_asset_residual_blocks_output,
            )
            print(f"CSV DIPENDENZA RESIDUA PER BLOCCHI: {destination}")
        if arguments.cross_asset_residual_pairs_output:
            destination = write_cross_asset_residual_pairs(
                cross_asset_dependence.residual_pairs,
                arguments.cross_asset_residual_pairs_output,
            )
            print(f"CSV RIEPILOGO DIPENDENZA RESIDUA: {destination}")
        if arguments.cross_asset_proxy_clusters_output:
            destination = write_cross_asset_proxy_clusters(
                cross_asset_dependence.proxy_clusters,
                arguments.cross_asset_proxy_clusters_output,
            )
            print(f"CSV GRUPPI PROXY STABILI: {destination}")
        if arguments.cross_asset_proxy_sensitivity_output:
            destination = write_cross_asset_proxy_sensitivity(
                cross_asset_dependence.proxy_sensitivity,
                arguments.cross_asset_proxy_sensitivity_output,
            )
            print(f"CSV SENSIBILITA' AI PROXY: {destination}")
        if arguments.cross_asset_proxy_blocks_output:
            destination = write_cross_asset_proxy_block_sensitivity(
                cross_asset_dependence.proxy_block_sensitivity,
                arguments.cross_asset_proxy_blocks_output,
            )
            print(f"CSV SENSIBILITA' PROXY PER BLOCCHI: {destination}")
        if arguments.cross_asset_proxy_block_stability_output:
            destination = write_cross_asset_proxy_block_stability(
                cross_asset_dependence.proxy_block_stability,
                arguments.cross_asset_proxy_block_stability_output,
            )
            print(f"CSV STABILITA' SCENARI PROXY: {destination}")
        if arguments.cross_asset_proxy_invariance_output:
            destination = write_cross_asset_proxy_representative_invariance(
                cross_asset_dependence.proxy_representative_invariance,
                arguments.cross_asset_proxy_invariance_output,
            )
            print(f"CSV INVARIANZA RAPPRESENTANTI PROXY: {destination}")
        if arguments.cross_asset_phase_factor_loadings_output:
            destination = write_cross_asset_phase_factor_loadings(
                cross_asset_dependence.phase_factor_loadings,
                arguments.cross_asset_phase_factor_loadings_output,
            )
            print(f"CSV CONTRIBUTI FATTORE PER FASE: {destination}")
        if arguments.cross_asset_phase_factor_blocks_output:
            destination = write_cross_asset_phase_factor_blocks(
                cross_asset_dependence.phase_factor_blocks,
                arguments.cross_asset_phase_factor_blocks_output,
            )
            print(f"CSV FATTORE PER FASE E BLOCCHI: {destination}")
        if arguments.cross_asset_phase_factor_summary_output:
            destination = write_cross_asset_phase_factor_summary(
                cross_asset_dependence.phase_factor_summary,
                arguments.cross_asset_phase_factor_summary_output,
            )
            print(f"CSV RIEPILOGO FATTORE PER FASE: {destination}")
        if arguments.cross_asset_phase_factor_contrast_output:
            destination = write_cross_asset_phase_factor_contrast(
                cross_asset_dependence.phase_factor_contrast,
                arguments.cross_asset_phase_factor_contrast_output,
            )
            print(f"CSV CONTRASTO CROSS-ASSET TRA FASI: {destination}")
        if arguments.cross_asset_phase_residual_blocks_output:
            destination = write_cross_asset_phase_residual_pair_blocks(
                cross_asset_dependence.phase_residual_pair_blocks,
                arguments.cross_asset_phase_residual_blocks_output,
            )
            print(f"CSV RESIDUI PER FASE E BLOCCHI: {destination}")
        if arguments.cross_asset_phase_residual_pairs_output:
            destination = write_cross_asset_phase_residual_pairs(
                cross_asset_dependence.phase_residual_pairs,
                arguments.cross_asset_phase_residual_pairs_output,
            )
            print(f"CSV RESIDUI PER COPPIA E FASE: {destination}")
        if arguments.cross_asset_phase_residual_contrast_output:
            destination = write_cross_asset_phase_residual_contrast(
                cross_asset_dependence.phase_residual_contrast,
                arguments.cross_asset_phase_residual_contrast_output,
            )
            print(f"CSV CONTRASTO RESIDUI TRA FASI: {destination}")
    elif (
        arguments.cross_asset_dependence_output
        or arguments.cross_asset_block_breadth_output
        or arguments.cross_asset_breadth_output
        or arguments.cross_asset_factor_loadings_output
        or arguments.cross_asset_factor_blocks_output
        or arguments.cross_asset_factor_summary_output
        or arguments.cross_asset_residual_blocks_output
        or arguments.cross_asset_residual_pairs_output
        or arguments.cross_asset_proxy_clusters_output
        or arguments.cross_asset_proxy_sensitivity_output
        or arguments.cross_asset_proxy_blocks_output
        or arguments.cross_asset_proxy_block_stability_output
        or arguments.cross_asset_proxy_invariance_output
        or arguments.cross_asset_phase_factor_loadings_output
        or arguments.cross_asset_phase_factor_blocks_output
        or arguments.cross_asset_phase_factor_summary_output
        or arguments.cross_asset_phase_factor_contrast_output
        or arguments.cross_asset_phase_residual_blocks_output
        or arguments.cross_asset_phase_residual_pairs_output
        or arguments.cross_asset_phase_residual_contrast_output
    ):
        raise ValueError(
            "Servono almeno due asset completi per i CSV cross-asset."
        )

    if feature_frames:
        regime_atlas = build_intraday_regime_atlas(
            feature_frames,
            source_audits,
            regime_atlas_config,
        )
        regime_summary = summarize_regime_atlas(regime_atlas)
        print("\nATLANTE DESCRITTIVO DEI CONTESTI INTRADAY")
        print(
            "Stato = fase seduta x CHOP x Squeeze; CMF entra soltanto come "
            "disponibilita' della sorgente."
        )
        print(
            f"{'TICKER':9} {'SOURCE':18} {'ROWS':>6} {'CELLS':>6} "
            f"{'ADEQ':>5} {'SPARSE':>6} {'INSUF':>5} "
            f"{'TR_TYPES':>8} {'TR_OBS':>7}"
        )
        print("-" * 91)
        for row in regime_summary.itertuples(index=False):
            print(
                f"{row.ticker:9} {row.source_status:18} "
                f"{row.assignments:6d} {row.context_cells:6d} "
                f"{row.adequate:5d} {row.sparse:6d} "
                f"{row.insufficient:5d} {row.transition_types:8d} "
                f"{row.transition_observations:7d}"
            )

        print("\nCONTESTI PRESENTI SU PIU' ASSET")
        cross_asset = regime_atlas.cross_asset.sort_values(
            ["assets_observed", "assets_adequate", "total_observations"],
            ascending=False,
            kind="mergesort",
        ).head(12)
        if cross_asset.empty:
            print("- Nessun contesto valido osservato.")
        else:
            print(
                f"{'PHASE':12} {'CHOP':10} {'SQUEEZE':12} "
                f"{'ASSET':>5} {'ADEQ':>5} {'OBS':>8} {'MED_SHARE':>10}"
            )
            print("-" * 75)
            for row in cross_asset.itertuples(index=False):
                print(
                    f"{row.session_phase:12} {row.chop_segment:10} "
                    f"{row.squeeze_state:12} {row.assets_observed:5d} "
                    f"{row.assets_adequate:5d} {row.total_observations:8d} "
                    f"{row.median_observation_share_phase:10.1%}"
                )
        print(
            "Le presenze multi-asset sono descrittive: gli asset correlati "
            "non sono prove indipendenti."
        )
        print(
            "Le transizioni includono solo barre consecutive della stessa "
            "seduta; notte e bucket mancanti interrompono la catena."
        )

        if arguments.regime_atlas_output:
            destination = write_regime_atlas(
                regime_atlas.atlas,
                arguments.regime_atlas_output,
            )
            print(f"\nCSV ATLANTE: {destination}")
        if arguments.regime_transition_output:
            destination = write_regime_transitions(
                regime_atlas.transitions,
                arguments.regime_transition_output,
            )
            print(f"CSV TRANSIZIONI: {destination}")

        context_stability = analyze_intraday_context_stability(
            regime_atlas,
            context_stability_config,
        )
        context_summary = summarize_context_stability(context_stability)
        print("\nSTABILITA' DEI CONTESTI TRA BLOCCHI STORICI")
        print(
            "TVD e Jensen-Shannon confrontano frequenze osservate; nessun "
            "outcome futuro viene utilizzato."
        )
        print(
            "Un drift e' elevato soltanto se supera anche il quantile nullo "
            "da permutazioni deterministiche."
        )
        print(
            f"{'TICKER':9} {'SOURCE':18} {'LOW':>4} {'ISOL':>5} "
            f"{'P_ELEV':>6} {'P_HIGH':>6} {'INSUF':>5}"
        )
        print("-" * 66)
        for row in context_summary.itertuples(index=False):
            print(
                f"{row.ticker:9} {row.source_status:18} "
                f"{row.low_or_none:4d} {row.isolated:5d} "
                f"{row.persistent_elevated:6d} "
                f"{row.persistent_high:6d} {row.insufficient:5d}"
            )

        persistent = context_stability.persistence.loc[
            context_stability.persistence["pattern"].isin(
                {"PERSISTENT_ELEVATED", "PERSISTENT_HIGH"}
            )
        ]
        print("\nCAMBIAMENTI PERSISTENTI DI CONTESTO")
        if persistent.empty:
            print("- Nessun cambiamento persistente rilevato.")
        else:
            print(
                f"{'TICKER':9} {'COMPONENT':12} {'SCOPE':12} "
                f"{'PATTERN':21} {'LATEST':14} {'MAX_TVD':>8} {'MAX_JS':>8}"
            )
            print("-" * 94)
            for row in persistent.itertuples(index=False):
                print(
                    f"{row.ticker:9} {row.component:12} {row.scope:12} "
                    f"{row.pattern:21} {row.latest_shift:14} "
                    f"{row.max_total_variation:8.3f} "
                    f"{row.max_jensen_shannon_bits:8.3f}"
                )
        print(
            "I pattern descrivono drift di distribuzione, non vantaggio "
            "predittivo o idoneita' operativa."
        )

        if arguments.context_stability_output:
            destination = write_context_stability(
                context_stability.persistence,
                arguments.context_stability_output,
            )
            print(f"\nCSV STABILITA' CONTESTI: {destination}")
    elif (
        arguments.regime_atlas_output
        or arguments.regime_transition_output
        or arguments.context_stability_output
    ):
        raise ValueError("Nessuna feature disponibile per l'atlante descrittivo.")

    all_errors = {**source_errors, **analysis_errors}
    if all_errors:
        print("\nERRORI ISOLATI")
        for ticker, error in sorted(all_errors.items()):
            print(f"- {ticker}: {error}")

    if arguments.output_dir:
        print("\nAUDIT ARTEFATTI DESCRITTIVI")
        if _standard_outputs_share_directory(arguments):
            audit = audit_intraday_report_directory(
                arguments.output_dir,
                expected_reports=STANDARD_INTRADAY_REPORT_FILENAMES,
            )
            manifest = write_intraday_report_manifest(
                audit,
                Path(arguments.output_dir) / REPORT_MANIFEST_FILENAME,
            )
            print(
                "Report attesi/osservati: "
                f"{audit.observed_expected_reports}/{audit.expected_reports}; "
                f"CSV inattesi: {audit.unexpected_reports}; "
                f"righe: {audit.total_rows}."
            )
            print(f"Esito audit: {audit.overall_state}")
            print(f"Manifest audit: {manifest}")
            if audit.overall_state != "PASS":
                failed = audit.details.loc[
                    audit.details["audit_state"].ne("PASS")
                ]
                problems = ", ".join(
                    f"{row.report_name}={row.audit_state}"
                    for row in failed.itertuples(index=False)
                )
                raise ValueError(f"Audit report non superato: {problems}")
        else:
            print(
                "- Audit automatico saltato: almeno un percorso esplicito "
                "e' esterno a --output-dir. Usa run_intraday_report_audit "
                "sulla directory completa."
            )

    print("\nESITO")
    ready = sum(
        audit.status is LocalSourceReadiness.READY_DESCRIPTIVE
        for audit in source_audits.values()
    )
    price_only_ready = sum(
        audit.status is LocalSourceReadiness.PRICE_ONLY
        for audit in source_audits.values()
    )
    print(
        f"- {ready}/{len(source_audits)} sorgenti sono READY_DESCRIPTIVE."
    )
    print(
        f"- {price_only_ready}/{len(source_audits)} sorgenti PRICE_ONLY sono "
        "incluse senza CMF."
    )
    print(
        "- Il report migliora la base statistica descrittiva; non dimostra "
        "redditivita' e non approva una strategia."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
