"""Backtester event-driven del Long Burst Momentum Challenger.

Il simulatore è deliberatamente non operativo: applica esposizione unitaria
long-only, non calcola size monetarie, non usa leva e non invia ordini.  Una
decisione alla chiusura ``t`` può diventare esposizione soltanto all'apertura
della barra successiva.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from adaptive.long_burst import LongBurstSignalEngine
from adaptive.models import Direction


@dataclass(frozen=True)
class LongBurstBacktestConfig:
    initial_equity: float = 100_000.0
    minimum_holding_bars: int = 1
    maximum_holding_bars: int = 5
    signal_retention_ratio: float = 0.55
    cooldown_bars: int = 1
    round_trip_cost_bps: float = 10.0
    annualization_factor: int = 252

    def __post_init__(self) -> None:
        if self.initial_equity <= 0.0:
            raise ValueError("initial_equity deve essere positivo.")
        if not 1 <= int(self.minimum_holding_bars) <= 5:
            raise ValueError("minimum_holding_bars deve essere compreso tra 1 e 5.")
        if not 1 <= int(self.maximum_holding_bars) <= 5:
            raise ValueError("maximum_holding_bars deve essere compreso tra 1 e 5.")
        if self.minimum_holding_bars > self.maximum_holding_bars:
            raise ValueError("La durata minima non può superare quella massima.")
        if not 0.0 <= self.signal_retention_ratio <= 1.0:
            raise ValueError("signal_retention_ratio deve essere compreso tra 0 e 1.")
        if int(self.cooldown_bars) < 0:
            raise ValueError("cooldown_bars non può essere negativo.")
        if self.round_trip_cost_bps < 0.0:
            raise ValueError("round_trip_cost_bps non può essere negativo.")
        if int(self.annualization_factor) <= 1:
            raise ValueError("annualization_factor deve essere maggiore di 1.")


@dataclass(frozen=True)
class LongBurstBacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    benchmark_daily_returns: pd.Series
    gross_exposure: pd.Series
    asset_returns: pd.DataFrame
    asset_exposure: pd.DataFrame
    trades: pd.DataFrame
    signal_observations: pd.DataFrame
    regime_diagnostics: pd.DataFrame
    errors: Mapping[str, str]
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    benchmark_total_return: float
    benchmark_cagr: float
    benchmark_sharpe: float
    benchmark_max_drawdown: float
    signal_count: int
    trade_count: int
    win_rate: float
    expectancy: float
    profit_factor: float
    average_duration_bars: float
    maximum_duration_bars: int
    invested_fraction: float
    average_gross_exposure: float
    paper_only: bool = True


@dataclass(frozen=True)
class _AssetSimulation:
    returns: pd.Series
    exposure: pd.Series
    benchmark_returns: pd.Series
    trades: pd.DataFrame
    observations: pd.DataFrame


def _performance_metrics(
    returns: pd.Series,
    annualization_factor: int,
) -> tuple[float, float, float, float, float]:
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    growth = (1.0 + clean).cumprod()
    total_return = float(growth.iloc[-1] - 1.0)
    years = len(clean) / annualization_factor
    cagr = (
        float(growth.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0.0 and growth.iloc[-1] > 0.0
        else -1.0
    )
    deviation = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    annualized_volatility = deviation * math.sqrt(annualization_factor)
    sharpe = (
        float(clean.mean()) / deviation * math.sqrt(annualization_factor)
        if deviation > 0.0
        else 0.0
    )
    running_peak = growth.cummax().clip(lower=1.0)
    max_drawdown = float((1.0 - growth / running_peak).max())
    return total_return, cagr, annualized_volatility, sharpe, max_drawdown


class LongBurstBacktester:
    """Simula esposizioni brevi su uno o più asset, sempre long-only."""

    paper_only = True

    def __init__(
        self,
        engine: LongBurstSignalEngine | None = None,
        config: LongBurstBacktestConfig | None = None,
    ) -> None:
        self.engine = engine or LongBurstSignalEngine()
        self.config = config or LongBurstBacktestConfig(
            round_trip_cost_bps=self.engine.config.round_trip_cost_bps
        )
        if self.engine.config.forecast_horizon_bars > self.config.maximum_holding_bars:
            raise ValueError(
                "L'orizzonte del segnale non può superare il time-stop del backtest."
            )

    def _signal_observations(
        self,
        ticker: str,
        frame: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> pd.DataFrame:
        horizon = self.engine.config.forecast_horizon_bars
        rows: list[dict[str, object]] = []
        close = frame["close"].to_numpy(dtype=float)
        for position, (_, signal) in enumerate(signals.iterrows()):
            if signal["direction"] != Direction.LONG.value:
                continue
            target = position + horizon
            if target >= len(frame):
                continue
            realized = float(close[target] / close[position] - 1.0)
            rows.append(
                {
                    "ticker": ticker,
                    "decision_date": frame.index[position],
                    "outcome_date": frame.index[target],
                    "horizon_bars": horizon,
                    "regime": signal["regime"],
                    "confidence": float(signal["confidence"]),
                    "expected_return": float(signal["expected_return"]),
                    "realized_return": realized,
                    "realized_net_return": (
                        (1.0 + realized)
                        * (1.0 - self.config.round_trip_cost_bps / 10_000.0)
                        - 1.0
                    ),
                    "lorentzian_score": float(signal["lorentzian_score"]),
                    "squeeze_score": float(signal["squeeze_score"]),
                    "cmf_score": float(signal["cmf_score"]),
                    "raw_score": float(signal["raw_score"]),
                    "opportunity_score": float(signal["opportunity_score"]),
                }
            )
        return pd.DataFrame(rows)

    def _simulate_asset(self, ticker: str, data: pd.DataFrame) -> _AssetSimulation:
        frame, _ = self.engine._prepare(data)
        signals = self.engine.evaluate_history(frame, ticker)
        count = len(frame)
        strategy_returns = np.zeros(count, dtype=float)
        exposure = np.zeros(count, dtype=float)
        trades: list[dict[str, object]] = []
        pending: dict[str, object] | None = None
        position: dict[str, object] | None = None
        cooldown_until = 0
        side_cost = self.config.round_trip_cost_bps / 20_000.0

        for offset in range(1, count):
            entered = False
            if (
                pending is not None
                and int(pending["entry_offset"]) == offset
                and position is None
            ):
                entry_price = float(frame.iloc[offset]["open"])
                position = {
                    **pending,
                    "entry_date": frame.index[offset],
                    "entry_price": entry_price,
                    "highest_price": float(frame.iloc[offset]["high"]),
                    "lowest_price": float(frame.iloc[offset]["low"]),
                }
                pending = None
                entered = True

            if position is not None:
                exposure[offset] = 1.0
                current = frame.iloc[offset]
                prior_close = float(frame.iloc[offset - 1]["close"])
                price_ratio = (
                    float(current["close"]) / float(current["open"])
                    if entered
                    else float(current["close"]) / prior_close
                )
                daily_factor = price_ratio * ((1.0 - side_cost) if entered else 1.0)
                position["highest_price"] = max(
                    float(position["highest_price"]),
                    float(current["high"]),
                )
                position["lowest_price"] = min(
                    float(position["lowest_price"]),
                    float(current["low"]),
                )
                duration = offset - int(position["entry_offset"]) + 1
                current_signal = signals.iloc[offset]
                exit_reason: str | None = None
                if duration >= self.config.minimum_holding_bars:
                    if duration >= self.config.maximum_holding_bars:
                        exit_reason = "TIME_STOP"
                    elif current_signal["direction"] != Direction.LONG.value:
                        exit_reason = "SIGNAL_DECAY"
                    elif float(current_signal["confidence"]) < (
                        float(position["entry_confidence"])
                        * self.config.signal_retention_ratio
                    ):
                        exit_reason = "SIGNAL_DECAY"
                if offset == count - 1 and exit_reason is None:
                    exit_reason = "END_OF_DATA"

                if exit_reason is not None:
                    daily_factor *= 1.0 - side_cost
                    exit_price = float(current["close"])
                    entry_price = float(position["entry_price"])
                    gross_return = exit_price / entry_price - 1.0
                    net_return = (
                        exit_price
                        / entry_price
                        * (1.0 - side_cost) ** 2
                        - 1.0
                    )
                    trades.append(
                        {
                            "ticker": ticker,
                            "signal_date": position["signal_date"],
                            "entry_date": position["entry_date"],
                            "exit_date": frame.index[offset],
                            "duration_bars": duration,
                            "exit_reason": exit_reason,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "gross_return": gross_return,
                            "net_return": net_return,
                            "mfe": float(position["highest_price"]) / entry_price - 1.0,
                            "mae": float(position["lowest_price"]) / entry_price - 1.0,
                            "entry_confidence": float(position["entry_confidence"]),
                            "entry_raw_score": float(position["entry_raw_score"]),
                            "entry_opportunity_score": float(
                                position["entry_opportunity_score"]
                            ),
                            "entry_regime": position["entry_regime"],
                        }
                    )
                    position = None
                    cooldown_until = offset + self.config.cooldown_bars
                strategy_returns[offset] = daily_factor - 1.0

            if (
                position is None
                and pending is None
                and offset < count - 1
                and offset >= cooldown_until
            ):
                signal = signals.iloc[offset]
                if signal["direction"] == Direction.LONG.value:
                    pending = {
                        "signal_date": frame.index[offset],
                        "entry_offset": offset + 1,
                        "entry_confidence": float(signal["confidence"]),
                        "entry_raw_score": float(signal["raw_score"]),
                        "entry_opportunity_score": float(
                            signal["opportunity_score"]
                        ),
                        "entry_regime": signal["regime"],
                    }

        warmup = min(self.engine.config.minimum_history - 1, count - 1)
        return _AssetSimulation(
            returns=pd.Series(
                strategy_returns,
                index=frame.index,
                name=ticker,
                dtype=float,
            ).iloc[warmup:],
            exposure=pd.Series(
                exposure,
                index=frame.index,
                name=ticker,
                dtype=float,
            ).iloc[warmup:],
            benchmark_returns=frame["close"]
            .pct_change()
            .fillna(0.0)
            .rename(ticker)
            .iloc[warmup:],
            trades=pd.DataFrame(trades),
            observations=self._signal_observations(ticker, frame, signals),
        )

    @staticmethod
    def _regime_diagnostics(observations: pd.DataFrame) -> pd.DataFrame:
        columns = (
            "signals",
            "hit_rate",
            "mean_return",
            "mean_net_return",
            "mean_confidence",
            "forecast_correlation",
        )
        if observations.empty:
            return pd.DataFrame(columns=columns).rename_axis("regime")

        rows: list[dict[str, object]] = []
        for regime, group in observations.groupby("regime", sort=True):
            correlation = 0.0
            if len(group) > 1:
                candidate = float(
                    group["expected_return"].corr(group["realized_return"])
                )
                correlation = candidate if math.isfinite(candidate) else 0.0
            rows.append(
                {
                    "regime": regime,
                    "signals": len(group),
                    "hit_rate": float((group["realized_net_return"] > 0.0).mean()),
                    "mean_return": float(group["realized_return"].mean()),
                    "mean_net_return": float(group["realized_net_return"].mean()),
                    "mean_confidence": float(group["confidence"].mean()),
                    "forecast_correlation": correlation,
                }
            )
        return pd.DataFrame(rows).set_index("regime")

    def run(self, markets: Mapping[str, pd.DataFrame]) -> LongBurstBacktestResult:
        if not markets:
            raise ValueError("Serve almeno un mercato.")
        simulations: dict[str, _AssetSimulation] = {}
        errors: dict[str, str] = {}
        for raw_ticker, data in markets.items():
            ticker = str(raw_ticker).upper().strip()
            if not ticker:
                errors[str(raw_ticker)] = "Ticker vuoto."
                continue
            try:
                simulations[ticker] = self._simulate_asset(ticker, data)
            except Exception as exc:
                errors[ticker] = f"{type(exc).__name__}: {exc}"
        if not simulations:
            details = "; ".join(
                f"{ticker}: {error}" for ticker, error in sorted(errors.items())
            )
            raise ValueError("Nessun mercato simulabile. " + details)

        common_start = max(
            item.returns.index[0] for item in simulations.values()
        )
        asset_returns = pd.concat(
            [item.returns for item in simulations.values()],
            axis=1,
        ).sort_index().loc[common_start:].fillna(0.0)
        asset_exposure = pd.concat(
            [item.exposure for item in simulations.values()],
            axis=1,
        ).fillna(0.0).reindex(asset_returns.index).fillna(0.0)
        benchmark_assets = pd.concat(
            [item.benchmark_returns for item in simulations.values()],
            axis=1,
        ).fillna(0.0).reindex(asset_returns.index).fillna(0.0)
        daily_returns = asset_returns.mean(axis=1).rename("long_burst_return")
        benchmark_returns = benchmark_assets.mean(axis=1).rename(
            "equal_weight_return"
        )
        gross_exposure = asset_exposure.mean(axis=1).rename("gross_exposure")
        equity_curve = (
            self.config.initial_equity
            * (1.0 + daily_returns).cumprod()
        ).rename("equity")
        trades = pd.concat(
            [item.trades for item in simulations.values() if not item.trades.empty],
            ignore_index=True,
        ) if any(not item.trades.empty for item in simulations.values()) else pd.DataFrame()
        observations = pd.concat(
            [
                item.observations
                for item in simulations.values()
                if not item.observations.empty
            ],
            ignore_index=True,
        ) if any(not item.observations.empty for item in simulations.values()) else pd.DataFrame()

        strategy_metrics = _performance_metrics(
            daily_returns,
            self.config.annualization_factor,
        )
        benchmark_metrics = _performance_metrics(
            benchmark_returns,
            self.config.annualization_factor,
        )
        if trades.empty:
            win_rate = expectancy = average_duration = 0.0
            maximum_duration = 0
            profit_factor = 0.0
        else:
            wins = trades.loc[trades["net_return"] > 0.0, "net_return"]
            losses = trades.loc[trades["net_return"] < 0.0, "net_return"]
            win_rate = float((trades["net_return"] > 0.0).mean())
            expectancy = float(trades["net_return"].mean())
            profit_factor = (
                float(wins.sum() / abs(losses.sum()))
                if not losses.empty and abs(float(losses.sum())) > 0.0
                else (float("inf") if not wins.empty else 0.0)
            )
            average_duration = float(trades["duration_bars"].mean())
            maximum_duration = int(trades["duration_bars"].max())

        return LongBurstBacktestResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            benchmark_daily_returns=benchmark_returns,
            gross_exposure=gross_exposure,
            asset_returns=asset_returns,
            asset_exposure=asset_exposure,
            trades=trades,
            signal_observations=observations,
            regime_diagnostics=self._regime_diagnostics(observations),
            errors=errors,
            total_return=strategy_metrics[0],
            cagr=strategy_metrics[1],
            annualized_volatility=strategy_metrics[2],
            sharpe=strategy_metrics[3],
            max_drawdown=strategy_metrics[4],
            benchmark_total_return=benchmark_metrics[0],
            benchmark_cagr=benchmark_metrics[1],
            benchmark_sharpe=benchmark_metrics[3],
            benchmark_max_drawdown=benchmark_metrics[4],
            signal_count=len(observations),
            trade_count=len(trades),
            win_rate=win_rate,
            expectancy=expectancy,
            profit_factor=profit_factor,
            average_duration_bars=average_duration,
            maximum_duration_bars=maximum_duration,
            invested_fraction=float((gross_exposure > 0.0).mean()),
            average_gross_exposure=float(gross_exposure.mean()),
        )
