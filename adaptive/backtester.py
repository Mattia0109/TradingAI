"""Backtester causale della pipeline adaptive, esclusivamente paper-only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from adaptive.journal import SQLiteDecisionJournal
from adaptive.diagnostics import summarize_forecasts
from adaptive.models import PortfolioState
from adaptive.orchestrator import AdaptiveTradingSystem


@dataclass(frozen=True)
class AdaptiveBacktestConfig:
    initial_equity: float = 100_000.0
    rebalance_every: int = 1
    cost_bps_per_turnover: float = 5.0
    annualization_factor: int = 252
    flatten_rejected_signals: bool = True

    def __post_init__(self) -> None:
        if self.initial_equity <= 0.0:
            raise ValueError("initial_equity deve essere positivo.")
        if int(self.rebalance_every) <= 0:
            raise ValueError("rebalance_every deve essere positivo.")
        if self.cost_bps_per_turnover < 0.0:
            raise ValueError("cost_bps_per_turnover non può essere negativo.")
        if int(self.annualization_factor) <= 1:
            raise ValueError("annualization_factor deve essere maggiore di 1.")


@dataclass(frozen=True)
class AdaptiveBacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    statuses: pd.DataFrame
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    total_turnover: float
    total_cost_return: float
    decision_count: int
    analysis_error_count: int
    average_gross_exposure: float
    average_net_exposure: float
    invested_fraction: float
    equal_weight_total_return: float
    equal_weight_cagr: float
    equal_weight_sharpe: float
    equal_weight_max_drawdown: float
    forecast_observations: pd.DataFrame
    strategy_diagnostics: pd.DataFrame
    strategy_regime_diagnostics: pd.DataFrame
    paper_only: bool = True


class AdaptiveBacktester:
    """Decide alla chiusura t e applica il rendimento soltanto da t a t+1."""

    def __init__(self, system: AdaptiveTradingSystem | None = None, config: AdaptiveBacktestConfig | None = None) -> None:
        self.system = system or AdaptiveTradingSystem(journal=SQLiteDecisionJournal(":memory:"))
        self.config = config or AdaptiveBacktestConfig()
        if not self.system.paper_only:
            raise ValueError("Il backtester accetta soltanto sistemi paper-only.")

    @staticmethod
    def _prepare(markets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        if not markets:
            raise ValueError("Serve almeno un mercato.")
        prepared: dict[str, pd.DataFrame] = {}
        for raw_ticker, frame in markets.items():
            ticker = str(raw_ticker).upper().strip()
            if not ticker or not isinstance(frame, pd.DataFrame):
                continue
            clean = frame.copy()
            clean.columns = [str(column).lower().strip() for column in clean.columns]
            if "close" not in clean.columns:
                raise ValueError(f"{ticker}: colonna close mancante.")
            if "date" in clean.columns:
                clean["date"] = pd.to_datetime(clean["date"], errors="coerce", utc=True)
                clean = clean.dropna(subset=["date"]).set_index("date", drop=False)
                clean.index.name = None
            elif isinstance(clean.index, pd.DatetimeIndex):
                clean.index = pd.to_datetime(clean.index, utc=True)
            clean = clean.sort_index()
            clean = clean[~clean.index.duplicated(keep="last")]
            clean["close"] = pd.to_numeric(clean["close"], errors="coerce")
            clean = clean.dropna(subset=["close"])
            if not clean.empty:
                prepared[ticker] = clean
        if not prepared:
            raise ValueError("Nessuna serie valida.")
        return prepared

    @staticmethod
    def _portfolio(equity: float, peak: float, weights: Mapping[str, float], asset_classes: Mapping[str, str]) -> PortfolioState:
        class_exposure: dict[str, float] = {}
        for ticker, weight in weights.items():
            key = str(asset_classes.get(ticker, "UNKNOWN")).upper().strip()
            class_exposure[key] = class_exposure.get(key, 0.0) + abs(weight)
        return PortfolioState(
            equity=equity, peak_equity=peak,
            gross_exposure=sum(abs(value) for value in weights.values()),
            net_exposure=sum(weights.values()), current_weights=dict(weights),
            asset_class_exposure=class_exposure,
        )

    def run(self, markets: Mapping[str, pd.DataFrame], *, asset_classes: Mapping[str, str] | None = None) -> AdaptiveBacktestResult:
        data = self._prepare(markets)
        classes = {str(k).upper(): str(v).upper() for k, v in (asset_classes or {}).items()}
        common_index = None
        for frame in data.values():
            common_index = frame.index if common_index is None else common_index.intersection(frame.index)
        dates = common_index.sort_values() if common_index is not None else pd.Index([])
        minimum = self.system.feature_engine.config.minimum_history
        if len(dates) <= minimum:
            raise ValueError(f"Storico comune insufficiente: servono più di {minimum} barre.")

        equity = peak = self.config.initial_equity
        weights = {ticker: 0.0 for ticker in data}
        output_dates: list[object] = []
        returns_list: list[float] = []
        equity_list: list[float] = []
        turnovers: list[float] = []
        costs: list[float] = []
        weight_rows: list[dict[str, float]] = []
        status_rows: list[dict[str, str]] = []
        decision_count = 0
        analysis_error_count = 0
        forecast_records: list[dict[str, object]] = []

        for offset in range(minimum - 1, len(dates) - 1):
            date, next_date = dates[offset], dates[offset + 1]
            turnover = 0.0
            status_row: dict[str, str] = {}
            if (offset - minimum + 1) % self.config.rebalance_every == 0:
                history = {ticker: frame.loc[:date] for ticker, frame in data.items()}
                portfolio = self._portfolio(equity, peak, weights, classes)
                analysis = self.system.analyze_universe(history, portfolio, asset_classes=classes)
                analysis_error_count += len(analysis.errors)
                if not analysis.cycles:
                    details = "; ".join(
                        f"{ticker}: {error}"
                        for ticker, error in sorted(analysis.errors.items())
                    )
                    raise RuntimeError(
                        "Nessun asset analizzabile durante il backtest. " + details
                    )
                status_row = {ticker: cycle.status.value for ticker, cycle in analysis.cycles.items()}
                for ticker, cycle in analysis.cycles.items():
                    estimated_cost = cycle.execution_decision.estimated_cost_bps
                    for forecast in cycle.forecasts:
                        forecast_records.append(
                            {
                                "decision_offset": offset,
                                "date": date,
                                "ticker": ticker,
                                "strategy_id": forecast.strategy_id,
                                "regime": cycle.regime.primary.value,
                                "direction": forecast.direction.value,
                                "direction_sign": forecast.direction.sign,
                                "confidence": forecast.confidence,
                                "expected_return": forecast.expected_return,
                                "horizon_bars": forecast.horizon_bars,
                                "estimated_cost_bps": estimated_cost,
                            }
                        )
                    forecast_records.append(
                        {
                            "decision_offset": offset,
                            "date": date,
                            "ticker": ticker,
                            "strategy_id": "META_MODEL",
                            "regime": cycle.regime.primary.value,
                            "direction": cycle.meta_decision.direction.value,
                            "direction_sign": cycle.meta_decision.direction.sign,
                            "confidence": cycle.meta_decision.confidence,
                            "expected_return": cycle.meta_decision.expected_return,
                            "horizon_bars": cycle.meta_decision.horizon_bars,
                            "estimated_cost_bps": estimated_cost,
                        }
                    )
                targets = dict(weights)
                if self.config.flatten_rejected_signals:
                    for ticker, cycle in analysis.cycles.items():
                        if not cycle.execution_decision.approved:
                            targets[ticker] = 0.0
                targets.update(analysis.allocation.target_weights)
                turnover = sum(abs(targets[ticker] - weights.get(ticker, 0.0)) for ticker in targets)
                weights = targets
                decision_count += len(analysis.cycles)

            cost_return = turnover * self.config.cost_bps_per_turnover / 10_000.0
            asset_returns = {ticker: float(frame.loc[next_date, "close"] / frame.loc[date, "close"] - 1.0) for ticker, frame in data.items()}
            net_return = sum(weights.get(ticker, 0.0) * value for ticker, value in asset_returns.items()) - cost_return
            equity *= 1.0 + net_return
            peak = max(peak, equity)
            output_dates.append(next_date)
            returns_list.append(net_return)
            equity_list.append(equity)
            turnovers.append(turnover)
            costs.append(cost_return)
            weight_rows.append(dict(weights))
            status_rows.append(status_row)
            denominator = 1.0 + net_return
            if denominator <= 0.0:
                raise RuntimeError("Il portafoglio ha perso almeno il 100% in una barra.")
            weights = {
                ticker: weight * (1.0 + asset_returns[ticker]) / denominator
                for ticker, weight in weights.items()
            }

        returns = pd.Series(returns_list, index=output_dates, name="return", dtype=float)
        curve = pd.Series(equity_list, index=output_dates, name="equity", dtype=float)
        years = len(returns) / self.config.annualization_factor
        total_return = equity / self.config.initial_equity - 1.0
        cagr = (equity / self.config.initial_equity) ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else -1.0
        std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        volatility = std * math.sqrt(self.config.annualization_factor)
        sharpe = float(returns.mean()) / std * math.sqrt(self.config.annualization_factor) if std > 0 else 0.0
        running_peak = curve.cummax().clip(lower=self.config.initial_equity)
        max_drawdown = float((1.0 - curve / running_peak).max()) if not curve.empty else 0.0
        weight_frame = pd.DataFrame(weight_rows, index=output_dates).fillna(0.0)
        gross_series = weight_frame.abs().sum(axis=1)
        net_series = weight_frame.sum(axis=1)
        equal_weight_returns = pd.Series(
            [
                np.mean(
                    [
                        float(frame.loc[next_date, "close"] / frame.loc[date, "close"] - 1.0)
                        for frame in data.values()
                    ]
                )
                for date, next_date in zip(dates[minimum - 1 : -1], dates[minimum:])
            ],
            index=output_dates,
            dtype=float,
        )
        equal_growth = (1.0 + equal_weight_returns).cumprod()
        equal_total = float(equal_growth.iloc[-1] - 1.0)
        equal_cagr = float(equal_growth.iloc[-1] ** (1.0 / years) - 1.0)
        equal_std = float(equal_weight_returns.std(ddof=1))
        equal_sharpe = (
            float(equal_weight_returns.mean()) / equal_std
            * math.sqrt(self.config.annualization_factor)
            if equal_std > 0.0
            else 0.0
        )
        equal_drawdown = float((1.0 - equal_growth / equal_growth.cummax()).max())
        matured_records: list[dict[str, object]] = []
        for record in forecast_records:
            target_offset = int(record["decision_offset"]) + int(record["horizon_bars"])
            enriched = dict(record)
            if target_offset < len(dates):
                ticker = str(record["ticker"])
                start_date = dates[int(record["decision_offset"])]
                target_date = dates[target_offset]
                enriched["realized_return"] = float(
                    data[ticker].loc[target_date, "close"]
                    / data[ticker].loc[start_date, "close"]
                    - 1.0
                )
                enriched["outcome_date"] = target_date
            else:
                enriched["realized_return"] = np.nan
                enriched["outcome_date"] = pd.NaT
            matured_records.append(enriched)
        observations, strategy_diagnostics, regime_diagnostics = (
            summarize_forecasts(matured_records)
        )
        return AdaptiveBacktestResult(
            equity_curve=curve, daily_returns=returns,
            weights=weight_frame,
            turnover=pd.Series(turnovers, index=output_dates, name="turnover"),
            costs=pd.Series(costs, index=output_dates, name="cost_return"),
            statuses=pd.DataFrame(status_rows, index=output_dates),
            total_return=total_return, cagr=cagr, annualized_volatility=volatility,
            sharpe=sharpe, max_drawdown=max_drawdown,
            total_turnover=float(np.sum(turnovers)), total_cost_return=float(np.sum(costs)),
            decision_count=decision_count,
            analysis_error_count=analysis_error_count,
            average_gross_exposure=float(gross_series.mean()),
            average_net_exposure=float(net_series.mean()),
            invested_fraction=float((gross_series > 1e-12).mean()),
            equal_weight_total_return=equal_total,
            equal_weight_cagr=equal_cagr,
            equal_weight_sharpe=equal_sharpe,
            equal_weight_max_drawdown=equal_drawdown,
            forecast_observations=observations,
            strategy_diagnostics=strategy_diagnostics,
            strategy_regime_diagnostics=regime_diagnostics,
        )
