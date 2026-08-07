"""Audit trail append-only per decisioni, previsioni e outcome osservati."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from adaptive.models import AdaptiveCycleResult


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class SQLiteDecisionJournal:
    """SQLite locale; nessun record viene usato per il training senza outcome."""

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._memory_connection: sqlite3.Connection | None = None
        if self.database_path == ":memory:":
            self._memory_connection = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = self._memory_connection or sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _close_if_needed(self, connection: sqlite3.Connection) -> None:
        if connection is not self._memory_connection:
            connection.close()

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            if self.database_path != ":memory:":
                connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS adaptive_cycles (
                    cycle_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    status TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    paper_only INTEGER NOT NULL CHECK (paper_only = 1),
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS adaptive_outcomes (
                    cycle_id TEXT PRIMARY KEY,
                    closed_at TEXT NOT NULL,
                    realized_asset_return REAL NOT NULL,
                    transaction_cost_return REAL NOT NULL,
                    slippage_return REAL NOT NULL,
                    model_error REAL,
                    FOREIGN KEY (cycle_id)
                        REFERENCES adaptive_cycles(cycle_id)
                );
                """
            )
            connection.commit()
            self._close_if_needed(connection)

    def record_cycle(self, result: AdaptiveCycleResult) -> None:
        payload = json.dumps(
            _jsonable(result),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT INTO adaptive_cycles (
                        cycle_id,
                        created_at,
                        ticker,
                        status,
                        regime,
                        paper_only,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        result.cycle_id,
                        datetime.now(timezone.utc).isoformat(),
                        result.snapshot.ticker,
                        result.status.value,
                        result.regime.primary.value,
                        payload,
                    ),
                )
                connection.commit()
            finally:
                self._close_if_needed(connection)

    def record_outcome(
        self,
        cycle_id: str,
        *,
        realized_asset_return: float,
        transaction_cost_return: float = 0.0,
        slippage_return: float = 0.0,
        model_error: float | None = None,
        closed_at: datetime | None = None,
    ) -> None:
        numeric_values = (
            float(realized_asset_return),
            float(transaction_cost_return),
            float(slippage_return),
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Outcome, costi e slippage devono essere finiti.")
        if numeric_values[1] < 0.0 or numeric_values[2] < 0.0:
            raise ValueError("Costi e slippage non possono essere negativi.")
        if model_error is not None and not math.isfinite(float(model_error)):
            raise ValueError("model_error deve essere finito.")
        with self._lock:
            connection = self._connect()
            try:
                exists = connection.execute(
                    "SELECT 1 FROM adaptive_cycles WHERE cycle_id = ?",
                    (cycle_id,),
                ).fetchone()
                if exists is None:
                    raise ValueError(f"cycle_id sconosciuto: {cycle_id}")
                connection.execute(
                    """
                    INSERT INTO adaptive_outcomes (
                        cycle_id,
                        closed_at,
                        realized_asset_return,
                        transaction_cost_return,
                        slippage_return,
                        model_error
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cycle_id,
                        (closed_at or datetime.now(timezone.utc)).isoformat(),
                        numeric_values[0],
                        numeric_values[1],
                        numeric_values[2],
                        None if model_error is None else float(model_error),
                    ),
                )
                connection.commit()
            finally:
                self._close_if_needed(connection)

    def fetch_training_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT
                        cycles.payload_json,
                        outcomes.closed_at,
                        outcomes.realized_asset_return,
                        outcomes.transaction_cost_return,
                        outcomes.slippage_return,
                        outcomes.model_error
                    FROM adaptive_cycles AS cycles
                    INNER JOIN adaptive_outcomes AS outcomes
                        ON outcomes.cycle_id = cycles.cycle_id
                    ORDER BY outcomes.closed_at, cycles.cycle_id
                    """
                ).fetchall()
            finally:
                self._close_if_needed(connection)

        return [
            {
                "cycle": json.loads(row[0]),
                "closed_at": row[1],
                "realized_asset_return": row[2],
                "transaction_cost_return": row[3],
                "slippage_return": row[4],
                "model_error": row[5],
            }
            for row in rows
        ]
