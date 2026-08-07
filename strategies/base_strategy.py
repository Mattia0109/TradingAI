from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """
    Interfaccia comune per tutte le strategie TradingAI.

    Ogni strategia deve restituire una decisione
    compatibile con SignalGenerator.
    """

    name = "base_strategy"

    @abstractmethod
    def generate_decision(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> dict:
        """
        Restituisce:

        {
            "ticker": str,
            "strategy": str,
            "action": "BUY" | "SELL" | "WAIT",
            "confidence": float,
            "atr": float | None,
            "reasons": list[str]
        }
        """

        raise NotImplementedError


    @staticmethod
    def validate_ohlcv_data(
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Verifica e normalizza un DataFrame OHLCV.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "data deve essere un pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Il DataFrame storico è vuoto."
            )

        normalized = data.copy()

        normalized.columns = [
            str(column).lower().strip()
            for column in normalized.columns
        ]

        required_columns = {
            "open",
            "high",
            "low",
            "close",
            "volume"
        }

        missing_columns = (
            required_columns -
            set(normalized.columns)
        )

        if missing_columns:
            raise ValueError(
                "Colonne mancanti: "
                f"{sorted(missing_columns)}"
            )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for column in numeric_columns:
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="coerce"
            )

        normalized = normalized.dropna(
            subset=numeric_columns
        )

        normalized = normalized.reset_index(
            drop=True
        )

        if normalized.empty:
            raise ValueError(
                "Nessuna candela valida disponibile."
            )

        return normalized


    @staticmethod
    def wait_decision(
        ticker: str,
        strategy: str,
        reasons: list[str],
        atr: float | None = None
    ) -> dict:
        """
        Crea una decisione neutrale standard.
        """

        return {
            "ticker": ticker,
            "strategy": strategy,
            "action": "WAIT",
            "confidence": 0.0,
            "atr": atr,
            "reasons": reasons
        }