import pandas as pd


class ResearchCalendarAligner:
    """
    Allinea un universo multi-asset a un calendario
    di riferimento.

    Per la ricerca ETF + crypto viene normalmente
    utilizzato SPY come calendario principale.

    Questo garantisce che:

    - 21 barre rappresentino 21 sessioni comuni;
    - BTC ed ETH non introducano barre di weekend;
    - i ribilanciamenti non avvengano nei weekend;
    - l'annualizzazione a 252 sessioni sia coerente;
    - tutti gli asset utilizzino la stessa timebase.
    """

    def __init__(
        self,
        reference_ticker="SPY"
    ):
        normalized_reference = (
            str(reference_ticker)
            .upper()
            .strip()
        )

        if not normalized_reference:
            raise ValueError(
                "reference_ticker non può essere vuoto."
            )

        self.reference_ticker = (
            normalized_reference
        )


    @staticmethod
    def normalize_ticker(
        ticker
    ):
        normalized = (
            str(ticker)
            .upper()
            .strip()
        )

        if not normalized:
            raise ValueError(
                "ticker non può essere vuoto."
            )

        return normalized


    @staticmethod
    def prepare_frame(
        data,
        ticker
    ):
        if not isinstance(
            data,
            pd.DataFrame
        ):
            raise TypeError(
                f"I dati di {ticker} devono essere "
                "un pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                f"I dati di {ticker} sono vuoti."
            )

        normalized = data.copy()

        normalized.columns = [
            str(column)
            .lower()
            .strip()
            for column in normalized.columns
        ]

        if "date" not in normalized.columns:
            normalized["date"] = (
                normalized.index
            )

        normalized["date"] = pd.to_datetime(
            normalized["date"],
            errors="coerce",
            utc=True
        )

        normalized = normalized.dropna(
            subset=["date"]
        )

        normalized["date"] = (
            normalized["date"]
            .dt
            .tz_convert(None)
            .dt
            .normalize()
        )

        normalized = normalized.sort_values(
            "date"
        )

        normalized = normalized.drop_duplicates(
            subset=["date"],
            keep="last"
        )

        normalized = normalized.reset_index(
            drop=True
        )

        if normalized.empty:
            raise ValueError(
                f"Nessuna data valida per {ticker}."
            )

        return normalized


    def align(
        self,
        market_data
    ):
        if not isinstance(
            market_data,
            dict
        ):
            raise TypeError(
                "market_data deve essere un dizionario "
                "ticker -> DataFrame."
            )

        if not market_data:
            raise ValueError(
                "market_data non può essere vuoto."
            )

        prepared_data = {}

        for ticker, data in market_data.items():
            normalized_ticker = (
                self.normalize_ticker(
                    ticker
                )
            )

            prepared_data[
                normalized_ticker
            ] = self.prepare_frame(
                data=data,
                ticker=normalized_ticker
            )

        if (
            self.reference_ticker
            not in prepared_data
        ):
            raise ValueError(
                "Il ticker di riferimento "
                f"{self.reference_ticker} "
                "non è presente nei dati."
            )

        reference_data = prepared_data[
            self.reference_ticker
        ]

        reference_dates = set(
            reference_data[
                "date"
            ].tolist()
        )

        if not reference_dates:
            raise ValueError(
                "Il calendario di riferimento è vuoto."
            )

        aligned_data = {}
        asset_reports = {}

        total_rows_before = 0
        total_rows_after = 0

        for ticker, data in (
            prepared_data.items()
        ):
            rows_before = len(
                data
            )

            aligned = data[
                data["date"].isin(
                    reference_dates
                )
            ].copy()

            aligned = aligned.sort_values(
                "date"
            ).reset_index(
                drop=True
            )

            rows_after = len(
                aligned
            )

            removed_rows = (
                rows_before
                -
                rows_after
            )

            total_rows_before += (
                rows_before
            )

            total_rows_after += (
                rows_after
            )

            aligned_data[
                ticker
            ] = aligned

            weekend_rows_before = int(
                (
                    data["date"]
                    .dt
                    .dayofweek
                    >= 5
                ).sum()
            )

            weekend_rows_after = int(
                (
                    aligned["date"]
                    .dt
                    .dayofweek
                    >= 5
                ).sum()
            )

            asset_reports[
                ticker
            ] = {
                "rows_before": (
                    rows_before
                ),
                "rows_after": (
                    rows_after
                ),
                "removed_rows": (
                    removed_rows
                ),
                "weekend_rows_before": (
                    weekend_rows_before
                ),
                "weekend_rows_after": (
                    weekend_rows_after
                ),
                "first_session": (
                    aligned.iloc[0][
                        "date"
                    ]
                    if not aligned.empty
                    else None
                ),
                "last_session": (
                    aligned.iloc[-1][
                        "date"
                    ]
                    if not aligned.empty
                    else None
                )
            }

        report = {
            "reference_ticker": (
                self.reference_ticker
            ),
            "reference_sessions": len(
                reference_dates
            ),
            "asset_count": len(
                aligned_data
            ),
            "total_rows_before": (
                total_rows_before
            ),
            "total_rows_after": (
                total_rows_after
            ),
            "total_removed_rows": (
                total_rows_before
                -
                total_rows_after
            ),
            "assets": asset_reports
        }

        return {
            "market_data": aligned_data,
            "report": report
        }


def align_market_data_to_reference_calendar(
    market_data,
    reference_ticker="SPY"
):
    aligner = ResearchCalendarAligner(
        reference_ticker=reference_ticker
    )

    return aligner.align(
        market_data
    )