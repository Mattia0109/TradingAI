from dataclasses import replace

from market.specifications import (
    MarketSpecificationRegistry
)


RESEARCH_UNIVERSE = {
    # Indici azionari
    "SPY": "EQUITY",
    "QQQ": "EQUITY",
    "IWM": "EQUITY",
    "EFA": "EQUITY",
    "EEM": "EQUITY",

    # Obbligazioni
    "SHY": "BOND",
    "IEF": "BOND",
    "TLT": "BOND",

    # Materie prime tramite ETF
    "GLD": "COMMODITY",
    "SLV": "COMMODITY",
    "USO": "COMMODITY",
    "DBA": "COMMODITY",

    # Valute tramite ETF
    "UUP": "FX",
    "FXE": "FX",
    "FXY": "FX",

    # Crypto spot
    "BTC-USD": "CRYPTO",
    "ETH-USD": "CRYPTO"
}


ETF_RESEARCH_TICKERS = {
    ticker
    for ticker in RESEARCH_UNIVERSE
    if not ticker.endswith("-USD")
}


def register_research_universe_specifications():
    """
    Registra gli ETF della ricerca con:

    - quantità intera;
    - moltiplicatore 1;
    - costi ETF;
    - margine pari al nozionale.

    BTC ed ETH sono già riconosciuti
    automaticamente come CRYPTO.
    """

    for ticker in ETF_RESEARCH_TICKERS:
        specification = replace(
            MarketSpecificationRegistry.ETF_TEMPLATE,
            code=ticker
        )

        MarketSpecificationRegistry.register(
            ticker=ticker,
            specification=specification
        )


def get_research_universe(
    selected_tickers=None
):
    """
    Restituisce l'universo completo oppure
    un sottoinsieme validato.
    """

    if selected_tickers is None:
        return dict(
            RESEARCH_UNIVERSE
        )

    result = {}

    for ticker in selected_tickers:
        normalized = (
            str(ticker)
            .upper()
            .strip()
        )

        if normalized not in RESEARCH_UNIVERSE:
            raise ValueError(
                f"Ticker non presente nell'universo: "
                f"{normalized}"
            )

        result[normalized] = (
            RESEARCH_UNIVERSE[
                normalized
            ]
        )

    if not result:
        raise ValueError(
            "L'universo selezionato è vuoto."
        )

    return result