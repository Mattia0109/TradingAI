from datetime import datetime


class DataNormalizer:
    """
    Converte dati grezzi di mercato
    in formato standard.
    """

    def extract_value(self, value):

        if hasattr(value, "iloc"):

            return float(value.iloc[0])

        return float(value)



    def normalize_candle(
        self,
        ticker,
        candle
    ):

        return {

            "ticker": ticker,

            "timestamp":
                candle.name
                if hasattr(candle, "name")
                else datetime.now(),

            "open":
                self.extract_value(candle["Open"]),

            "high":
                self.extract_value(candle["High"]),

            "low":
                self.extract_value(candle["Low"]),

            "close":
                self.extract_value(candle["Close"]),

            "volume":
                self.extract_value(candle["Volume"])

        }