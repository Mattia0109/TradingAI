import pandas as pd
import numpy as np


class IndicatorEngine:
    """
    Calcola indicatori tecnici
    senza dipendenze esterne.
    """

    def ema(self, series, period):

        return series.ewm(
            span=period,
            adjust=False
        ).mean()



    def rsi(self, series, period=14):

        delta = series.diff()

        gain = delta.clip(lower=0)

        loss = -delta.clip(upper=0)


        avg_gain = gain.rolling(period).mean()

        avg_loss = loss.rolling(period).mean()


        rs = avg_gain / avg_loss


        return 100 - (
            100 / (1 + rs)
        )



    def macd(self, series):

        ema12 = self.ema(series, 12)

        ema26 = self.ema(series, 26)


        return ema12 - ema26



    def atr(self, df, period=14):

        high_low = df["High"] - df["Low"]

        high_close = abs(
            df["High"] - df["Close"].shift()
        )

        low_close = abs(
            df["Low"] - df["Close"].shift()
        )


        ranges = pd.concat(
            [
                high_low,
                high_close,
                low_close
            ],
            axis=1
        )


        true_range = ranges.max(axis=1)


        return true_range.rolling(period).mean()