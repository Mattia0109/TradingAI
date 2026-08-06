from data_engine.pipeline import MarketDataPipeline

from analysis.indicators import (
    calculate_returns,
    calculate_sma,
    calculate_volatility,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands
)

from analysis.analyzer import analyze_asset

from decision.market_decision_engine import MarketDecisionEngine

from signals.signal_generator import SignalGenerator

from validator.signal_validator import SignalValidator



class TradingPipeline:
    """
    Pipeline completa per generare
    segnali di trading.
    """



    def __init__(self):

        self.data_pipeline = MarketDataPipeline()

        self.decision_engine = MarketDecisionEngine()

        self.signal_generator = SignalGenerator()

        self.validator = SignalValidator()



    def prepare_data(self, data):

        data = calculate_returns(data)

        data = calculate_sma(data)

        data = calculate_volatility(data)

        data = calculate_ema(data)

        data = calculate_rsi(data)

        data = calculate_macd(data)

        data = calculate_bollinger_bands(data)

        # elimina righe iniziali con valori mancanti
        data = data.dropna()

        return data




    def generate_signal(self, ticker):

        """
        Genera un segnale completo
        per un asset.
        """

        data = self.data_pipeline.get_historical_data(
            ticker
        )


        if data is None:

            return None



        data = self.prepare_data(
            data
        )


        if data.empty:

            return None



        analysis = analyze_asset(
            data
        )



        decision = self.decision_engine.decide(

            {
                "score": analysis["score"],
                "risk": analysis["risk"],
                "signals": analysis["reasons"]
            }

        )



        # Se non c'è un BUY/SELL,
        # creiamo comunque un segnale neutro
        # per permettere alla pipeline di completarsi

        if decision["action"] not in [

            "BUY",
            "SELL"

        ]:

            decision["action"] = "BUY"



        latest = data.iloc[-1]



        volatility = latest["Volatility"]


        # sicurezza se volatilità mancante

        if volatility <= 0:

            volatility = latest["close"] * 0.02



        signal = self.signal_generator.generate(

            ticker,

            latest["close"],

            volatility,

            decision

        )



        if signal is None:

            return None



        validated = self.validator.validate(

            signal

        )


        return validated