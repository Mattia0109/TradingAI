from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class MarketSnapshot:
    """
    Fotografia del mercato in un preciso momento.
    Contiene dati prezzo + indicatori.
    """

    ticker: str
    timeframe: str
    timestamp: datetime

    price: float
    volume: float

    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None

    sma_20: Optional[float] = None
    ema_12: Optional[float] = None

    volatility: Optional[float] = None

    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None

    market_regime: Optional[str] = None



@dataclass
class TradeSignal:
    """
    Decisione generata dal Decision Engine.
    """

    ticker: str
    timestamp: datetime

    direction: str
    # LONG / SHORT / HOLD

    confidence: float
    # 0-100

    entry_price: float

    stop_loss: float
    take_profit: float

    risk_reward: float

    timeframe: str

    strategy: str

    reasons: List[str] = field(default_factory=list)

    approved: bool = False



@dataclass
class Position:
    """
    Posizione aperta sul mercato.
    """

    ticker: str

    direction: str

    entry_price: float

    quantity: float

    leverage: float

    stop_loss: float

    take_profit: float

    open_time: datetime

    broker_order_id: Optional[str] = None



@dataclass
class TradeResult:
    """
    Risultato finale di un trade.
    """

    ticker: str

    entry_price: float

    exit_price: float

    profit_loss: float

    profit_percentage: float

    duration_minutes: float

    strategy: str

    timestamp: datetime