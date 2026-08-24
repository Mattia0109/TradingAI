"""Fondazione paper-only del sistema TradingAI adattivo."""

from adaptive.backtester import (
    AdaptiveBacktestConfig,
    AdaptiveBacktestResult,
    AdaptiveBacktester,
)
from adaptive.learning import (
    CausalRegimePerformanceTracker,
    RegimeLearningConfig,
    RegimePerformanceState,
)
from adaptive.long_burst import (
    LongBurstConfig,
    LongBurstRegime,
    LongBurstSignal,
    LongBurstSignalEngine,
)
from adaptive.long_burst_backtester import (
    LongBurstBacktestConfig,
    LongBurstBacktestResult,
    LongBurstBacktester,
)
from adaptive.validation import PromotionGateResult, evaluate_challenger

from adaptive.models import (
    AdaptiveCycleResult,
    CycleStatus,
    Direction,
    ExecutionDecision,
    ExecutionEstimate,
    FeatureSnapshot,
    MarketRegime,
    MetaDecision,
    PortfolioState,
    RegimeAssessment,
    RiskDecision,
    StrategyForecast,
)
from adaptive.news_intelligence import (
    ClassifiedNewsEvent,
    NewsAssessment,
    NewsEventType,
    StructuredNewsIntelligence,
)
from adaptive.orchestrator import (
    AdaptiveSystemConfig,
    AdaptiveTradingSystem,
    UniverseAnalysisResult,
)
from adaptive.portfolio_allocator import (
    AdaptivePortfolioAllocator,
    PortfolioAllocation,
    PortfolioAllocatorConfig,
)

__all__ = [
    "AdaptiveBacktestConfig",
    "AdaptiveBacktestResult",
    "AdaptiveBacktester",
    "CausalRegimePerformanceTracker",
    "AdaptiveCycleResult",
    "AdaptiveSystemConfig",
    "AdaptiveTradingSystem",
    "AdaptivePortfolioAllocator",
    "CycleStatus",
    "ClassifiedNewsEvent",
    "Direction",
    "ExecutionDecision",
    "ExecutionEstimate",
    "FeatureSnapshot",
    "LongBurstBacktestConfig",
    "LongBurstBacktestResult",
    "LongBurstBacktester",
    "LongBurstConfig",
    "LongBurstRegime",
    "LongBurstSignal",
    "LongBurstSignalEngine",
    "MarketRegime",
    "MetaDecision",
    "NewsAssessment",
    "NewsEventType",
    "PortfolioState",
    "PromotionGateResult",
    "PortfolioAllocation",
    "PortfolioAllocatorConfig",
    "RegimeAssessment",
    "RegimeLearningConfig",
    "RegimePerformanceState",
    "RiskDecision",
    "StrategyForecast",
    "StructuredNewsIntelligence",
    "UniverseAnalysisResult",
    "evaluate_challenger",
]
