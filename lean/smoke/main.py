from AlgorithmImports import *


class LeanSmokeTestAlgorithm(QCAlgorithm):
    """Minimal local-only backtest that proves the LEAN runtime is usable."""

    def initialize(self) -> None:
        self.set_start_date(2013, 10, 7)
        self.set_end_date(2013, 10, 11)
        self.set_cash(100_000)

        self._spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        self._order_submitted = False

    def on_data(self, data: Slice) -> None:
        if self._order_submitted:
            return

        if data.bars.contains_key(self._spy):
            self.set_holdings(self._spy, 0.10)
            self._order_submitted = True
