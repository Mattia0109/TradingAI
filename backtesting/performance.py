import math
import statistics


class PerformanceAnalyzer:
    """
    Calcola le metriche di rendimento e rischio
    di una strategia di trading.
    """

    def calculate(
        self,
        initial_capital,
        final_capital,
        trades,
        equity_curve
    ):
        if initial_capital <= 0:
            raise ValueError(
                "Il capitale iniziale deve essere maggiore di zero."
            )

        net_profit = (
            final_capital -
            initial_capital
        )

        total_return = (
            net_profit /
            initial_capital *
            100
        )

        winning_trades = [
            trade
            for trade in trades
            if trade["pnl"] > 0
        ]

        losing_trades = [
            trade
            for trade in trades
            if trade["pnl"] < 0
        ]

        breakeven_trades = [
            trade
            for trade in trades
            if trade["pnl"] == 0
        ]

        trade_count = len(trades)

        win_rate = (
            len(winning_trades) /
            trade_count *
            100
            if trade_count > 0
            else 0.0
        )

        loss_rate = (
            len(losing_trades) /
            trade_count *
            100
            if trade_count > 0
            else 0.0
        )

        gross_profit = sum(
            trade["pnl"]
            for trade in winning_trades
        )

        gross_loss = abs(
            sum(
                trade["pnl"]
                for trade in losing_trades
            )
        )

        profit_factor = self.calculate_profit_factor(
            gross_profit=gross_profit,
            gross_loss=gross_loss
        )

        average_trade = (
            net_profit /
            trade_count
            if trade_count > 0
            else 0.0
        )

        average_win = (
            gross_profit /
            len(winning_trades)
            if winning_trades
            else 0.0
        )

        average_loss = (
            gross_loss /
            len(losing_trades)
            if losing_trades
            else 0.0
        )

        payoff_ratio = (
            average_win /
            average_loss
            if average_loss > 0
            else (
                float("inf")
                if average_win > 0
                else 0.0
            )
        )

        win_probability = (
            len(winning_trades) /
            trade_count
            if trade_count > 0
            else 0.0
        )

        loss_probability = (
            len(losing_trades) /
            trade_count
            if trade_count > 0
            else 0.0
        )

        expectancy = (
            win_probability *
            average_win
            -
            loss_probability *
            average_loss
        )

        trade_returns = [
            float(
                trade.get(
                    "return_percent",
                    0.0
                )
            )
            for trade in trades
        ]

        median_trade_return = (
            statistics.median(
                trade_returns
            )
            if trade_returns
            else 0.0
        )

        max_drawdown = self.calculate_max_drawdown(
            equity_curve
        )

        recovery_factor = (
            net_profit /
            (
                initial_capital *
                max_drawdown /
                100
            )
            if max_drawdown > 0
            else (
                float("inf")
                if net_profit > 0
                else 0.0
            )
        )

        max_consecutive_losses = (
            self.calculate_max_consecutive_losses(
                trades
            )
        )

        max_consecutive_wins = (
            self.calculate_max_consecutive_wins(
                trades
            )
        )

        exposure_percent = (
            self.calculate_exposure_percent(
                trades=trades,
                equity_curve=equity_curve
            )
        )

        return {
            "initial_capital": float(
                initial_capital
            ),
            "final_capital": float(
                final_capital
            ),
            "net_profit": float(
                net_profit
            ),
            "total_return_percent": float(
                total_return
            ),

            "total_trades": trade_count,
            "winning_trades": len(
                winning_trades
            ),
            "losing_trades": len(
                losing_trades
            ),
            "breakeven_trades": len(
                breakeven_trades
            ),

            "win_rate_percent": float(
                win_rate
            ),
            "loss_rate_percent": float(
                loss_rate
            ),

            "gross_profit": float(
                gross_profit
            ),
            "gross_loss": float(
                gross_loss
            ),
            "profit_factor": float(
                profit_factor
            ),

            "average_trade": float(
                average_trade
            ),
            "average_win": float(
                average_win
            ),
            "average_loss": float(
                average_loss
            ),
            "payoff_ratio": float(
                payoff_ratio
            ),
            "expectancy": float(
                expectancy
            ),

            "median_trade_return_percent": float(
                median_trade_return
            ),

            "max_drawdown_percent": float(
                max_drawdown
            ),
            "recovery_factor": float(
                recovery_factor
            ),

            "max_consecutive_losses": (
                max_consecutive_losses
            ),
            "max_consecutive_wins": (
                max_consecutive_wins
            ),

            "exposure_percent": float(
                exposure_percent
            ),

            "statistical_warning": (
                trade_count < 30
            )
        }


    @staticmethod
    def calculate_profit_factor(
        gross_profit,
        gross_loss
    ):
        if gross_loss > 0:
            return (
                gross_profit /
                gross_loss
            )

        if gross_profit > 0:
            return float("inf")

        return 0.0


    @staticmethod
    def calculate_max_drawdown(
        equity_curve
    ):
        if not equity_curve:
            return 0.0

        first_equity = float(
            equity_curve[0]["equity"]
        )

        if first_equity <= 0:
            return 0.0

        peak = first_equity
        max_drawdown = 0.0

        for point in equity_curve:
            equity = float(
                point["equity"]
            )

            if equity > peak:
                peak = equity

            if peak <= 0:
                continue

            drawdown = (
                peak -
                equity
            ) / peak * 100

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

        return max_drawdown


    @staticmethod
    def calculate_max_consecutive_losses(
        trades
    ):
        maximum = 0
        current = 0

        for trade in trades:
            if trade["pnl"] < 0:
                current += 1

                maximum = max(
                    maximum,
                    current
                )

            else:
                current = 0

        return maximum


    @staticmethod
    def calculate_max_consecutive_wins(
        trades
    ):
        maximum = 0
        current = 0

        for trade in trades:
            if trade["pnl"] > 0:
                current += 1

                maximum = max(
                    maximum,
                    current
                )

            else:
                current = 0

        return maximum


    @staticmethod
    def calculate_exposure_percent(
        trades,
        equity_curve
    ):
        """
        Stima la percentuale del periodo durante
        la quale il sistema aveva una posizione aperta.
        """

        if not trades or not equity_curve:
            return 0.0

        timestamps = [
            point.get("timestamp")
            for point in equity_curve
            if point.get("timestamp") is not None
        ]

        if len(timestamps) < 2:
            return 0.0

        start_time = min(
            timestamps
        )

        end_time = max(
            timestamps
        )

        total_duration = (
            end_time -
            start_time
        ).total_seconds()

        if total_duration <= 0:
            return 0.0

        exposed_seconds = 0.0

        for trade in trades:
            open_time = trade.get(
                "open_time"
            )

            close_time = trade.get(
                "close_time"
            )

            if (
                open_time is None
                or close_time is None
            ):
                continue

            duration = (
                close_time -
                open_time
            ).total_seconds()

            exposed_seconds += max(
                0.0,
                duration
            )

        exposure = (
            exposed_seconds /
            total_duration *
            100
        )

        return min(
            exposure,
            100.0
        )