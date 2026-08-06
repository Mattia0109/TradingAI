class OpportunityRanker:

    def calculate_score(self, result):

        signal = result["signal"]

        if isinstance(signal, dict):

            confidence = signal["confidence"]
            rr = signal["risk_reward"]
            ticker = signal["ticker"]

        else:

            confidence = signal.confidence
            rr = signal.risk_reward
            ticker = signal.ticker

        score = confidence
        score += rr * 10

        return score

    def rank(self, results):

        ranked = []

        for result in results:

            if not result["approved"]:
                continue

            signal = result["signal"]

            if isinstance(signal, dict):
                ticker = signal["ticker"]
            else:
                ticker = signal.ticker

            score = self.calculate_score(result)

            ranked.append(
                {
                    "ticker": ticker,
                    "score": score,
                    "signal": signal
                }
            )

        ranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked