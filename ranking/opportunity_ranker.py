class OpportunityRanker:
    """
    Classifica le opportunità di trading.
    """

    def __init__(self):
        pass


    def calculate_score(self, result):

        signal = result["signal"]

        score = 0


        # Qualità del segnale

        score += signal.confidence


        # Rapporto rischio rendimento

        score += signal.risk_reward * 10


        # Bonus se il trade è approvato

        if result["approved"]:

            score += 10


        # Penalità se non ha motivazioni positive

        if len(signal.reasons) > 0:

            score -= 5


        return round(score, 2)



    def rank(self, results):

        ranked = []


        for result in results:


            if result["approved"]:


                score = self.calculate_score(
                    result
                )


                ranked.append({

                    "ticker": result["signal"].ticker,

                    "score": score,

                    "signal": result["signal"]

                })


        ranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )


        return ranked