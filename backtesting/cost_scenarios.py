import copy

from backtesting.walk_forward import WalkForwardValidator


class CostScenarioAnalyzer:
    """
    Confronta una strategia walk-forward
    con differenti livelli di costi operativi.

    Lo scopo è distinguere:

    - assenza di edge lordo;
    - edge distrutto dai costi;
    - strategia promettente;
    - strategia resistente anche a costi stressati.
    """

    DEFAULT_COST_PROFILES = {
        "zero_costs": {
            "commission_percent": 0.0,
            "slippage_percent": 0.0
        },
        "realistic": {
            "commission_percent": 0.001,
            "slippage_percent": 0.0005
        },
        "stress": {
            "commission_percent": 0.002,
            "slippage_percent": 0.001
        }
    }

    REQUIRED_PROFILES = {
        "zero_costs",
        "realistic",
        "stress"
    }

    def __init__(
        self,
        train_bars=1500,
        test_bars=500,
        step_bars=500,
        initial_capital=10000.0,
        max_position_percent=0.10,
        minimum_history=30,
        max_holding_bars=78,
        cooldown_bars=12,
        max_trades_per_day=3,
        strategy=None,
        cost_profiles=None,
        validator_class=WalkForwardValidator
    ):
        if train_bars <= 0:
            raise ValueError(
                "train_bars deve essere maggiore di zero."
            )

        if test_bars <= 0:
            raise ValueError(
                "test_bars deve essere maggiore di zero."
            )

        if step_bars <= 0:
            raise ValueError(
                "step_bars deve essere maggiore di zero."
            )

        if initial_capital <= 0:
            raise ValueError(
                "initial_capital deve essere maggiore di zero."
            )

        if not 0 < max_position_percent <= 0.10:
            raise ValueError(
                "max_position_percent deve essere "
                "compreso tra 0 e 0.10."
            )

        if minimum_history < 26:
            raise ValueError(
                "minimum_history deve essere almeno 26."
            )

        if max_holding_bars <= 0:
            raise ValueError(
                "max_holding_bars deve essere maggiore di zero."
            )

        if cooldown_bars < 0:
            raise ValueError(
                "cooldown_bars non può essere negativo."
            )

        if max_trades_per_day <= 0:
            raise ValueError(
                "max_trades_per_day deve essere maggiore di zero."
            )

        if (
            strategy is not None
            and not hasattr(
                strategy,
                "generate_decision"
            )
        ):
            raise TypeError(
                "La strategia deve implementare "
                "generate_decision()."
            )

        self.train_bars = int(
            train_bars
        )

        self.test_bars = int(
            test_bars
        )

        self.step_bars = int(
            step_bars
        )

        self.initial_capital = float(
            initial_capital
        )

        self.max_position_percent = float(
            max_position_percent
        )

        self.minimum_history = int(
            minimum_history
        )

        self.max_holding_bars = int(
            max_holding_bars
        )

        self.cooldown_bars = int(
            cooldown_bars
        )

        self.max_trades_per_day = int(
            max_trades_per_day
        )

        self.strategy = strategy

        self.validator_class = (
            validator_class
        )

        self.cost_profiles = (
            copy.deepcopy(
                cost_profiles
            )
            if cost_profiles is not None
            else copy.deepcopy(
                self.DEFAULT_COST_PROFILES
            )
        )

        self.validate_cost_profiles(
            self.cost_profiles
        )


    @property
    def strategy_name(self):
        if self.strategy is None:
            return "legacy_momentum"

        return getattr(
            self.strategy,
            "name",
            self.strategy.__class__.__name__
        )


    @classmethod
    def validate_cost_profiles(
        cls,
        cost_profiles
    ):
        if not isinstance(
            cost_profiles,
            dict
        ):
            raise TypeError(
                "cost_profiles deve essere un dizionario."
            )

        missing_profiles = (
            cls.REQUIRED_PROFILES -
            set(cost_profiles)
        )

        if missing_profiles:
            raise ValueError(
                "Profili obbligatori mancanti: "
                f"{sorted(missing_profiles)}"
            )

        for profile_name, profile in (
            cost_profiles.items()
        ):
            if not isinstance(
                profile,
                dict
            ):
                raise TypeError(
                    f"Il profilo {profile_name} "
                    "deve essere un dizionario."
                )

            commission = profile.get(
                "commission_percent"
            )

            slippage = profile.get(
                "slippage_percent"
            )

            if commission is None:
                raise ValueError(
                    f"commission_percent mancante "
                    f"nel profilo {profile_name}."
                )

            if slippage is None:
                raise ValueError(
                    f"slippage_percent mancante "
                    f"nel profilo {profile_name}."
                )

            if commission < 0:
                raise ValueError(
                    f"Commissione negativa nel "
                    f"profilo {profile_name}."
                )

            if slippage < 0:
                raise ValueError(
                    f"Slippage negativo nel "
                    f"profilo {profile_name}."
                )


    @staticmethod
    def calculate_round_trip_cost_percent(
        commission_percent,
        slippage_percent
    ):
        """
        Calcola il costo teorico percentuale
        completo di ingresso e uscita.
        """

        one_way_cost = (
            float(commission_percent)
            +
            float(slippage_percent)
        )

        return (
            one_way_cost *
            2 *
            100
        )


    def build_validator(
        self,
        profile
    ):
        """
        Crea un validatore indipendente
        per uno specifico scenario di costi.
        """

        strategy_copy = (
            copy.deepcopy(
                self.strategy
            )
            if self.strategy is not None
            else None
        )

        return self.validator_class(
            train_bars=self.train_bars,
            test_bars=self.test_bars,
            step_bars=self.step_bars,
            initial_capital=(
                self.initial_capital
            ),
            max_position_percent=(
                self.max_position_percent
            ),
            minimum_history=(
                self.minimum_history
            ),
            commission_percent=float(
                profile[
                    "commission_percent"
                ]
            ),
            slippage_percent=float(
                profile[
                    "slippage_percent"
                ]
            ),
            max_holding_bars=(
                self.max_holding_bars
            ),
            cooldown_bars=(
                self.cooldown_bars
            ),
            max_trades_per_day=(
                self.max_trades_per_day
            ),
            strategy=strategy_copy
        )


    @staticmethod
    def get_scenario_summary(
        scenarios,
        scenario_name
    ):
        if scenario_name not in scenarios:
            raise ValueError(
                f"Scenario mancante: {scenario_name}"
            )

        return scenarios[
            scenario_name
        ]["summary"]


    @classmethod
    def build_comparison(
        cls,
        scenarios
    ):
        """
        Confronta i risultati dei tre scenari
        e assegna una classificazione.
        """

        zero_summary = (
            cls.get_scenario_summary(
                scenarios,
                "zero_costs"
            )
        )

        realistic_summary = (
            cls.get_scenario_summary(
                scenarios,
                "realistic"
            )
        )

        stress_summary = (
            cls.get_scenario_summary(
                scenarios,
                "stress"
            )
        )

        zero_profit = float(
            zero_summary[
                "total_net_profit"
            ]
        )

        realistic_profit = float(
            realistic_summary[
                "total_net_profit"
            ]
        )

        stress_profit = float(
            stress_summary[
                "total_net_profit"
            ]
        )

        realistic_cost_drag = (
            zero_profit -
            realistic_profit
        )

        stress_cost_drag = (
            zero_profit -
            stress_profit
        )

        if zero_profit > 0:
            realistic_profit_retention = (
                realistic_profit /
                zero_profit *
                100
            )

            stress_profit_retention = (
                stress_profit /
                zero_profit *
                100
            )

        else:
            realistic_profit_retention = 0.0
            stress_profit_retention = 0.0

        gross_edge_positive = (
            zero_profit > 0
        )

        realistic_edge_positive = (
            realistic_profit > 0
        )

        stress_edge_positive = (
            stress_profit > 0
        )

        realistic_robust = bool(
            realistic_summary.get(
                "is_robust",
                False
            )
        )

        stress_robust = bool(
            stress_summary.get(
                "is_robust",
                False
            )
        )

        if not gross_edge_positive:
            classification = (
                "REJECT_NO_GROSS_EDGE"
            )

            explanation = (
                "La strategia perde anche senza "
                "commissioni e slippage."
            )

        elif not realistic_edge_positive:
            classification = (
                "REJECT_COST_SENSITIVE"
            )

            explanation = (
                "Esiste un vantaggio lordo, ma viene "
                "completamente distrutto dai costi realistici."
            )

        elif not realistic_robust:
            classification = (
                "RESEARCH_NOT_ROBUST"
            )

            explanation = (
                "La strategia resta positiva con costi "
                "realistici, ma non supera i criteri "
                "di robustezza walk-forward."
            )

        elif not stress_edge_positive:
            classification = (
                "PAPER_TRADE_ONLY"
            )

            explanation = (
                "La strategia supera i costi realistici, "
                "ma non resiste allo scenario stress."
            )

        elif not stress_robust:
            classification = (
                "PROMISING_CANDIDATE"
            )

            explanation = (
                "La strategia rimane positiva nello "
                "scenario stress, ma la robustezza "
                "non è ancora sufficiente."
            )

        else:
            classification = (
                "ROBUST_CANDIDATE"
            )

            explanation = (
                "La strategia supera i criteri "
                "walk-forward anche con costi stressati."
            )

        accepted_for_research = (
            classification
            in {
                "RESEARCH_NOT_ROBUST",
                "PAPER_TRADE_ONLY",
                "PROMISING_CANDIDATE",
                "ROBUST_CANDIDATE"
            }
        )

        accepted_for_paper_trading = (
            classification
            in {
                "PAPER_TRADE_ONLY",
                "PROMISING_CANDIDATE",
                "ROBUST_CANDIDATE"
            }
        )

        return {
            "zero_cost_net_profit": (
                zero_profit
            ),
            "realistic_net_profit": (
                realistic_profit
            ),
            "stress_net_profit": (
                stress_profit
            ),
            "realistic_cost_drag": (
                realistic_cost_drag
            ),
            "stress_cost_drag": (
                stress_cost_drag
            ),
            "realistic_profit_retention_percent": (
                realistic_profit_retention
            ),
            "stress_profit_retention_percent": (
                stress_profit_retention
            ),
            "gross_edge_positive": (
                gross_edge_positive
            ),
            "realistic_edge_positive": (
                realistic_edge_positive
            ),
            "stress_edge_positive": (
                stress_edge_positive
            ),
            "realistic_robust": (
                realistic_robust
            ),
            "stress_robust": (
                stress_robust
            ),
            "classification": (
                classification
            ),
            "explanation": explanation,
            "accepted_for_research": (
                accepted_for_research
            ),
            "accepted_for_paper_trading": (
                accepted_for_paper_trading
            )
        }


    def run(
        self,
        ticker,
        data
    ):
        """
        Esegue tutti gli scenari usando
        esattamente gli stessi dati e parametri.
        """

        scenarios = {}

        for (
            profile_name,
            profile
        ) in self.cost_profiles.items():
            validator = self.build_validator(
                profile
            )

            result = validator.run(
                ticker=ticker,
                data=data
            )

            scenarios[profile_name] = {
                "commission_percent": float(
                    profile[
                        "commission_percent"
                    ]
                ),
                "slippage_percent": float(
                    profile[
                        "slippage_percent"
                    ]
                ),
                "round_trip_cost_percent": (
                    self.calculate_round_trip_cost_percent(
                        commission_percent=(
                            profile[
                                "commission_percent"
                            ]
                        ),
                        slippage_percent=(
                            profile[
                                "slippage_percent"
                            ]
                        )
                    )
                ),
                "summary": result[
                    "summary"
                ],
                "folds": result[
                    "folds"
                ]
            }

        comparison = self.build_comparison(
            scenarios
        )

        return {
            "ticker": ticker,
            "strategy": self.strategy_name,
            "settings": {
                "train_bars": (
                    self.train_bars
                ),
                "test_bars": (
                    self.test_bars
                ),
                "step_bars": (
                    self.step_bars
                ),
                "initial_capital": (
                    self.initial_capital
                ),
                "max_position_percent": (
                    self.max_position_percent
                ),
                "max_holding_bars": (
                    self.max_holding_bars
                ),
                "cooldown_bars": (
                    self.cooldown_bars
                ),
                "max_trades_per_day": (
                    self.max_trades_per_day
                )
            },
            "scenarios": scenarios,
            "comparison": comparison
        }