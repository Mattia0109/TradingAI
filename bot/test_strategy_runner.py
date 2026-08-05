from bot.strategy_runner import StrategyRunner



def test_strategy_runner_creation():


    runner = StrategyRunner(

        decision_engine=None,

        ranker=None,

        execution_engine=None

    )


    assert runner is not None