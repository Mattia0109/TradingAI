from portfolio.portfolio_manager import calculate_position_size



def test_max_position_limit():


    result = calculate_position_size(

        10000,

        {

            "confidence": 95,

            "risk_reward": 3,

            "entry_price": 200

        }

    )


    assert result["allocation_percent"] == 0.10

    assert result["position_value"] == 1000

    assert result["shares"] == 5



def test_bad_trade_blocked():


    result = calculate_position_size(

        10000,

        {

            "confidence": 90,

            "risk_reward": 1,

            "entry_price": 100

        }

    )


    assert result["position_value"] == 0