from portfolio.portfolio_risk import PortfolioRiskManager



def test_exposure_limit():


    manager = PortfolioRiskManager()


    positions = [

        {
            "ticker": "AAPL",
            "value": 1000
        },

        {
            "ticker": "NVDA",
            "value": 1000
        }

    ]


    result = manager.can_open_position(

        10000,

        positions,

        1000

    )


    assert result["approved"] is True



def test_max_positions():


    manager = PortfolioRiskManager()


    positions = [

        {
            "ticker": "A",
            "value": 1000
        },

        {
            "ticker": "B",
            "value": 1000
        },

        {
            "ticker": "C",
            "value": 1000
        },

        {
            "ticker": "D",
            "value": 1000
        },

        {
            "ticker": "E",
            "value": 1000
        }

    ]


    result = manager.can_open_position(

        10000,

        positions,

        1000

    )


    assert result["approved"] is False