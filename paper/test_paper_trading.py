from paper.broker_simulator import BrokerSimulator



def test_open_and_close_trade():


    broker = BrokerSimulator()


    position = broker.open_position(

        ticker="NVDA",

        direction="LONG",

        quantity=10,

        price=100

    )


    assert len(broker.positions) == 1



    trade = broker.close_position(

        position,

        exit_price=110

    )


    assert trade["pnl"] == 100


    assert broker.balance == 10100