from adaptive.run_adaptive_scan import infer_asset_class, parse_arguments


def test_infer_asset_class_supports_known_and_external_tickers() -> None:
    assert infer_asset_class("spy") == "EQUITY"
    assert infer_asset_class("btc-usd") == "CRYPTO"
    assert infer_asset_class("eurusd=x") == "FX"
    assert infer_asset_class("aapl") == "EQUITY"


def test_scan_arguments_are_paper_oriented() -> None:
    arguments = parse_arguments(["--tickers", "SPY", "QQQ", "--journal", ":memory:"])

    assert arguments.tickers == ["SPY", "QQQ"]
    assert arguments.journal == ":memory:"
