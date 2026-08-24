from pathlib import Path


def test_freeze_baseline_script_is_local_research_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "freeze_intraday_baseline.ps1").read_text(
        encoding="utf-8"
    )

    assert "venv\\Scripts\\python.exe" in script
    assert "adaptive.run_zip_integrity_check" in script
    assert "$zipValidator" not in script
    assert " -c " not in script
    assert "adaptive.run_research_freeze" in script
    assert "adaptive.run_forward_research_readiness" in script
    assert '"--baseline-freeze", $freezePath' in script
    assert '"--candidate-freeze", $freezePath' in script
    assert "run_lean_smoke" not in script
    assert "QuantConnect" not in script
    assert "broker" not in script.lower()


def test_freeze_baseline_script_declares_complete_universe() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "freeze_intraday_baseline.ps1").read_text(
        encoding="utf-8"
    )

    for ticker in (
        "SPY",
        "QQQ",
        "DIA",
        "EEM",
        "AAPL",
        "AMZN",
        "META",
        "TSLA",
        "SPX",
        "NDX",
        "VXX",
    ):
        assert f'"{ticker}"' in script
