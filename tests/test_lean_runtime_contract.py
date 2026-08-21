import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "lean" / "runtime.json"
RUNNER_PATH = ROOT / "scripts" / "run_lean_smoke.ps1"
ALGORITHM_PATH = ROOT / "lean" / "smoke" / "main.py"


def test_runtime_is_pinned_and_backtest_only() -> None:
    runtime = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert runtime["engine"] == "LEAN"
    assert runtime["image"] == "quantconnect/lean:18020"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["digest"])
    assert runtime["mode"] == "backtesting"
    assert runtime["live_enabled"] is False
    assert runtime["requires_quantconnect_account"] is False


def test_runner_enforces_manifest_and_backtesting_environment() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "[bool]$runtime.live_enabled" in runner
    assert re.search(r'"--environment"\s*,\s*"backtesting"', runner)
    assert re.search(r'"--data-folder"\s*,\s*"/Lean/Data"', runner)
    assert re.search(r'"--results-destination-folder"\s*,\s*"/Results"', runner)
    assert "[regex]::Escape($expectedDigest)" in runner
    assert "--live-mode" not in runner


def test_smoke_algorithm_uses_only_bundled_historical_sample() -> None:
    algorithm = ALGORITHM_PATH.read_text(encoding="utf-8")

    assert "class LeanSmokeTestAlgorithm(QCAlgorithm):" in algorithm
    assert "self.set_start_date(2013, 10, 7)" in algorithm
    assert "self.set_end_date(2013, 10, 11)" in algorithm
    assert 'self.add_equity("SPY", Resolution.MINUTE)' in algorithm
