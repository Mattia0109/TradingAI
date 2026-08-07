# TradingAI

TradingAI è un laboratorio Python per ricerca quantitativa multi-mercato,
backtest e selezione prudenziale di candidati di trading.

La nuova fondazione `adaptive` coordina:

`DATA → FEATURE → REGIME → STRATEGIES → META → RISK → EXECUTION GATE → AUDIT → LEARNING`

La fondazione è **paper-only**: non contiene chiamate a broker e
`PAPER_APPROVED` significa soltanto che un candidato ha superato i filtri di
ricerca. Non è una garanzia di profitto né un'autorizzazione a usare capitale
reale.

## Installazione su Windows / Visual Studio Code

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

## Primo adaptive scan

```powershell
python -m adaptive.run_adaptive_scan --period 1y --interval 1d --tickers SPY QQQ IWM EFA EEM SHY IEF TLT GLD SLV USO DBA UUP FXE FXY
```

Il comando scarica dati storici, stampa i candidati e salva l'audit trail in
`.tradingai/adaptive_decisions.db`. Non invia ordini.

Per una prova senza salvare il journal:

```powershell
python -m adaptive.run_adaptive_scan --tickers SPY QQQ GLD --journal :memory:
```

## Documentazione

- [Architettura Adaptive Multi-Market](docs/adaptive_system.md)
- Ricerca multi-asset esistente:

  ```powershell
  python -m backtesting.run_multi_asset_research --period max --interval 1d --frequencies 21 42 63 84 126 --tickers SPY QQQ IWM EFA EEM SHY IEF TLT GLD SLV USO DBA UUP FXE FXY
  ```
