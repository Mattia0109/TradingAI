# Runtime LEAN locale e gratuito

Questa cartella integra il motore open source LEAN tramite Docker, senza LEAN CLI e senza account QuantConnect.

## Garanzie iniziali

- immagine fissata a `quantconnect/lean:18020` e al digest approvato;
- ambiente forzato su `backtesting`;
- `live_enabled` impostato a `false`;
- nessuna credenziale di broker o QuantConnect;
- risultati locali salvati sotto `.tradingai/lean/results/`, gia ignorato da Git.

Lo smoke test non implementa la strategia di trading: verifica soltanto che Docker, LEAN, Python e i dati storici demo inclusi nell'immagine funzionino insieme.

## Esecuzione

Da PowerShell, nella root del repository:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lean_smoke.ps1
```

Se l'immagine non e presente:

```powershell
docker pull quantconnect/lean:18020
```

Il test usa i dati demo SPY a risoluzione minuto dal 7 all'11 ottobre 2013 inclusi nell'immagine LEAN. Al termine stampa la cartella contenente i risultati JSON.

## Test del contratto di sicurezza

```powershell
python -m pytest tests/test_lean_runtime_contract.py -q
```

Il passaggio a paper trading verra aggiunto separatamente solo dopo backtest riproducibili e controlli out-of-sample. L'operativita con denaro reale non e disponibile in questo runtime.
