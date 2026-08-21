# Parità delle feature intraday

Questo blocco traduce in Python soltanto le formule numeriche necessarie allo
studio descrittivo delle feature a 15 minuti. Non contiene combinazioni di
punteggi, classificazioni direzionali, segnali, ordini, size, stop, leva o P&L.

I sorgenti Pine ricevuti non vengono copiati nel repository. Le formule sono
state reimplementate separatamente e accompagnate da test deterministici e
anti-look-ahead.

## Contratto di parità

| Componente | Stato | Note |
|---|---|---|
| Squeeze Momentum | `EXACT_REFERENCE_FORMULA` | SMA, deviazione standard population, True Range e regressione lineare trailing |
| Choppiness Index | `EXACT_REFERENCE_FORMULA` | periodo 10, ATR Wilder/RMA, soglie strette 60/40 |
| Chaikin Money Flow | `EXACT_REFERENCE_FORMULA` | periodo 20 e gestione `high == low` equivalente al riferimento |
| Distanza Lorentziana | `EXACT_REFERENCE_FORMULA` | somma di `log(1 + differenza assoluta)` su 2–5 feature |
| Classificatore Lorentzian completo | `PARTIAL_MISSING_LIBRARY_SOURCE` | dipende dalle implementazioni importate di normalizzazioni, filtri e kernel |

`EXACT_REFERENCE_FORMULA` certifica la traduzione della formula disponibile,
non l'identità numerica con un grafico TradingView finché non viene confrontato
un export di valori sulle stesse barre e con la stessa sorgente dati.

## Anomalia conservata nel riferimento Squeeze

Il sorgente fornito dichiara un moltiplicatore Bollinger pari a 2,0, ma nella
formula applica il moltiplicatore Keltner pari a 1,5 anche alla deviazione delle
Bollinger Bands. Il motore conserva esattamente questo comportamento. Cambiare
il parametro BB dichiarato non cambia quindi il risultato; un test impedisce
che questa particolarità venga corretta silenziosamente.

## Lorentzian: limite verificabile

Il sorgente principale importa due librerie esterne:

- `jdehorty/MLExtensions/2` per RSI, WaveTrend, CCI e ADX normalizzati e per i
  filtri;
- `jdehorty/KernelFunctions/2` per le stime kernel.

Le pagine ufficiali pubblicano il contratto delle funzioni, ma non forniscono
nel testo acquisibile tutte le equazioni necessarie a una replica numerica
indipendente. Per questo motivo il progetto implementa soltanto la distanza
Lorentziana esatta e marca il classificatore completo come parziale. Non usa
approssimazioni nascoste.

Riferimenti ufficiali:

- [Machine Learning: Lorentzian Classification](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/)
- [MLExtensions](https://www.tradingview.com/script/ia5ozyMF-MLExtensions/)
- [KernelFunctions](https://www.tradingview.com/script/e0Ek9x99-KernelFunctions/)

## Report distributivo

```powershell
python -m adaptive.run_intraday_feature_report --period 60d --interval 15m --tickers SPY QQQ IWM GLD TLT
```

Il report mostra soltanto quantità di righe valide e distribuzioni sull'intero
campione. Non mostra l'ultima lettura, non calcola performance e non modifica
il Challenger esistente.
