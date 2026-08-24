# Gate di ricerca per dati a 15 minuti

Questo blocco verifica la qualità dei dati intraday prima di studiare feature e
regimi. Non è una strategia, non esegue un backtest operativo e non calcola
segnali, ordini, size, stop, leva o P&L.

## Perché il V1 giornaliero non viene semplicemente portato a 15m

Passare da barre giornaliere a barre da 15 minuti cambia microstruttura,
sessioni, gap overnight, distribuzione dei volumi, rumore e incidenza dei
costi. I parametri giornalieri non sono quindi trasferibili automaticamente.
Prima serve dimostrare che timestamp e barre siano coerenti e che lo storico
copra abbastanza condizioni differenti.

L'architettura multi-timeframe prevista resta separata per responsabilità:

- timeframe superiori: contesto e regime descrittivo;
- 15 minuti: studio del comportamento delle feature;
- timeframe inferiori: fuori dallo scopo di questo gate.

## Profilo V1

Il profilo iniziale copre ETF e azioni USA nella sessione regolare di New York,
dalle 09:30 alle 16:00. Una sessione completa contiene 26 barre da 15 minuti.
Crypto e FX hanno calendari differenti e richiederanno profili dedicati.

L'audit misura:

- timestamp invalidi, duplicati o fuori dalla griglia 15m;
- barre OHLCV mancanti o incoerenti;
- copertura e completezza delle sessioni osservate;
- barre pre-market/after-hours escluse dal campione RTH;
- volume nullo, chiusure stale e outlier robusti;
- profondità dello storico disponibile.

Gli esiti significano:

- `READY`: dati utilizzabili per ricerca descrittiva sulle feature;
- `LIMITED`: controlli tecnici possibili, ma storico o qualità insufficienti
  per conclusioni robuste tra regimi;
- `REJECTED`: difetti strutturali da correggere prima dell'analisi.

Nessuno dei tre esiti approva l'uso operativo della strategia.

## Limite della sorgente dati corrente

Il provider corrente usa Yahoo tramite `yfinance`. La documentazione di
`yfinance.download` indica che i dati intraday non possono estendersi oltre gli
ultimi 60 giorni. Questa finestra è utile per collaudare la pipeline, ma non è
sufficiente da sola per una valutazione multi-regime. Per uno storico più lungo
servirà una sorgente intraday licenziata e un formato dati riproducibile.

## Esecuzione locale

```powershell
python -m adaptive.run_intraday_audit --period 30d --interval 15m --tickers SPY QQQ IWM GLD TLT
```

Il comando scarica e controlla i dati; non produce output di trading.

## Parità degli indicatori

Prima di implementare feature equivalenti alla specifica servono codice o
definizione esatta, parametri e versione di:

1. Lorentzian Classification;
2. Squeeze Momentum;
3. Choppiness Index (CHOP);
4. Chaikin Money Flow (CMF), se diverso dalla formula standard.

Senza queste sorgenti si potrebbe costruire soltanto un'approssimazione, non
una replica verificabile. Gli script verranno usati come riferimento per test
di parità numerica e causalità, non per produrre indicazioni operative.
