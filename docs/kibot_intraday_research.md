# Kibot tick -> ricerca descrittiva 15m

`adaptive.run_kibot_intraday_research` collega un campione Kibot nel formato
`Date,Time,Price,Bid,Ask,Size` alla pipeline intraday descrittiva.

Il flusso e':

1. validazione dei tick e delle quote;
2. aggregazione causale in barre OHLCV da 15 minuti;
3. calcolo delle formule Squeeze Momentum, CHOP e CMF;
4. confronto della stabilita' in blocchi fissi di sedute;
5. normalizzazione Lorentziana trailing per slot 15m;
6. ricerca di vicini soltanto storici, con embargo e stessa fase di seduta;
7. riepilogo della microstruttura osservata per apertura, centro e chiusura.

Esempio:

```powershell
python -m adaptive.run_kibot_intraday_research `
  --file "C:\path\to\IVE (1).txt" `
  --ticker IVE
```

Per i campioni sotto il minimo descrittivo, il runner usa configurazioni
esplicitamente ridotte soltanto per verificare la pipeline. Questa scelta non
trasforma uno storico breve in una validazione statistica. I numeri osservati
restano nel report locale e non vengono incorporati nella documentazione
pubblica.

Il runner non contiene classificazione direzionale, outcome futuri, segnali,
ordini, size, stop, leva o P&L. Nessun risultato approva una strategia.
