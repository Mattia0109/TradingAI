# Stabilità descrittiva delle feature intraday

Questo blocco verifica se le distribuzioni delle feature a 15 minuti restano
simili nel tempo e nelle diverse fasi della sessione. Non genera classificazioni
direzionali, segnali, ordini, size, stop, leva, outcome futuri o P&L.

## Pulizia session-aware

Le feature vengono calcolate soltanto dopo aver mantenuto le barre della
sessione regolare USA, dalle 09:30 alle 16:00 di New York. Pre-market,
after-hours e weekend vengono esclusi prima delle finestre mobili: escluderli
soltanto dal report finale contaminerebbe comunque i valori RTH.

I timestamp privi di timezone vengono interpretati come `America/New_York`.
Timestamp ambigui, inesistenti o duplicati causano un errore isolato per asset.

## Blocchi temporali immutabili

Le sessioni vengono assegnate in ordine a blocchi fissi da 15 sessioni. Un
blocco entra nel report soltanto quando è completo. Aggiungere nuove sessioni
non modifica l'identità dei blocchi già conclusi; questo impedisce che un
aggiornamento dei dati riscriva retroattivamente i confronti passati.

Con i 60 giorni disponibili tramite Yahoo si ottengono al massimo quattro
blocchi completi. Sono sufficienti per collaudare il report, non per concludere
che una feature sia robusta attraverso numerosi regimi di mercato.

## Misure

Per ogni feature numerica e transizione fra blocchi vengono calcolati:

- disponibilità dei valori dopo il warm-up;
- mediana e intervallo interquartile;
- Population Stability Index (`PSI`), usando come bin soltanto quantili del
  blocco precedente;
- spostamento della mediana espresso in IQR del blocco precedente.

Le etichette `LOW_SHIFT`, `MODERATE_SHIFT` e `HIGH_SHIFT` descrivono soltanto
quanto è cambiata la distribuzione. Non sono giudizi sulla qualità di una
strategia.

Per `squeeze_state` e `chop_segment` viene usata la distanza di variazione
totale tra le frequenze delle categorie. Il report calcola inoltre:

- mediane separate per `OPEN` (09:30–10:30), `MID_SESSION` e `CLOSE`
  (15:00–16:00);
- correlazioni Spearman tra feature numeriche;
- coppie oltre la soglia di ridondanza dichiarata.

La ridondanza elevata non elimina automaticamente alcuna feature: segnala
soltanto che due descrittori hanno raccontato informazioni simili nel campione.

## Esecuzione locale

```powershell
python -m adaptive.run_intraday_stability_report `
  --period 60d `
  --interval 15m `
  --sessions-per-block 15 `
  --minimum-complete-blocks 3 `
  --tickers SPY QQQ IWM GLD TLT
```

Il runtime previsto resta LEAN locale gratuito, non QuantConnect Cloud. Questo
comando usa il provider dati corrente soltanto per produrre un report storico
descrittivo e non collega le feature a un algoritmo di esecuzione.

