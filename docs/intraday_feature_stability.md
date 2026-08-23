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

Le due misure Squeeze usate nel confronto di stabilità sono rese
`dimensionless` dividendo valore e variazione per il close della stessa barra.
Le serie originali restano inalterate nel motore di parità. Questa derivazione
evita che un semplice cambiamento dell'unità o del livello nominale del prezzo
venga scambiato per drift della feature; non elimina invece i cambiamenti reali
di volatilità o forma della distribuzione.

Le etichette `LOW_SHIFT`, `MODERATE_SHIFT` e `HIGH_SHIFT` descrivono soltanto
quanto è cambiata la distribuzione. Non sono giudizi sulla qualità di una
strategia.

Il report non usa più soltanto il massimo osservato. Per ogni feature mostra
anche quante transizioni sono elevate, la sequenza consecutiva più lunga,
l'ultimo livello e uno dei seguenti pattern:

- `LOW_OR_NONE`: nessuna transizione elevata;
- `ISOLATED_SHIFT`: cambiamento presente in una sola zona del campione;
- `PERSISTENT_ELEVATED`: almeno due transizioni elevate consecutive;
- `PERSISTENT_HIGH`: almeno due transizioni `HIGH_SHIFT` consecutive;
- `INSUFFICIENT`: evidenza numerica non sufficiente.

La persistenza riduce il rischio di attribuire importanza a un singolo massimo
casuale, ma quattro blocchi restano un campione molto piccolo.

## Calibrazione per piccoli campioni

Le fasi `OPEN` e `CLOSE` contengono soltanto quattro barre per sessione. Con
blocchi da 15 sessioni il PSI grezzo può quindi apparire elevato anche quando
due campioni provengono dalla stessa distribuzione. Il report applica due
salvaguardie:

- una pseudocount di Jeffreys evita che un bin vuoto produca valori estremi;
- 96 riassegnazioni deterministiche di sessioni intere stimano il 95°
  percentile del rumore atteso per PSI e spostamento mediano.

Un'etichetta numerica può diventare `MODERATE_SHIFT` o `HIGH_SHIFT` soltanto se
supera sia la soglia assoluta dichiarata sia la soglia empirica. Le colonne
`PSI_EX` e `MED_EX` mostrano la parte eccedente tale soglia. Le sessioni vengono
riassegnate come unità, preservando la dipendenza tra le barre della stessa
giornata.

Per `squeeze_state` e `chop_segment` viene usata la distanza di variazione
totale tra le frequenze delle categorie. Il report calcola inoltre:

- mediane separate per `OPEN` (09:30–10:30), `MID_SESSION` e `CLOSE`
  (15:00–16:00);
- drift e persistenza separati per ciascuna delle tre fasi, evitando che una
  variazione circoscritta venga nascosta dalla distribuzione giornaliera;
- correlazioni Spearman tra feature numeriche;
- coppie oltre la soglia di ridondanza dichiarata.

Il riepilogo cross-asset conta, per ogni feature e fase, quanti asset mostrano
drift persistente e quanti lo mostrano nell'ultima transizione. Non costruisce
una classifica e non tratta gli ETF come osservazioni indipendenti: SPY, QQQ e
IWM condividono infatti una parte importante del rischio di mercato.

La ridondanza elevata non elimina automaticamente alcuna feature: segnala
soltanto che due descrittori hanno raccontato informazioni simili nel campione.

Il riepilogo cross-asset usa le etichette calibrate e descrive l'ultimo
cambiamento come `NO_SHARED_SHIFT`, `ISOLATED_ASSET_SHIFT`,
`MIXED_ASSET_SHIFT` o `COMMON_SHIFT`. Queste categorie distinguono la portata
del cambiamento, non la sua direzione o utilità. Poiché gli ETF osservati sono
correlati, i conteggi non equivalgono a osservazioni statistiche indipendenti.

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
