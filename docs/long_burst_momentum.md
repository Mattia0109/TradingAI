# Long Burst Momentum V1

## Scopo

Questa V1 è un **Challenger di ricerca**, separato dal Champion esistente. Cerca
soltanto brevi configurazioni rialziste e può restituire esclusivamente:

- `LONG`: previsione rialzista da verificare nel simulatore;
- `FLAT`: equivalente a `NO_TRADE`.

Non contiene collegamenti a broker, ordini reali, position sizing o leva. Il
backtester usa esposizione unitaria long-only e limita ogni simulazione a un
massimo di cinque barre.

## Pipeline

```text
OHLCV validato
      ↓
CHOP → qualità del regime
      ↓
Lorentzian k-NN → direzione e rendimento storico atteso
Squeeze Momentum → timing e accelerazione
CMF → conferma del volume
      ↓
normalizzazione [-1, +1]
      ↓
score + agreement + penalità di correlazione
      ↓
confidence + edge netto stimato
      ↓
LONG / NO_TRADE
```

Il punteggio statico iniziale è:

```text
RawScore = 0.45 × Lorentzian + 0.35 × Squeeze + 0.20 × CMF
```

I pesi non vengono ottimizzati sul campione e non cambiano durante il backtest.

## Componenti

### Choppiness Index

Misura se il mercato è direzionale o laterale. Non produce direttamente un
segnale. Un regime `SHOCK` azzera la qualità; una release positiva dello
Squeeze accompagnata da breakout viene classificata `BREAKOUT`.

### Classificatore Lorentziano

Il classificatore confronta lo stato corrente con esempi storici mediante:

```text
distance(x, y) = Σ log(1 + |xᵢ - yᵢ|)
```

Le feature sono momentum a 3 e 5 barre, RSI, trend EMA normalizzato per ATR e
volume normalizzato. Una label storica può entrare nei vicini soltanto se il
suo intero orizzonte futuro era già concluso alla data della previsione. Gli
esempi sono inoltre campionati con uno stride per ridurre la dipendenza tra
outcome sovrapposti.

### Squeeze Momentum

Confronta Bollinger Bands e Keltner Channels, quindi misura momentum
detrended e accelerazione. Una release positiva rafforza il timing; uno
squeeze ancora attivo riduce lo score perché il movimento non è ancora
confermato.

### Chaikin Money Flow

CMF e la sua accelerazione verificano se la posizione della chiusura nella
candela e il volume sostengono il rialzo.

## Confidence ed edge

La confidence dipende da:

- score positivo;
- agreement tra i tre componenti;
- qualità del regime;
- qualità dei dati;
- incertezza dei vicini Lorentziani;
- penalità quando i componenti diventano troppo correlati.

Il rendimento atteso combina la media pesata degli outcome dei vicini con una
stima tecnica limitata dalla volatilità. Il segnale viene scartato se l'edge
atteso, dopo il costo round-trip configurato, non supera le soglie minime.

## Garanzie causali

- Tutti gli indicatori usano finestre trailing.
- La decisione alla chiusura `t` può iniziare la simulazione soltanto
  all'apertura `t+1`.
- Le label Lorentziane devono essere completamente mature.
- I segnali senza intero orizzonte futuro non entrano nella diagnostica.
- Modificare prezzi futuri non può cambiare segnali, rendimenti o esposizioni
  precedenti.
- Le esposizioni sono sempre comprese tra `0` e `1`; non esistono short o leva.

## Uscite del simulatore

Il backtester chiude una simulazione per:

- `SIGNAL_DECAY`: il segnale non è più LONG o la confidence è decaduta;
- `TIME_STOP`: raggiunto il massimo di 1–5 barre;
- `END_OF_DATA`: chiusura tecnica dell'ultima osservazione.

MFE e MAE vengono registrati soltanto come diagnostica descrittiva.

## Esecuzione locale

```powershell
python -m adaptive.run_long_burst_backtest `
  --period 5y `
  --interval 1d `
  --horizon-bars 3 `
  --max-holding-bars 5 `
  --cost-bps 10 `
  --tickers SPY QQQ IWM EFA EEM SHY IEF TLT GLD SLV USO DBA UUP FXE FXY
```

Il comando scarica dati storici, esegue il Challenger, calcola diagnostica per
regime e applica lo stesso promotion gate temporale usato dal progetto contro
il Champion statico. Un eventuale `ELIGIBLE_FOR_REVIEW` non promuove
automaticamente la strategia.

Il runtime previsto resta **LEAN locale gratuito**, non QuantConnect Cloud. In
questa fase LEAN continua a essere verificato dallo smoke test; il nuovo
Challenger non viene collegato a un algoritmo di esecuzione finché causalità,
walk-forward e robustezza ai costi non sono dimostrate.
