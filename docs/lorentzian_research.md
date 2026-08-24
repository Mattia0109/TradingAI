# Lorentzian Neighborhood 15m — ricerca descrittiva

Questo componente confronta ogni stato intraday con stati storici simili. E'
un descrittore di struttura locale dei dati, non un classificatore direzionale
e non una strategia approvata.

## Formula scelta

Fra le due sorgenti fornite esistono definizioni diverse. La sorgente
`Lorentzian Classification` usa, da due a cinque feature:

```text
distance(x, y) = sum(log(1 + abs(x_i - y_i)))
```

Questa e' la formula implementata. La radice quadrata presente nell'altra
sorgente non e' equivalente e viene esclusa. Il codice Pine non viene copiato:
l'implementazione Python e' clean-room e la parita' riguarda soltanto la
formula matematica osservabile.

Il classificatore completo della sorgente non puo' essere dichiarato equivalente
perche' dipende da librerie importate ma non incluse (`MLExtensions/2`,
`KernelFunctions/2` e, nell'altro script, `BankNifty_CSM/16`). Non vengono
ricostruiti segnali, etichette direzionali o statistiche operative.

## Feature usate

Il vettore usa soltanto feature gia' verificate nel progetto:

- Squeeze Momentum diviso per il close;
- variazione dello Squeeze Momentum divisa per il close;
- Choppiness Index;
- Chaikin Money Flow.

Ogni feature viene normalizzata con mediana e IQR trailing. Nessuna statistica
globale o futura entra nella trasformazione. Il valore normalizzato viene
limitato a `[-8, 8]` per evitare che un singolo dato estremo domini tutte le
distanze.

Il motore accetta inoltre un contesto separato per la normalizzazione. Il
report sui file 1m locali usa lo slot esatto della sessione (`09:30`, `09:45`,
ecc.): ogni barra viene quindi standardizzata soltanto rispetto allo stesso
orario storico, con 60 osservazioni trailing e 30 di warm-up. Il contesto dei
vicini resta invece la fase piu' ampia `OPEN`, `MID_SESSION` o `CLOSE`.

## Vicinato causale

Per ogni barra il motore:

1. usa soltanto righe strettamente precedenti;
2. esclude le quattro barre piu' recenti tramite embargo;
3. campiona una riga ogni quattro per ridurre dipendenza seriale;
4. confronta soltanto la stessa fase `OPEN`, `MID_SESSION` o `CLOSE`;
5. limita la memoria a 2.000 barre;
6. richiede almeno 16 candidati;
7. seleziona gli otto vicini con distanza minima, con ordinamento deterministico.

Queste regole sono scelte clean-room e non vengono presentate come parita' del
classificatore originale.

## Output

Il motore produce esclusivamente:

- numero di vicini validi;
- distanza minima e mediana;
- IQR delle distanze;
- densita' locale `1 / (1 + distanza mediana)`;
- eta' minima e mediana dei vicini in barre.

Non produce direzione, previsione, operazione, entrata, uscita, size, stop,
leva, outcome futuro o P&L. Il report CLI riepiloga l'intero campione e non
stampa il valore dell'ultima barra.

## Esecuzione locale

Nel virtual environment del progetto:

```powershell
python -m adaptive.run_lorentzian_research_report `
  --period 60d `
  --interval 15m `
  --tickers SPY QQQ IWM GLD TLT
```

Questo report non richiede TradingView Premium, Docker o QuantConnect Cloud.
Usa la pipeline dati gia' presente. Lo storico intraday gratuito disponibile
resta limitato e consente soltanto ricerca descrittiva, non conclusioni robuste
tra molti regimi.

## Verifiche

I test coprono:

- formula esatta e assenza della radice quadrata;
- invarianza a trasformazioni affini positive delle feature;
- immutabilita' dei risultati passati quando vengono modificati dati futuri;
- isolamento e causalita' della normalizzazione per slot 15m;
- rispetto dell'embargo;
- determinismo e numero fisso di vicini;
- gestione di storia insufficiente e feature costanti;
- assenza di campi operativi nell'output.

Il report locale esteso confronta inoltre profondita' storiche annidate e due
filtri di contesto. La specifica e' in
`docs/lorentzian_neighborhood_stability.md`; tale confronto misura soltanto
copertura, eta', distanza e sovrapposizione dei vicini.
