# Stabilita' del profilo Lorentziano soft tra blocchi

Questo report congela il profilo descrittivo `HL_2080_CTX_0.5` e misura come
cambia la sua geometria tra blocchi cronologici fissi. Non confronta outcome
futuri e non seleziona o promuove alcun profilo.

## Blocchi

- 30 sessioni per blocco;
- confini ancorati alla prima sessione disponibile;
- i dati futuri non riassegnano le sessioni passate;
- minimo 60 query per rendere un blocco valutabile;
- minimo tre blocchi valutabili per assegnare uno stato di variazione.

Il warm-up senza query e l'ultimo blocco parziale rimangono visibili nel CSV,
ma non entrano nelle metriche aggregate.

## Metriche per blocco

- copertura delle query complete;
- distanza grezza e aggiustata mediane;
- eta' mediana e p90 dei vicini;
- frazione di contesto CHOP/Squeeze identico;
- disaccordi medi di contesto;
- sedute distinte rappresentate;
- effective count delle sedute;
- quota massima attribuita a una singola seduta;
- overlap con il vicinato Lorentziano grezzo.

L'effective count vale `1 / sum(share_sessione^2)`. E' massimo quando gli otto
vicini sono distribuiti uniformemente su otto sedute diverse.

## Stati descrittivi

- `LOW_BLOCK_VARIATION`;
- `MODERATE_BLOCK_VARIATION`;
- `HIGH_BLOCK_VARIATION`;
- `INSUFFICIENT_BLOCKS`.

Le soglie sono dichiarate nel dataclass
`LorentzianSoftBlockStabilityConfig`. Gli stati riassumono variazione di
copertura, distanza, eta', contesto, overlap e diversita'. Non descrivono
accuratezza, rendimento o idoneita' operativa.

## Output

Il report locale puo' esportare due CSV:

```text
lorentzian_soft_block_details.csv
lorentzian_soft_block_stability.csv
```

Il primo mantiene una riga per asset e blocco. Il secondo mantiene una riga
per asset per il profilo congelato.

Il modulo non genera direzioni, previsioni, ordini, posizioni, size, stop,
leva, outcome futuri o P&L.
