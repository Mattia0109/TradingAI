# Stabilita' del vicinato Lorentziano rispetto alla memoria

Questo report verifica quanto cambia la geometria dei vicini storici quando la
memoria Lorentziana viene limitata a profondita' diverse. Non usa etichette
future e non misura accuratezza o rendimento.

## Confronti

Con la configurazione locale predefinita vengono confrontate quattro finestre
annidate:

```text
520, 1040, 2080, 4000 barre da 15 minuti
```

La finestra da 4.000 barre e' il riferimento. Le query vengono valutate solo
dopo che tale profondita' storica e' disponibile. Per ogni query restano
invariati:

- normalizzazione robusta trailing per slot da 15 minuti;
- embargo di quattro barre;
- campionamento storico di una barra ogni quattro;
- otto vicini e almeno sedici candidati.

Il confronto viene eseguito separatamente in due modalita':

- `PHASE_ONLY`: stessa fase `OPEN`, `MID_SESSION` o `CLOSE`;
- `JOINT_CONTEXT`: stessa fase, stesso segmento CHOP e stesso stato Squeeze.

Gli stati insufficienti non vengono riempiti o sostituiti.

## Metriche

Per ogni profondita' il CSV contiene:

- copertura delle query con vicinato completo;
- distanza Lorentziana mediana;
- eta' mediana e p90 dei vicini;
- quota dei vicini condivisi con la finestra di riferimento;
- p10 della sovrapposizione;
- rapporto fra distanza mediana e distanza del riferimento.

Le etichette `HIGH_NEIGHBOR_OVERLAP`, `MODERATE_NEIGHBOR_OVERLAP`,
`LOW_NEIGHBOR_OVERLAP` e `INSUFFICIENT_COMPARISON` descrivono soltanto la
stabilita' geometrica. Le soglie sono dichiarate nel dataclass di
configurazione e non costituiscono approvazioni di modello.

## Causalita'

Ogni candidato precede strettamente la query e rispetta embargo e limite
storico. Le query passate non cambiano se vengono modificati dati successivi.
La finestra piu' corta e' sempre un sottoinsieme temporale della memoria di
riferimento; non vengono selezionati periodi in base a risultati successivi.

Il report non genera direzioni, previsioni, ordini, posizioni, size, stop,
leva, outcome futuri o P&L.

Quando le finestre corte mostrano bassa sovrapposizione e il contesto rigido
riduce la copertura, il report successivo usa penalita' continue senza outcome.
La specifica e' in `docs/lorentzian_soft_surface.md`.
