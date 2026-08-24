# Superficie Lorentziana morbida: recenza x contesto

Questo componente studia come cambia il vicinato storico quando alla distanza
Lorentziana grezza vengono aggiunte penalita' continue e pre-dichiarate per
eta' e disaccordo degli stati contemporanei. Non usa risultati futuri.

## Candidati

Ogni query usa soltanto candidati:

- strettamente precedenti;
- entro 4.000 barre;
- oltre un embargo di quattro barre;
- campionati una barra ogni quattro;
- appartenenti alla stessa fase `OPEN`, `MID_SESSION` o `CLOSE`;
- con feature e stati CHOP/Squeeze disponibili.

Non viene applicato il filtro rigido sul contesto completo. Questo evita di
eliminare a priori molte query, problema osservato con la memoria breve.

## Distanza corretta

Per i profili non di riferimento:

```text
adjusted_distance = raw_lorentzian_distance
                  + ln(2) * age_bars / half_life_bars
                  + context_penalty * context_mismatches
```

`context_mismatches` vale 0, 1 o 2 in base al disaccordo di CHOP e Squeeze.
Il termine di recenza vale `ln(2)` quando l'eta' del candidato coincide con la
half-life. La griglia e' fissata prima dell'analisi:

```text
half-life:       520, 1040, 2080 barre
context penalty: 0.00, 0.25, 0.50, 1.00
```

`REFERENCE_RAW` mantiene la distanza grezza senza penalita'.

## Metriche

Per ogni asset e profilo vengono riportati:

- copertura delle query;
- distanza grezza e corretta mediane;
- eta' mediana e p90 dei vicini;
- frazione dei vicini con contesto completo identico;
- numero medio di disaccordi di contesto;
- numero mediano di sedute distinte rappresentate;
- effective count mediano delle sedute rappresentate;
- quota mediana massima concentrata in una singola seduta;
- sovrapposizione con il vicinato grezzo di riferimento.

Le mediane cross-asset sono riepiloghi descrittivi. Gli asset correlati non
sono considerati osservazioni statistiche indipendenti.

## Limiti

La superficie non seleziona automaticamente un profilo. Un aumento della
coerenza di contesto puo' comportare vicini piu' lontani, meno diversi o una
geometria molto differente. Nessuna di queste metriche misura accuratezza o
redditivita'.

Il modulo non genera direzioni, previsioni, ordini, posizioni, size, stop,
leva, outcome futuri o P&L.

La stabilita' temporale del profilo descrittivo congelato e' trattata
separatamente in `docs/lorentzian_soft_block_stability.md`.
