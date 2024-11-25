# PyHCAS

# Documentazione dell'implementazione MDP

## 1. Introduzione

Questo progetto implementa un processo decisionale di Markov (MDP) per simulare e ottimizzare la gestione di manovre di risoluzione 
tra due velivoli (ownship e intruder) basato su advisory commands. I file principali sono:

- **SolveMDP.py**: Contiene il main script che guida il processo di risoluzione MDP.
- **rewards.py**: Fornisce il calcolo dei reward basati sullo stato e sull'azione.
- **transition.py** (descritto sopra): Definisce le funzioni per la transizione degli stati.

## 2. SolveMDP.py: File principale

Questo file guida il processo decisionale. Si occupa di:
1. Definire il modello MDP, con stati, azioni e dinamiche di transizione.
2. Calcolare i reward associati a ogni stato e azione.
3. Risolvere il problema usando iterazione su \( Q \)-values per trovare la politica ottimale.

Le costanti, come le velocità dei velivoli e i possibili angoli, sono predefinite in questo script per garantire coerenza nei calcoli.

## 3. Funzionamento della funzione `transition`

La funzione `transition` calcola i possibili stati successivi dati uno stato corrente \( s \) e un'azione \( ra \).

**Parametri:**

- `s`: Stato corrente descritto come \( s = (r, t, p, v_{own}, v_{int}, pra) \).
  - \( r \): Distanza tra il velivolo proprio e l'intruder.
  - \( t \): Angolo relativo rispetto alla direzione del velivolo proprio.
  - \( p \): Direzione dell'intruder.
  - \( v_{own}, v_{int} \): Velocità di ownship e intruder.
  - `pra`: Advisory precedente.
- `ra`: Advisory attuale.

**Formula:**

1. **Probabilità congiunte delle azioni:**  
   \[ P(i, j) = P_{own}(i) \cdot P_{int}(j) \]
   Dove \( i \) e \( j \) rappresentano gli indici delle azioni per ownship e intruder rispettivamente.

2. **Dinamiche degli stati:** La dinamica è modellata usando:
   \[
   (r', t', p', v_{own}', v_{int}') = \text{dynamics}(r, t, p, v_{own}, v_{int}, \text{ownTurn}, \text{intTurn}, pra)
   \]

**Output:**

- `nextStates`: Un array di stati successivi \( s' \).
- `nextProbs`: Un array di probabilità associate agli stati successivi.

## 4. Funzionamento della funzione `rewards`

La funzione `rewards` calcola i benefici o i costi di una specifica azione in uno stato. 

**Definizione:**

\[ R(s, a) = \text{Funzione basata su distanza e violazioni di sicurezza} \]

I dettagli sono implementati in `rewards.py`. Generalmente, il reward è negativo per situazioni che aumentano il rischio di collisione 
o violano separazioni minime.

## 5. Calcolo di \( Q_a(s) \)

Il valore \( Q_a(s) \) rappresenta il valore cumulativo atteso di seguire una politica a partire da uno stato \( s \) e un'azione \( a \).

**Formula:**

\[
Q_a(s) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \max_{a'} Q_{a'}(s')
\]

Dove:

- \( \gamma \): Fattore di sconto.
- \( R(s, a) \): Reward immediato.
- \( P(s'|s, a) \): Probabilità di transizione allo stato \( s' \).
- \( \max_{a'} Q_{a'}(s') \): Massimo valore atteso dello stato successivo.

## 6. Rappresentazioni matematiche

### Transizioni

\[
P(s'|s, a) = P_{own}(i) \cdot P_{int}(j)
\]

\[
s' = \text{dynamics}(r, t, p, v_{own}, v_{int}, \text{ownTurn}, \text{intTurn}, pra)
\]

### Rewards

\[
R(s, a) = \text{Funzione basata su distanza e violazioni}
\]

### Calcolo del valore \( Q \)

\[
Q_a(s) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \max_{a'} Q_{a'}(s')
\]

---

## Note Finali

- **Normalizzazione degli angoli:** La funzione `normalize_angle` assicura che gli angoli siano nel range \([-\pi, \pi]\).
- **Funzione `dynamics`:** Aggiorna la posizione e gli angoli usando cinematiche basate sulla fisica, implementate con trigonometria.

Per ulteriori dettagli, consultare i file caricati.

