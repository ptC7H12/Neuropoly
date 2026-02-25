# Neuropoly — Polymarket Trading Pipeline

Vorhersage der Gewinnwahrscheinlichkeit P(win) fuer Trades auf Polymarket.
CPU-only, RAM-effizient, verarbeitet 144 Mio+ Trades.

---

## Uebersicht

```
markets.csv + orderFilled.csv
        |
        v
[1] convert_to_parquet.py       CSV → Parquet (chunk-weise, < 1 GB RAM)
        |
        v
[2] run_pipeline.py             Bucketing → Features → Labels → Training → model.txt
    ODER train_chunked.py       (RAM-sparend, inkrementell, empfohlen ab 25 GB)
        |
        v
[3] evaluate_model.py           model.txt gegen historische Daten pruefen (kein Neutraining)
    benchmark_strategies.py     Klassische Strategien auf denselben Daten vergleichen
        |
        v
[4] collect_trades.py           Daemon: Live-Trades laufend in SQLite speichern
    +
    live_bid.py                 Gebot pruefen: BID YES / BID NO / NO BID
```

---

## Voraussetzungen

- Python 3.10+
- Kein GPU noetig (alles auf CPU)
- RAM: mindestens 8 GB (25 GB empfohlen fuer vollstaendigen Datensatz)

---

## Phase 0 — Installation

```bash
git clone https://github.com/ptC7H12/Neuropoly.git
cd Neuropoly
```

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## Phase 1 — Daten vorbereiten und Modell trainieren

### Schritt 1: CSV nach Parquet konvertieren

Lege `markets.csv` und `orderFilled.csv` in das Projektverzeichnis.
Erstelle den Datenordner:

```bash
mkdir -p data
```

Markets konvertieren:

```bash
python convert_to_parquet.py markets markets.csv data/markets.parquet
```

Trades konvertieren (benoetigt `markets.csv` fuer den Token-ID-Lookup):

```bash
python convert_to_parquet.py trades orderFilled.csv data/trades.parquet \
    --markets markets.csv
```

Bei wenig RAM (`--chunk-size` verkleinern):

```bash
python convert_to_parquet.py trades orderFilled.csv data/trades.parquet \
    --markets markets.csv --chunk-size 100000
```

Erwartete Ausgabe:
```
  Chunk    1:    500,000 /  72,400,000 (  0.7%) |   3.2s |   156,250 rows/s
  Chunk    2:  1,000,000 /  72,400,000 (  1.4%) |   3.1s |   161,290 rows/s
  ...
```

---

### Schritt 2a: Modell trainieren — Variante A (einfach, mehr RAM)

Fuer Datensaetze die komplett in den RAM passen (~15–25 GB):

```bash
python run_pipeline.py \
    --trades data/trades.parquet --trades-format parquet \
    --markets data/markets.parquet --markets-format parquet \
    --model-path model.txt
```

Nur Statistiken pruefen ohne Training (`--dry-run`):

```bash
python run_pipeline.py \
    --trades data/trades.parquet --trades-format parquet \
    --markets data/markets.parquet --markets-format parquet \
    --dry-run
```

---

### Schritt 2b: Modell trainieren — Variante B (empfohlen, RAM-sparend)

Zuerst einmalig `bucketed.parquet` erzeugen (mit `--dry-run`):

```bash
python run_pipeline.py \
    --trades data/trades.parquet --trades-format parquet \
    --markets data/markets.parquet --markets-format parquet \
    --dry-run
```

Dann inkrementell trainieren (90-Tage-Chunks, ~6–8 GB RAM-Peak):

```bash
python train_chunked.py \
    --bucketed bucketed.parquet \
    --markets data/markets.parquet \
    --chunk-days 90 \
    --n-estimators 200 \
    --n-jobs 8 \
    --model-path model.txt
```

Bei sehr wenig RAM (30-Tage-Chunks, ~2–4 GB):

```bash
python train_chunked.py \
    --bucketed bucketed.parquet \
    --markets data/markets.parquet \
    --chunk-days 30 \
    --low-memory \
    --model-path model.txt
```

Training nach Unterbrechung fortsetzen:

```bash
python train_chunked.py \
    --bucketed bucketed.parquet \
    --markets data/markets.parquet \
    --init-model model.txt \
    --chunk-days 90
```

Ergebnis: `model.txt` (das trainierte LightGBM-Modell)

---

### Test mit synthetischen Daten (ohne eigene Daten)

```bash
python tests/test_pipeline_e2e.py
```

---

## Phase 2 — Modell evaluieren (kein Neutraining)

Prueft wie gut `model.txt` auf den historischen Daten ist.
Laeuft die gesamte Preprocessing-Pipeline (Bucketing → Features → Labels → Split),
laed dann das bestehende Modell und zeigt Metriken fuer alle drei Splits.

```bash
python evaluate_model.py \
    --trades data/trades.parquet --trades-format parquet \
    --markets data/markets.parquet --markets-format parquet \
    --model model.txt
```

Mit angepasstem Backtest-Schwellenwert:

```bash
python evaluate_model.py \
    --trades data/trades.parquet --trades-format parquet \
    --markets data/markets.parquet --markets-format parquet \
    --model model.txt \
    --entry-threshold 0.65
```

Bei wenig RAM:

```bash
python evaluate_model.py \
    --trades data/trades.parquet --trades-format parquet \
    --markets data/markets.parquet --markets-format parquet \
    --model model.txt \
    --low-memory
```

Erwartete Ausgabe:
```
=======================================================
  TRAIN (in-sample — expect high)
=======================================================
  ROC AUC  : 0.7421       <- hoch erwartet, kein echtes Signal
  Brier    : 0.220
  ROI      : 31.2%

=======================================================
  VALIDATION (out-of-sample)
=======================================================
  ROC AUC  : 0.6012       <- echter Indikator waehrend Training
  Brier    : 0.238
  ROI      : 18.7%

=======================================================
  TEST (final holdout — trust this)
=======================================================
  ROC AUC  : 0.5873       <- die wichtige Zahl
  Brier    : 0.241
  Win rate : 61.2%
  ROI      : 14.3%
  Sharpe   : 1.8
```

**Wie du die Zahlen liest:**

| Kennzahl | Bedeutung | Gut | Warnsignal |
|---|---|---|---|
| ROC AUC (Test) | Trennschaerfe | > 0.55 | Train >> Test = Overfitting |
| Brier Score | Kalibrierung | < 0.23 | > 0.25 = schlechter als Muenzwurf |
| ROI (Backtest) | Simulierter Gewinn | > 0% | Negativ = Modell taugt nicht |
| Sharpe Ratio | Risikoadjustiert | > 1.0 | < 0 = inkonsistente Ergebnisse |

---

## Phase 2b — Klassische Strategien benchmarken (kein Modell noetig)

Testet 7 klassische Trading-Techniken auf denselben historischen Daten.
Laeuft die gleiche Preprocessing-Pipeline wie `evaluate_model.py` und gibt
die gleiche Ausgabe — damit das ML-Modell direkt mit Regelstrategien verglichen werden kann.

```bash
python benchmark_strategies.py \
    --trades data/trades.parquet \
    --markets data/markets.parquet
```

Mit angepasstem Backtest-Schwellenwert:

```bash
python benchmark_strategies.py \
    --trades data/trades.parquet \
    --markets data/markets.parquet \
    --entry-threshold 0.55
```

Erwartete Ausgabe (Vergleichstabelle auf dem TEST-Split):

```
  Strategy         Dir     Cover     AUC   Brier  WinRate      ROI  Sharpe  Trades
  ---------------------------------------------------------------------------------
  baseline         follow  100.0%  0.5000  0.249   51.2%    -4.0%   -0.12   45823
  random           follow  100.0%  0.5001  0.250   51.2%    -5.1%   -0.15   18350
  momentum         follow   34.5%  0.5312  0.246   53.4%    +4.2%    0.71     892
  reversion        AGAINST  12.3%  0.5187  0.248   52.1%    +2.1%    0.31     123
  volume           follow    8.2%  0.5421  0.243   55.7%    +6.8%    1.12     234
  closing          follow    5.1%  0.5634  0.238   58.9%    +9.2%    1.41      67
  contrarian       AGAINST  18.7%  0.5023  0.251   51.3%    +1.5%    0.22     456
```

**Strategien im Ueberblick:**

| Strategie | Richtung | Hypothese |
|---|---|---|
| `baseline` | follow | Immer dominante Seite wetten (keine Filterung, Referenzpunkt) |
| `random` | follow | Zufaellige Gebote (Untergrenze, erwarteter ROI ≈ -2×Fee) |
| `momentum` | follow | Kurzzeitige Preisbewegung setzt sich fort (Herdenverhalten) |
| `reversion` | AGAINST | Ueberdehnter Preis kehrt zum 24-Bucket-Mittelwert zurueck |
| `volume` | follow | Ungewoehnlich hohes Volumen = Smart Money, Richtung folgen |
| `closing` | follow | Nahe Markt-Ende konvergieren Preise zur echten Wahrscheinlichkeit |
| `contrarian` | AGAINST | Wenn yes_ratio UND Preis beide extrem sind — gegen die Masse wetten |

**Dir=follow** benutzt das originale `win`-Label (Wette MIT der Marktmehrheit).
**Dir=AGAINST** flippt das Label — ROI > 0 bedeutet: die Masse liegt systematisch falsch.

### benchmark_strategies.py CLI-Parameter

| Parameter | Default | Beschreibung |
|---|---|---|
| `--trades` | `data/trades.parquet` | Pfad zu Trades-Daten |
| `--markets` | `data/markets.parquet` | Pfad zu Markets-Daten |
| `--trades-format` | `parquet` | `csv`, `parquet`, `sqlite` |
| `--markets-format` | `parquet` | `csv`, `parquet`, `sqlite` |
| `--entry-threshold` | `0.6` | Backtest-Einstiegsschwelle (gleich wie evaluate_model.py) |
| `--bucket-minutes` | `5` | Bucket-Groesse in Minuten |
| `--forward-window` | `6` | Label-Fenster in Buckets (6 = 30 Min) |
| `--seed` | `42` | Zufalls-Seed fuer die Random-Strategie |
| `--keep-intermediates` | — | Zwischenparquet-Dateien behalten |
| `--log-file` | `results_log.jsonl` | JSONL-Log fuer historischen Vergleich (s.u.) |

### Historisches Ergebnis-Log (`results_log.jsonl`)

Jeder Lauf von `evaluate_model.py` **und** `benchmark_strategies.py` haengt automatisch
eine Zeile an `results_log.jsonl` an. Am Ende jedes Laufs erscheint eine kompakte
Vergleichstabelle aller bisherigen Laeufe — so sieht man sofort, ob sich das Modell
oder die Strategien verbessert haben.

```
  ──────────────────────────────────────────────────────────────────────────────
  Model history  (3 of 3 shown)
  ──────────────────────────────────────────────────────────────────────────────
  Timestamp              Model             Thr  TestAUC   TestROI  Sharpe  Trades
  ─────────────────────  ────────────────  ───  ───────  ────────  ──────  ──────
  2024-01-10T08:12:00Z   model_v1.txt      60%   0.5721    +8.4%    0.92     312
  2024-01-15T14:23:00Z   model_v2.txt      60%   0.5873   +14.3%    1.81     432
  2024-01-20T09:05:00Z   model_v3.txt      60%   0.6012   +17.1%    2.05     518
```

Das Log wird im JSONL-Format gespeichert (eine JSON-Zeile pro Lauf) und laesst sich
direkt mit Python/pandas/polars lesen:

```python
import polars as pl
log = pl.read_ndjson("results_log.jsonl")
log.filter(pl.col("type") == "model").select(["ts", "test_auc", "test_roi", "test_sharpe"])
```

Logging deaktivieren: `--log-file ''`

---

## Phase 3 — Live-Gebote pruefen

### Schritt 3.1: Token-ID herausfinden

Die Token-ID steht in `markets.csv` in den Spalten `token1` (YES-Seite) oder `token2` (NO-Seite).
Beispiel: `5313507246...` (256-stellige Zahl).

### Schritt 3.2a: Direkter API-Abruf (aktive Maerkte)

Funktioniert bei Maerkten mit genuegend Handelsaktivitaet (~300+ Trades pro Stunde):

```bash
python live_bid.py \
    --token-id <TOKEN_ID_AUS_MARKETS_CSV> \
    --model model.txt \
    --threshold 0.6
```

Erwartete Ausgabe:
```
=======================================================
  Polymarket Live Bid Validator
=======================================================
  Token ID : ...abc123def456
  Model    : model.txt
  Threshold: 60%
  Time     : 2026-02-25 14:32 UTC
=======================================================

[1/5] Loading model...
  Loaded. Features: 93
[2/5] Fetching market metadata...
  Market : Will X happen before Y?
  Closes : 2026-06-01T00:00:00Z
[3/5] Fetching live trades from CLOB API (last ~300 min)...
  API returned 847 raw trades
  Trades in window: 612
[4/5] Aggregating and building features...
  Buckets: 58 x 5 min
[5/5] Predicting...

=======================================================
  Current bucket yes_ratio : 0.32
  Dominant side            : NO  (price must fall to win)
  P(win) for NO            : 0.6741  (67.4%)
  Threshold                : 60.0%
  Decision  : *** BID NO ***
=======================================================
```

Exit-Code: `0` = BID, `1` = NO BID, `2` = Fehler

### Schritt 3.2b: Collector-Daemon + SQLite (zuverlaessiger, empfohlen)

Fuer inaktive Maerkte oder wenn die CLOB-API nicht genuegend Historie zurueckgibt,
sammelt der Daemon laufend Daten in einer lokalen SQLite-Datenbank.

**Terminal 1 — Daemon starten (laeuft dauerhaft):**

```bash
python collect_trades.py \
    --token-ids <TOKEN_ID_A> <TOKEN_ID_B> \
    --db trades.db \
    --poll-interval 60
```

Ausgabe:
```
Collector started  —  DB: trades.db
Tracking 2 token(s) every 60s
  ...abc123def456
  ...fed987cba321
Press Ctrl+C to stop.

[14:32:01 UTC] ...abc123def456  +23 new trades
[14:32:01 UTC] ...fed987cba321  +7 new trades
[14:33:01 UTC] ...abc123def456  +18 new trades
...
```

Warte mindestens **4 Stunden**, dann hat der Daemon genuegend Daten fuer alle Features.

**Terminal 2 — Gebot pruefen (mit DB):**

```bash
python live_bid.py \
    --token-id <TOKEN_ID_A> \
    --db trades.db \
    --model model.txt \
    --threshold 0.6
```

### YES vs. NO — was bedeutet die Entscheidung?

Das Modell erkennt automatisch welche Seite dominant ist:

- `yes_ratio > 0.5` → Mehrheit kauft YES → `BID YES` bedeutet: **YES-Token kaufen** (wettest auf Preisanstieg)
- `yes_ratio <= 0.5` → Mehrheit kauft NO → `BID NO` bedeutet: **NO-Token kaufen** (wettest auf Preisrueckgang)

---

## Projektstruktur

```
Neuropoly/
├── config.py               Alle Parameter zentral konfigurierbar
│
├── convert_to_parquet.py   [Phase 1] CSV → Parquet (chunk-weise, RAM-schonend)
├── run_pipeline.py         [Phase 1] Komplette Pipeline inkl. Training
├── train_chunked.py        [Phase 1] Inkrementelles Training (~25 GB RAM)
│
├── evaluate_model.py       [Phase 2]  Modell auswerten ohne Neutraining
├── benchmark_strategies.py [Phase 2b] Klassische Strategien vs. Modell vergleichen
│
├── collect_trades.py       [Phase 3] Daemon: Live-Trades → SQLite
├── live_bid.py             [Phase 3] Live-Gebot pruefen mit model.txt
│
├── requirements.txt        Python-Abhaengigkeiten
│
├── pipeline/
│   ├── data_loader.py      Daten laden (CSV/Parquet/SQLite)
│   ├── aggregation.py      Trades → 5-Min-Buckets
│   ├── gap_handler.py      Luecken erkennen + behandeln
│   ├── features.py         93 Features berechnen
│   ├── labeling.py         Win/Loss Labels setzen
│   ├── splitter.py         Train/Val/Test aufteilen
│   ├── model.py            LightGBM Training + Inference
│   ├── monitor.py          Live-Dashboard
│   └── evaluation.py       Metriken + Backtest
└── tests/
    └── test_pipeline_e2e.py  E2E-Test mit Fake-Daten
```

---

## CLI-Referenz

### convert_to_parquet.py

| Parameter | Default | Beschreibung |
|---|---|---|
| `source_type` | — | `trades` oder `markets` |
| `input` | — | Pfad zur Eingabe-CSV |
| `output` | — | Pfad zur Ausgabe-Parquet |
| `--markets` | — | Pfad zur markets.csv (nur bei `trades`) |
| `--chunk-size` | `500000` | Zeilen pro Chunk (kleiner = weniger RAM) |
| `--compression` | `snappy` | `snappy` (schnell), `gzip`, `zstd` (klein) |

### run_pipeline.py

| Parameter | Default | Beschreibung |
|---|---|---|
| `--trades` | `data/trades.csv` | Pfad zu Trades-Daten |
| `--markets` | `data/polymarket_active.csv` | Pfad zu Markets-Daten |
| `--trades-format` | `csv` | `csv`, `parquet`, `sqlite` |
| `--markets-format` | `csv` | `csv`, `parquet`, `sqlite` |
| `--bucket-minutes` | `5` | Bucket-Groesse in Minuten |
| `--forward-window` | `6` | Label-Fenster (Buckets voraus, 6 = 30 Min) |
| `--learning-rate` | `0.05` | LightGBM Lernrate |
| `--num-leaves` | `31` | LightGBM Blaetter pro Baum |
| `--max-depth` | `7` | Maximale Baumtiefe |
| `--n-jobs` | `10` | CPU-Kerne fuer Training |
| `--entry-threshold` | `0.6` | Min P(win) fuer Backtest-Trade |
| `--model-path` | `model.txt` | Speicherpfad fuer trainiertes Modell |
| `--no-rich` | — | Rich-Dashboard deaktivieren |
| `--dry-run` | — | Nur Statistiken, kein Training |
| `--low-memory` | — | Reduzierte Features + kleines Modell |

### train_chunked.py

| Parameter | Default | Beschreibung |
|---|---|---|
| `--bucketed` | — | Pfad zur bucketed.parquet (aus run_pipeline.py --dry-run) |
| `--markets` | — | Pfad zur markets.parquet oder markets.csv |
| `--markets-format` | `parquet` | Format der Markets-Datei |
| `--chunk-days` | `90` | Tage pro Trainings-Chunk |
| `--model-path` | `model.txt` | Ausgabepfad fuer das Modell |
| `--init-model` | — | Vorhandenes Modell weiterschreiben (Warm-Start) |
| `--n-estimators` | `200` | LightGBM-Baeume pro Chunk |
| `--learning-rate` | `0.05` | Lernrate |
| `--n-jobs` | `8` | CPU-Kerne |
| `--low-memory` | — | Kleineres Modell + weniger Features |

### evaluate_model.py

| Parameter | Default | Beschreibung |
|---|---|---|
| `--trades` | `data/trades.parquet` | Pfad zu Trades-Daten |
| `--markets` | `data/markets.parquet` | Pfad zu Markets-Daten |
| `--trades-format` | `parquet` | `csv`, `parquet`, `sqlite` |
| `--markets-format` | `parquet` | `csv`, `parquet`, `sqlite` |
| `--model` | `model.txt` | Pfad zum trainierten Modell |
| `--entry-threshold` | `0.6` | Backtest-Einstiegsschwelle |
| `--bucket-minutes` | `5` | Bucket-Groesse (muss mit Training uebereinstimmen) |
| `--forward-window` | `6` | Label-Fenster (muss mit Training uebereinstimmen) |
| `--low-memory` | — | Kleineres Feature-Set |
| `--keep-intermediates` | — | Zwischenparquet-Dateien behalten |
| `--log-file` | `results_log.jsonl` | JSONL-Log fuer historischen Vergleich (s.u.) |

### collect_trades.py

| Parameter | Default | Beschreibung |
|---|---|---|
| `--token-ids` | — | Eine oder mehrere Token-IDs (Leerzeichen getrennt) |
| `--db` | `trades.db` | Pfad zur SQLite-Datenbank |
| `--poll-interval` | `60` | Sekunden zwischen API-Abfragen |
| `--keep-days` | `7` | Tage Handelshistorie in der DB behalten |

### live_bid.py

| Parameter | Default | Beschreibung |
|---|---|---|
| `--token-id` | — | Polymarket Token-ID (aus markets.csv token1/token2) |
| `--model` | `model.txt` | Pfad zum trainierten Modell |
| `--threshold` | `0.6` | P(win)-Schwellenwert fuer BID (60%) |
| `--history-buckets` | `60` | Anzahl 5-Min-Buckets (60 = 5 Stunden) |
| `--db` | — | SQLite-DB aus collect_trades.py (optional) |
| `--verbose` | — | Alle Feature-Werte ausgeben |

---

## RAM-Verbrauch

| Modus | Geschaetzter Peak-RAM | Wann benutzen? |
|---|---|---|
| `convert_to_parquet.py` | < 1 GB | Immer — echt streaming |
| `run_pipeline.py` | 15–25 GB | Standard, moderater Datensatz |
| `run_pipeline.py --low-memory` | 3–5 GB | Kleiner Datensatz, wenig RAM |
| `train_chunked.py` | 6–10 GB pro Chunk | Grosser Datensatz, ~25 GB RAM |
| `train_chunked.py --low-memory` | 2–4 GB pro Chunk | Maximale RAM-Einsparung |
| `evaluate_model.py` | 3–8 GB | Wie run_pipeline.py ohne Training |
| `collect_trades.py` | < 100 MB | Dauerhafter Daemon |
| `live_bid.py` | < 500 MB | Einmaliger Aufruf |

---

## Konzepte

### Was ist ein Bucket?

Statt jeden einzelnen Trade zu betrachten, fassen wir alle Trades in 5-Minuten-Fenstern
zusammen. Das reduziert Rauschen und macht die Daten handhabbar.

### Was ist LightGBM?

Ein Machine-Learning-Algorithmus der Entscheidungsbaeume baut. Fuer tabellarische Daten
(Zahlen in Spalten) ist er oft besser als neuronale Netze und deutlich schneller.

### Was ist Walk-Forward Split?

Wir teilen die Daten zeitlich auf: Trainieren mit alten Daten, testen mit neuen.
Das simuliert die echte Situation — du kannst nur aus der Vergangenheit lernen.

```
|-------- Training --------|-- Gap --|-- Validation --|-- Gap --|--- Test ---|
Jan 2020                   Jun 2025  Jul 2025          Aug 2025  Sep 2025
```

### Was ist P(win)?

Die vom Modell vorhergesagte Wahrscheinlichkeit, dass ein Trade gewinnt.
Werte von 0.0 (sicher verloren) bis 1.0 (sicher gewonnen).
Standard-Schwellenwert: 0.6 (60%).

### Wie wird das Label bestimmt?

```
Jetzt (Bucket t)          +30 Min (Bucket t+6)

  Preis: 0.45             Preis: 0.52
     |                          |
     +-- Mehrheit kauft YES --> Preis gestiegen --> win = 1  (BID YES)
     +-- Mehrheit kauft NO  --> Preis gestiegen --> win = 0  (BID NO)
```

- `yes_ratio > 0.5` im Bucket → Mehrheit kauft YES → `win=1` wenn Preis steigt
- `yes_ratio <= 0.5` im Bucket → Mehrheit kauft NO → `win=1` wenn Preis faellt

### Was ist der Brier Score?

Misst wie gut die vorhergesagten Wahrscheinlichkeiten kalibriert sind.
Niedrigere Werte = besser (0.0 = perfekt, 0.25 = Muenzwurf).

### Was ist inkrementelles Training?

Bei sehr grossen Datensaetzen passt nicht alles gleichzeitig in den RAM.
`train_chunked.py` trainiert daher in Zeitscheiben:

```
Chunk 1 (Okt–Dez 2020) → Modell v1
Chunk 2 (Jan–Mar 2021) → Modell v1 + neue Baeume = Modell v2
Chunk 3 (Apr–Jun 2021) → Modell v2 + neue Baeume = Modell v3
...
```

---

## Tipps und Fehlerbehebung

**Overfitting erkennen:** Wenn Train-AUC >> Test-AUC (z.B. 0.80 vs 0.52), hat das Modell
die Trainingsdaten auswendig gelernt. Abhilfe: `min_child_samples` in `config.py` erhoehen.

**Zu wenig Trades fuer live_bid.py:** Bei inaktiven Maerkten `collect_trades.py` mindestens
4 Stunden laufen lassen, dann `--db trades.db` verwenden.

**Training zu langsam:** `--num-leaves 15 --max-depth 5` oder `--low-memory` verwenden.

**RAM-Fehler bei run_pipeline.py:** Auf `train_chunked.py` wechseln oder `--low-memory` setzen.

**Modell weiterschreiben:** `train_chunked.py --init-model model.txt` fuegt neue Baeume
zu einem vorhandenen Modell hinzu ohne von vorne zu beginnen.

**Dashboard deaktivieren** (fuer Logs / Server ohne Terminal):

```bash
python run_pipeline.py --no-rich ...
```

---

## Lizenz

MIT
