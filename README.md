# Neuropoly — Polymarket Trading Pipeline

Vorhersage der Gewinnwahrscheinlichkeit P(win) fuer Trades auf Polymarket.
CPU-only, RAM-effizient, verarbeitet 144 Mio+ Trades.

## Was macht dieses Projekt?

Polymarket ist ein Prediction Market — du wettest auf Ereignisse (z.B. "Wird Trump X tun?").
Jeder Trade hat einen Preis zwischen 0.00 und 1.00, der die Wahrscheinlichkeit widerspiegelt.

Diese Pipeline:
1. Konvertiert die Roh-CSV-Dateien (markets.csv + orderFilled.csv) RAM-effizient nach Parquet
2. Aggregiert Trades in 5-Minuten-Zeitfenster (Buckets)
3. Berechnet 93 Features (Preis-Trends, Volumen, Momentum, ...)
4. Trainiert ein LightGBM-Modell das vorhersagt: **"Wird dieser Trade gewinnen?"**
5. Testet die Vorhersagen mit einem simulierten Backtest

```
markets.csv + orderFilled.csv
         |
         v
  [convert_to_parquet.py]        <-- Chunk-weise, kein Full-Load
  markets.parquet + trades.parquet
         |
         v
 [5-Min Buckets] --> [Luecken fuellen] --> [93 Features] --> [Labels]
                                                                |
                                                                v
                                              [Train / Val / Test Split]
                                                                |
                                                                v
                                                    [LightGBM Training]
                                        run_pipeline.py  ODER  train_chunked.py
                                        (komplett)              (RAM-sparend, inkrementell)
                                                                |
                                                                v
                                                  [P(win) Vorhersage]
                                                  [Backtest + Metriken]
```

## Voraussetzungen

- Python 3.10+
- Kein GPU noetig (alles auf CPU)
- Ca. 25 GB RAM empfohlen (oder `train_chunked.py` fuer weniger)

## Installation

```bash
# 1. Repository klonen
git clone https://github.com/ptC7H12/Neuropoly.git
cd Neuropoly

# 2. (Optional) Virtuelle Umgebung erstellen
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Abhaengigkeiten installieren
pip install -r requirements.txt
```

## Datenquellen

### `markets.csv` — Market-Stammdaten

Enthaelt Metadaten zu jedem Polymarket-Markt.

| Spalte        | Typ      | Beispiel                                         |
|---------------|----------|--------------------------------------------------|
| `id`          | Integer  | `12`                                             |
| `question`    | String   | `Will Joe Biden get Coronavirus before the election?` |
| `answer1`     | String   | `Yes`                                            |
| `answer2`     | String   | `No`                                             |
| `volume`      | Float    | `32257.45`                                       |
| `closedTime`  | Datetime | `2020-11-02 16:31:01+00`                         |
| `market_slug` | String   | `will-joe-biden-get-coronavirus-before-the-election` |
| `ticker`      | String   | `will-joe-biden-get-coronavirus-before-the-election` |
| `token1`      | String   | `5313507246...` (256-Bit Token-ID fuer YES)      |
| `token2`      | String   | `6086987146...` (256-Bit Token-ID fuer NO)       |

> **Wichtig**: `token1`/`token2` sind die Schluesselspalten — sie verbinden jeden Trade
> in `orderFilled.csv` mit dem richtigen Markt und der richtigen Seite (YES/NO).
> `yes_price`, `no_price` und `liquidity` sind in dieser Quelle nicht enthalten und werden
> als `null` ergaenzt (LightGBM verarbeitet Nullwerte problemlos).

### `orderFilled.csv` — On-Chain Trade-Ereignisse

Jede Zeile ist ein ausgefuehrtes Order-Match auf der Polymarket-Blockchain (Polygon).

| Spalte              | Typ    | Bedeutung                                              |
|---------------------|--------|--------------------------------------------------------|
| `timestamp`         | Int    | Unix-Timestamp (Sekunden)                              |
| `maker`             | String | Ethereum-Adresse des Makers                            |
| `makerAssetId`      | String | Token-ID oder `"0"` (= USDC-Zahlung)                  |
| `makerAmountFilled` | Int    | Menge in 6-Dezimalstellen (1 000 000 = 1.0)            |
| `taker`             | String | Ethereum-Adresse des Takers                            |
| `takerAssetId`      | String | Token-ID oder `"0"` (= USDC-Zahlung)                  |
| `takerAmountFilled` | Int    | Menge in 6-Dezimalstellen                              |
| `transactionHash`   | String | On-Chain Transaktions-Hash                             |

**Besonderheit**: Jeder Trade erscheint **zweimal** — einmal aus Maker-Sicht, einmal aus Taker-Sicht.
Der Converter filtert automatisch die USDC-Leg-Zeilen (`makerAssetId == "0"`) heraus.

**Preis-Berechnung** (automatisch durch Converter):
```
price        = takerAmountFilled / makerAmountFilled   (USDC pro Token, 0–1)
usd_amount   = takerAmountFilled / 1_000_000           (USDC)
token_amount = makerAmountFilled / 1_000_000           (Conditional Tokens)
```

## Schritt 1: CSV → Parquet konvertieren

```bash
# Markets konvertieren (einfach)
python convert_to_parquet.py markets markets.csv data/markets.parquet

# Trades konvertieren (braucht markets.csv fuer den Token-ID-Lookup)
python convert_to_parquet.py trades orderFilled.csv data/trades.parquet \
    --markets markets.csv

# Ultra-low-RAM: kleinere Chunks (Standard: 500 000 Zeilen)
python convert_to_parquet.py trades orderFilled.csv data/trades.parquet \
    --markets markets.csv --chunk-size 100000

# Beste Kompression (langsamer, kleinere Dateien)
python convert_to_parquet.py trades orderFilled.csv data/trades.parquet \
    --markets markets.csv --compression zstd
```

Waehrend der Konvertierung laufen **immer nur `chunk-size` Zeilen im RAM** —
kein vollstaendiger File-Load noetig. Der Fortschritt wird zeilenweise ausgegeben:

```
  Chunk    1:    500,000 /  72,400,000 (  0.7%) |   3.2s |   156,250 rows/s
  Chunk    2:  1,000,000 /  72,400,000 (  1.4%) |   3.1s |   161,290 rows/s
  ...
```

## Schritt 2: Pipeline starten

### Variante A — Komplette Pipeline (run_pipeline.py)

Fuer Datensaetze die komplett in den RAM passen:

```bash
python run_pipeline.py \
  --trades data/trades.parquet --trades-format parquet \
  --markets data/markets.parquet --markets-format parquet

# Nur Statistiken pruefen (kein Training)
python run_pipeline.py \
  --trades data/trades.parquet --trades-format parquet \
  --markets data/markets.parquet --markets-format parquet \
  --dry-run
```

### Variante B — Inkrementelles Chunk-Training (train_chunked.py)

**Empfohlen bei 25 GB RAM** — verarbeitet Daten in Zeitfenstern und trainiert
LightGBM schrittweise mit Warm-Start (`init_model`):

```bash
# Zuerst bucketing durchfuehren (erzeugt bucketed.parquet):
python run_pipeline.py \
  --trades data/trades.parquet --trades-format parquet \
  --markets data/markets.parquet --markets-format parquet \
  --dry-run

# Dann inkrementell trainieren (90-Tage-Chunks, ~6–8 GB RAM-Peak):
python train_chunked.py \
  --bucketed bucketed.parquet \
  --markets data/markets.parquet \
  --chunk-days 90 \
  --n-estimators 200 \
  --n-jobs 8 \
  --model-path model.txt

# 30-Tage-Chunks fuer noch weniger RAM:
python train_chunked.py \
  --bucketed bucketed.parquet \
  --markets data/markets.parquet \
  --chunk-days 30 \
  --low-memory

# Training fortsetzen (nach Unterbrechung):
python train_chunked.py \
  --bucketed bucketed.parquet \
  --markets data/markets.parquet \
  --init-model model.txt \
  --chunk-days 90
```

### Test mit synthetischen Daten (ohne eigene Daten)

```bash
python tests/test_pipeline_e2e.py
```

## Was passiert wenn ich es starte?

### run_pipeline.py — 8-Schritte-Ausgabe

```
[1/8] Loading data...
  Markets loaded: 1523 rows

[2/8] Aggregating trades into 5-min buckets...
  Bucketed: 2,340,000 rows across 1523 markets

[3/8] Detecting and handling gaps...
  Mean gap ratio: 34.5%
  Markets with >50% gaps: 89
  After filling: 3,100,000 rows (420,000 excluded from training)

[4/8] Engineering features...
  Features: 93 columns

[5/8] Generating labels...
  Labeled: 1,850,000 | Win rate: 0.512

[6/8] Walk-forward split...
  Train: 1,295,000 rows | ends 2025-06-15
  Val:     277,500 rows | 2025-06-15 -> 2025-08-20
  Test:    277,500 rows | starts 2025-08-20

[7/8] Training LightGBM...          <-- Hier siehst du das Live-Dashboard!
  [  100/5000] | 12.3s | train_auc: 0.62 | val_auc: 0.58 | best: 0.58 @95

[8/8] Evaluating on test set...
  ROC AUC: 0.587 | Brier: 0.241
  Backtest: 1,234 trades | Win rate: 58.3% | ROI: 12.4%
```

### train_chunked.py — Inkrementelle Ausgabe

```
[3] Incremental training ...

  Chunk   1: 2020-10-01  →  2020-12-31
    Read 4,312,500 rows (incl. context)
    Rows: 3,890,000 | Labeled: 1,102,000 | Win rate: 0.508
    X shape: (1,012,450, 93) | positives: 514,830
    Done in 47.3s | estimators so far: 200
    Model saved → model.txt

  Chunk   2: 2021-01-01  →  2021-03-31
    ...
```

## Live-Dashboard waehrend des Trainings

Waehrend Schritt 7 siehst du in Echtzeit:

- **Fortschritt**: Iteration 250/5000 (5.0%) | 45s elapsed | ETA 855s
- **Metriken**: Train-Loss, Validation-Loss, AUC
- **Feature Importance**: Welche Features am wichtigsten sind
- **Loss-Kurve**: Grafische Darstellung des Trainingsfortschritts

Alle Metriken werden auch in `training_log.jsonl` gespeichert fuer spaetere Analyse.

```bash
# Dashboard deaktivieren (fuer Log-Dateien / Server)
python run_pipeline.py --no-rich ...
```

## Alle CLI-Parameter

### convert_to_parquet.py

| Parameter        | Default     | Beschreibung                                      |
|------------------|-------------|---------------------------------------------------|
| `source_type`    | —           | `trades` (orderFilled.csv) oder `markets`         |
| `input`          | —           | Pfad zur Eingabe-CSV                              |
| `output`         | —           | Pfad zur Ausgabe-Parquet                          |
| `--markets`      | —           | Pfad zur markets.csv (nur bei `trades` benoetigt) |
| `--chunk-size`   | `500000`    | Zeilen pro Chunk (kleiner = weniger RAM)          |
| `--compression`  | `snappy`    | `snappy` (schnell), `gzip`, `zstd` (klein)        |

### run_pipeline.py

| Parameter            | Default    | Beschreibung                          |
|----------------------|------------|---------------------------------------|
| `--trades`           | `data/trades.csv` | Pfad zu Trades-Daten            |
| `--markets`          | `data/polymarket_active.csv` | Pfad zu Markets-Daten |
| `--trades-format`    | `csv`      | Format: `csv`, `parquet`, `sqlite`    |
| `--markets-format`   | `csv`      | Format: `csv`, `parquet`, `sqlite`    |
| `--bucket-minutes`   | `5`        | Bucket-Groesse in Minuten             |
| `--forward-window`   | `6`        | Label-Fenster (Buckets in die Zukunft)|
| `--learning-rate`    | `0.05`     | LightGBM Lernrate                     |
| `--num-leaves`       | `31`       | LightGBM Blaetter pro Baum           |
| `--max-depth`        | `7`        | Maximale Baumtiefe                    |
| `--n-jobs`           | `10`       | CPU-Kerne fuer Training               |
| `--entry-threshold`  | `0.6`      | Min P(win) fuer Backtest-Trade        |
| `--model-path`       | `model.txt`| Speicherpfad fuer trainiertes Modell  |
| `--log-interval`     | `10`       | Dashboard-Update alle N Iterationen   |
| `--no-rich`          | —          | Rich-Dashboard deaktivieren           |
| `--dry-run`          | —          | Nur Statistiken, kein Training        |
| `--low-memory`       | —          | Reduzierte Features + kleines Modell  |

### train_chunked.py

| Parameter            | Default     | Beschreibung                                         |
|----------------------|-------------|------------------------------------------------------|
| `--bucketed`         | —           | Pfad zur bucketed.parquet (aus run_pipeline.py)      |
| `--markets`          | —           | Pfad zur markets.parquet oder markets.csv            |
| `--markets-format`   | `parquet`   | Format der Markets-Datei                             |
| `--chunk-days`       | `90`        | Tage pro Trainings-Chunk                             |
| `--context-buckets`  | `60`        | Kontext-Buckets an Chunk-Grenzen (fuer Lag-Features) |
| `--model-path`       | `model.txt` | Ausgabepfad fuer das Modell                          |
| `--init-model`       | —           | Vorhandenes Modell weiterschreiben (Warm-Start)      |
| `--test-days`        | `60`        | Tage am Ende fuer Hold-out-Test                      |
| `--n-estimators`     | `200`       | LightGBM-Baeume pro Chunk                            |
| `--learning-rate`    | `0.05`      | Lernrate                                             |
| `--num-leaves`       | `31`        | Blaetter pro Baum                                    |
| `--max-depth`        | `7`         | Maximale Baumtiefe                                   |
| `--n-jobs`           | `8`         | CPU-Kerne                                            |
| `--low-memory`       | —           | Kleineres Modell + weniger Features                  |
| `--no-eval`          | —           | Test-Auswertung ueberspringen                        |

## Projektstruktur

```
Neuropoly/
├── config.py                 # Alle Parameter zentral konfigurierbar
├── run_pipeline.py           # Hauptprogramm — komplette Pipeline
├── train_chunked.py          # Inkrementelles Training fuer ~25 GB RAM
├── convert_to_parquet.py     # CSV → Parquet (chunk-weise, RAM-schonend)
├── requirements.txt          # Python-Abhaengigkeiten
├── pipeline/
│   ├── data_loader.py        # Daten laden (CSV/Parquet/SQLite)
│   ├── aggregation.py        # Trades -> 5-Min-Buckets (streamt nach Parquet)
│   ├── gap_handler.py        # Luecken erkennen + behandeln
│   ├── features.py           # 93 Features berechnen
│   ├── labeling.py           # Win/Loss Labels setzen
│   ├── splitter.py           # Train/Val/Test aufteilen
│   ├── model.py              # LightGBM Training
│   ├── monitor.py            # Live-Dashboard
│   └── evaluation.py         # Metriken + Backtest
└── tests/
    └── test_pipeline_e2e.py  # E2E-Test mit Fake-Daten
```

## Wie funktioniert das Labeling?

Das Modell lernt: "Wenn jemand jetzt einen Trade macht, gewinnt er in 30 Minuten?"

```
Jetzt (Bucket t)          +30 Min (Bucket t+6)
     |                          |
  Preis: 0.45                Preis: 0.52
     |                          |
     +-- YES Trade? ----------> Preis gestiegen --> win = 1
     +-- NO Trade?  ----------> Preis gestiegen --> win = 0
```

- **YES-Trade** (`side = "token1"`): Gewinnt wenn der Preis steigt (du hast auf JA gewettet)
- **NO-Trade** (`side = "token2"`): Gewinnt wenn der Preis faellt (du hast auf NEIN gewettet)
- Buckets in der **Datenluecke** (Okt 2025 – Feb 2026) bekommen kein Label

## Wichtige Konzepte fuer Anfaenger

### Was ist ein Bucket?

Statt jeden einzelnen Trade zu betrachten, fassen wir alle Trades in 5-Minuten-Fenstern zusammen. Das reduziert Rauschen und macht die Daten handhabbar.

### Was ist LightGBM?

Ein Machine-Learning-Algorithmus der Entscheidungsbaeume baut. Fuer tabellarische Daten (Zahlen in Spalten) ist er oft besser als neuronale Netze und deutlich schneller.

### Was ist Walk-Forward Split?

Wir teilen die Daten zeitlich auf: Trainieren mit alten Daten, testen mit neuen. Das simuliert die echte Situation — du kannst nur aus der Vergangenheit lernen.

```
|-------- Training --------|-- Gap --|--- Validation ---|-- Gap --|--- Test ---|
Jan 2020                   Jun 2025  Jul 2025           Aug 2025  Sep 2025
```

### Was ist inkrementelles Training?

Bei sehr grossen Datensaetzen passt nicht alles gleichzeitig in den RAM.
`train_chunked.py` trainiert daher in Zeitscheiben:

```
Chunk 1 (Okt–Dez 2020) → Modell v1
Chunk 2 (Jan–Mär 2021) → Modell v1 + neue Baeume = Modell v2
Chunk 3 (Apr–Jun 2021) → Modell v2 + neue Baeume = Modell v3
...
```

LightGBM's `init_model`-Parameter ermoeglicht dieses Weiterschreiben ohne Neustart.

### Was ist P(win)?

Die vom Modell vorhergesagte Wahrscheinlichkeit, dass ein Trade gewinnt. Werte von 0.0 (sicher verloren) bis 1.0 (sicher gewonnen). Wir setzen nur Trades mit P(win) > 0.6.

### Was ist der Brier Score?

Misst wie gut die vorhergesagten Wahrscheinlichkeiten kalibriert sind. Wenn das Modell sagt "70% Gewinnchance", sollten auch wirklich ~70% dieser Trades gewinnen. Niedrigere Werte = besser (0.0 = perfekt, 0.25 = Muenzwurf).

## RAM-Verbrauch

| Modus                          | Geschaetzter Peak-RAM | Wann benutzen?              |
|-------------------------------|----------------------|-----------------------------|
| `run_pipeline.py`              | 15–25 GB             | Standard, moderater Datensatz |
| `run_pipeline.py --low-memory` | 3–5 GB               | Kleiner Datensatz, wenig RAM |
| `train_chunked.py`             | 6–10 GB pro Chunk    | Grosser Datensatz, ~25 GB RAM |
| `train_chunked.py --low-memory`| 2–4 GB pro Chunk     | Maximale RAM-Einsparung      |
| `convert_to_parquet.py`        | < 1 GB               | Immer — echt streaming       |

## Tipps

- **Erster Lauf**: Nutze `--dry-run` um die Daten-Statistiken zu pruefen bevor du trainierst
- **Datenformat**: Parquet >> CSV (schneller, weniger RAM)
- **Langsames Training**: Reduziere `--num-leaves` oder `--max-depth`
- **Overfitting**: Wenn Train-AUC >> Val-AUC, erhoehe `min_child_samples` in `config.py`
- **Modell fortsetzen**: `train_chunked.py --init-model model.txt` schreibt ein vorhandenes Modell weiter
- **Kompression**: `--compression zstd` bei `convert_to_parquet.py` spart ~30% Speicher vs. snappy

## Lizenz

MIT
