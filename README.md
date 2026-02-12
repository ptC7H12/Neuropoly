# Neuropoly — Polymarket Trading Pipeline

Vorhersage der Gewinnwahrscheinlichkeit P(win) fuer Trades auf Polymarket.
CPU-only, RAM-effizient, verarbeitet 144 Mio+ Trades.

## Was macht dieses Projekt?

Polymarket ist ein Prediction Market — du wettest auf Ereignisse (z.B. "Wird Trump X tun?").
Jeder Trade hat einen Preis zwischen 0.00 und 1.00, der die Wahrscheinlichkeit widerspiegelt.

Diese Pipeline:
1. Liest historische Trade-Daten + Market-Snapshots
2. Aggregiert Trades in 5-Minuten-Zeitfenster (Buckets)
3. Berechnet 93 Features (Preis-Trends, Volumen, Momentum, ...)
4. Trainiert ein LightGBM-Modell das vorhersagt: **"Wird dieser Trade gewinnen?"**
5. Testet die Vorhersagen mit einem simulierten Backtest

```
Trades (144 Mio+)
     |
     v
 [5-Min Buckets] --> [Luecken fuellen] --> [93 Features] --> [Labels]
                                                                |
                                                                v
                                              [Train / Val / Test Split]
                                                                |
                                                                v
                                                    [LightGBM Training]
                                                    (mit Live-Dashboard)
                                                                |
                                                                v
                                                  [P(win) Vorhersage]
                                                  [Backtest + Metriken]
```

## Voraussetzungen

- Python 3.10+
- Kein GPU noetig (alles auf CPU)

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

## Daten vorbereiten

Du brauchst zwei Datenquellen:

### Trades (deine Trade-Historie)

CSV, Parquet oder SQLite mit diesen Spalten:

| Spalte         | Typ      | Beispiel                           |
|----------------|----------|------------------------------------|
| `timestamp`    | Datetime | `2022-11-21T19:50:09.000000`       |
| `market_id`    | Integer  | `240380`                           |
| `side`         | String   | `token1` (YES) oder `token2` (NO)  |
| `price`        | Float    | `0.50`                             |
| `usd_amount`   | Float    | `50.0`                             |
| `token_amount` | Float    | `100.0`                            |
| `direction`    | String   | `SELL->BUY` (optional)             |

### Markets (Market-Snapshots)

| Spalte       | Typ      | Beispiel                              |
|--------------|----------|---------------------------------------|
| `id`         | Integer  | `517310`                              |
| `question`   | String   | `Will Trump deport less than 250k?`   |
| `volume`     | Float    | `1096126.26`                          |
| `liquidity`  | Float    | `14864.74`                            |
| `yes_price`  | Float    | `0.0445`                              |
| `no_price`   | Float    | `0.9555`                              |
| `close_time` | Datetime | `2025-12-31`                          |

Leg die Dateien z.B. so ab:

```
Neuropoly/
  data/
    trades.csv              # oder trades.parquet
    polymarket_active.csv   # oder polymarket.db (SQLite)
```

## Schnellstart

### Mit CSV-Dateien

```bash
python run_pipeline.py \
  --trades data/trades.csv \
  --markets data/polymarket_active.csv
```

### Mit SQLite-Datenbank

```bash
# Beide Tabellen in einer DB:
python run_pipeline.py \
  --sqlite-path data/polymarket.db \
  --trades-format sqlite \
  --markets-format sqlite \
  --trades-table trades \
  --markets-table markets

# Kurzform (Tabellennamen = "trades" / "markets"):
python run_pipeline.py \
  --trades data/polymarket.db --trades-format sqlite \
  --markets data/polymarket.db --markets-format sqlite
```

### Nur Statistiken ansehen (ohne Training)

```bash
python run_pipeline.py --trades data/trades.csv --markets data/markets.csv --dry-run
```

### Test mit synthetischen Daten (ohne eigene Daten)

```bash
python tests/test_pipeline_e2e.py
```

## Was passiert wenn ich es starte?

Die Pipeline durchlaeuft 8 Schritte:

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

## Live-Dashboard waehrend des Trainings

Waehrend Schritt 7 siehst du in Echtzeit:

- **Fortschritt**: Iteration 250/5000 (5.0%) | 45s elapsed | ETA 855s
- **Metriken**: Train-Loss, Validation-Loss, AUC
- **Feature Importance**: Welche Features am wichtigsten sind
- **Loss-Kurve**: Grafische Darstellung des Trainingsfortschritts

Alle Metriken werden auch in `training_log.jsonl` gespeichert fuer spaetere Analyse.

Das Dashboard laeuft automatisch mit dem `rich`-Paket. Falls du es deaktivieren willst:

```bash
python run_pipeline.py --no-rich ...
```

## Alle CLI-Parameter

| Parameter            | Default    | Beschreibung                          |
|----------------------|------------|---------------------------------------|
| `--trades`           | `data/trades.csv` | Pfad zu Trades-Daten            |
| `--markets`          | `data/polymarket_active.csv` | Pfad zu Markets-Daten |
| `--trades-format`    | `csv`      | Format: `csv`, `parquet`, `sqlite`    |
| `--markets-format`   | `csv`      | Format: `csv`, `parquet`, `sqlite`    |
| `--sqlite-path`      | —          | SQLite-DB-Pfad (wenn beide in einer DB) |
| `--trades-table`     | `trades`   | SQLite Tabellenname fuer Trades       |
| `--markets-table`    | `markets`  | SQLite Tabellenname fuer Markets      |
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

## Projektstruktur

```
Neuropoly/
├── config.py                 # Alle Parameter zentral konfigurierbar
├── run_pipeline.py           # Hauptprogramm (hier starten)
├── requirements.txt          # Python-Abhaengigkeiten
├── pipeline/
│   ├── data_loader.py        # Daten laden (CSV/Parquet/SQLite)
│   ├── aggregation.py        # Trades -> 5-Min-Buckets
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

- **YES-Trade**: Gewinnt wenn der Preis steigt (du hast auf JA gewettet)
- **NO-Trade**: Gewinnt wenn der Preis faellt (du hast auf NEIN gewettet)
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
Jan 2023                   Jun 2025  Jul 2025           Aug 2025  Sep 2025
```

### Was ist P(win)?

Die vom Modell vorhergesagte Wahrscheinlichkeit, dass ein Trade gewinnt. Werte von 0.0 (sicher verloren) bis 1.0 (sicher gewonnen). Wir setzen nur Trades mit P(win) > 0.6.

### Was ist der Brier Score?

Misst wie gut die vorhergesagten Wahrscheinlichkeiten kalibriert sind. Wenn das Modell sagt "70% Gewinnchance", sollten auch wirklich ~70% dieser Trades gewinnen. Niedrigere Werte = besser (0.0 = perfekt, 0.25 = Muenzwurf).

## Tipps

- **Erster Lauf**: Nutze `--dry-run` um die Daten-Statistiken zu pruefen bevor du trainierst
- **Wenig RAM**: Nutze Parquet statt CSV (schneller, kleiner)
- **Langsames Training**: Reduziere `--num-leaves` oder `--max-depth`
- **Overfitting**: Wenn Train-AUC >> Val-AUC, erhoehe `min_child_samples` in `config.py`
- **Modell laden**: Nach dem Training liegt das Modell in `model.txt` und kann wiederverwendet werden

## Lizenz

MIT
