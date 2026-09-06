# Neuropoly — Polymarket Trading Pipeline

Vorhersage der Gewinnwahrscheinlichkeit P(win) fuer Trades auf Polymarket.
CPU-only, RAM-effizient, verarbeitet 144 Mio+ Trades.

---

## Wichtig: bestehende Artefakte neu erzeugen

Zwei Aenderungen veraendern die Bedeutung der Daten selbst. Vorhandene
`bucketed.parquet`-Dateien und jedes darauf trainierte `model.txt` sind
damit ungueltig und muessen neu erzeugt werden:

1. **Preise sind jetzt durchgaengig P(YES).** NO-seitige Trades werden beim
   Laden auf `1 - price` umgerechnet (siehe *Preiskonvention* weiter unten).
   `mean_price` bedeutet dadurch etwas anderes als vorher.
2. **Feature-Set 93 → 91.** `volume` und `liquidity` sind raus (Snapshot-Werte
   vom Export-Zeitpunkt, also Look-ahead), und `volume_concentration` teilt
   jetzt durch das Volumen im laengsten Rolling-Fenster statt durch das
   Lifetime-Volumen. Der Fenster-Nenner ist bewusst gewaehlt: er ist kausal
   *und* liefert in `run_pipeline.py` und `train_chunked.py` denselben Wert,
   weil die Kontext-Buckets eines Chunks ihn abdecken.

Ausserdem sind die ROI-Zahlen aus frueheren Laeufen in `results_log.jsonl`
nicht mit neuen vergleichbar: der Backtest rechnete mit einer
Even-Money-Wette statt mit der tatsaechlichen Preisbewegung und lag damit um
Groessenordnungen zu hoch. Am besten die Datei archivieren und neu anfangen.

Ablauf nach dem Update:

```bash
rm -f bucketed.parquet model.txt trades.db
mv results_log.jsonl results_log.old.jsonl 2>/dev/null || true
# dann Phase 1 wie unten beschrieben neu durchlaufen
```

---

## Uebersicht

```
markets.csv + orderFilled.csv
        |
        v
[0] sweep_horizon.py            Traegt sich die Haltedauer? (vor allem anderen)
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

## Phase 0 — Traegt sich die Haltedauer? (`sweep_horizon.py`)

Bevor du ein Modell trainierst: pruefe, ob im gewaehlten Label-Fenster
ueberhaupt genug Preisbewegung steckt, um die Handelskosten zu bezahlen.

```bash
python sweep_horizon.py \
    --trades data/trades.parquet --markets data/markets.parquet \
    --windows 3 6 12 24 48 96 288 --by-band
```

Ausgabe:

```
    Horizon      Labeled    MedianRet     p90Ret  MedianCost   Ret>Cost   MaxROI*
  ----------------------------------------------------------------------------
     15 min       46,574      0.121%    1.745%     10.01%     0.06%   +4.71%
     30 min       46,558      0.870%    2.933%      9.94%     0.36%   +7.26%
        1 h       46,520      1.315%    4.339%      9.91%     1.38%   +9.13%
        2 h       46,443      1.934%    6.312%      9.83%     4.12%  +10.50%
        4 h       46,292      2.843%    8.975%      9.73%    10.38%   +9.59%
        8 h       45,984      4.031%   12.618%      9.72%    22.11%   +9.98%
        1 d       44,808      6.894%   20.137%      9.58%    46.25%  +10.64%
```

| Spalte | Bedeutung |
|---|---|
| `MedianRet` | Rendite der **besseren** der beiden Seiten (YES oder NO) |
| `MedianCost` | Round-Trip-Kosten genau dieser Seite |
| `Ret>Cost` | Anteil der Buckets, in denen diese Rendite ihre Kosten schlaegt |
| `MaxROI*` | ROI bei **perfekter Voraussicht** — nur diese Buckets handeln und immer die richtige Seite treffen |

**`Ret>Cost` ist die Entscheidungszahl.** Sie ist eine Obergrenze: sie
unterstellt, dass du fuer jeden Bucket die bessere Seite kennst. Liegt sie
unter ~5 %, kann kein Modell profitabel werden — dann ist nicht das Modell
das Problem, sondern die Haltedauer. Kein Grund, an Features oder am Label
zu drehen; erst laengere Fenster probieren.

Beide Seiten werden dabei getrennt gerechnet. Eine YES- und eine NO-Share
desselben Marktes sind keine Spiegelbilder: bei P(YES) = 0.20 kostet die
YES-Share 0.20 und die NO-Share 0.80. Ein Preisrutsch auf 0.19 ist deshalb
−5,0 % auf der YES-Seite, aber +1,25 % auf der NO-Seite — bei 13 % gegen
3,2 % Kosten.

Mit `--by-band` kommt eine Aufschluesselung nach Tokenpreis dazu. Weil die
Kosten mit `1/Preis` skalieren, kann ein Horizont in der liquiden Mitte
funktionieren und an den Raendern hoffnungslos sein — oder umgekehrt.

Das Ergebnis haengt stark an `--spread-abs` (Default 0.010, der gemessene
Median). Mit 0.002 springt `Ret>Cost` bei 30 Minuten von 0,36 % auf 7,61 %.
Setz den Wert auf das, was du in deinen Maerkten wirklich siehst.

| Parameter | Default | Beschreibung |
|---|---|---|
| `--trades` / `--markets` | `data/*.parquet` | Datenpfade |
| `--windows` | `3 6 12 24 48 96 288` | Horizonte in Buckets |
| `--bucket-minutes` | `5` | Bucket-Groesse |
| `--spread-abs` | `0.010` | Absoluter Spread (siehe `CostConfig`) |
| `--fee-legs` | `2` | Als Taker ueberquerte Legs |
| `--by-band` | — | Zusaetzlich nach Tokenpreis aufschluesseln |
| `--keep-intermediates` | — | Zwischendateien behalten |

Der Sweep laeuft Bucketing und Luecken-Behandlung **einmal** und danach je
Horizont nur noch das Labeling — Features werden gar nicht gebaut, sie
spielen fuer diese Frage keine Rolle.

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
python tests/test_pipeline_e2e.py            # kompletter Durchlauf
python tests/test_data_loader.py             # Preis-Normalisierung
python tests/test_streaming_equivalence.py   # Batching == Einzelmarkt
```

Oder alle zusammen mit pytest:

```bash
pip install pytest && pytest tests/ -q
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
  Win rate   : 61.2%      <- Preis lief in die richtige Richtung
  Profitable : 18.4%      <- davon nach Kosten im Plus
  Mean return: +0.412%    <- pro Trade, vor Kosten
  Mean cost  : 9.8%       <- Spread + Gebuehr, skaliert mit 1/Preis
  ROI        : -9.6%      <- auf eingesetztes Kapital
  Sharpe     : 0.09       <- pro Trade, nicht annualisiert
```

Die Groessenordnung ist Absicht: eine 30-Minuten-Bewegung auf einem
Prediction Market liegt typisch bei ~1 %, nicht bei zweistelligen Prozenten.
Frueher rechnete der Backtest mit einer Even-Money-Wette (+100 % / -100 %)
und produzierte dadurch ROI-Zahlen, die um Groessenordnungen zu hoch waren.

**Wie du die Zahlen liest:**

| Kennzahl | Bedeutung | Gut | Warnsignal |
|---|---|---|---|
| ROC AUC (Test) | Trennschaerfe | > 0.55 | Train >> Test = Overfitting |
| Brier Score | Kalibrierung | < 0.23 | > 0.25 = schlechter als Muenzwurf |
| Win rate | Anteil Trades mit richtiger Preisrichtung | > 50% | — |
| Profitable | Anteil Trades die **nach Kosten** Geld machten | > 50% | << Win rate = Bewegungen zu klein |
| Mean return | Mittlere realisierte Rendite pro Trade | > Mean cost | < Mean cost = strukturell unprofitabel |
| Mean cost | Mittlere Round-Trip-Kosten pro Trade | — | steigt stark bei billigen Tokens |
| ROI (Backtest) | Gewinn / eingesetztes Kapital | > 0% | Negativ = Modell taugt nicht |
| Sharpe (pro Trade) | Rendite / Streuung, **nicht** annualisiert | > 0.1 | < 0 = inkonsistente Ergebnisse |

`ROI` ist die Rendite auf das **tatsaechlich eingesetzte Kapital**
(`Summe PnL / Summe Einsaetze`) und haengt damit weder von `initial_bankroll`
noch von der Zeilenreihenfolge ab — bei fester Positionsgroesse gilt exakt
`ROI = Mean return - fee_rate`. Daneben steht `Bankroll growth`: das ist die
Entwicklung der simulierten Bankroll und haengt sehr wohl von
`initial_bankroll` und `max_position_usd` ab. Sie ist Kontext fuer die
Equity-Kurve, **keine** Aussage ueber die Strategie. Reicht die Bankroll
nicht fuer alle Trades, wird das als `Bankroll exhausted` gemeldet; die
Trade-Statistiken darueber decken trotzdem jeden qualifizierten Trade ab.

**Win rate und Profitable auseinanderzuhalten ist der wichtigste Teil.**
Das Label sagt nur, dass der Preis sich um mindestens `min_price_move`
(0.001) in die richtige Richtung bewegt hat — nicht, um wie viel. Eine Share,
die von 0.500 auf 0.501 laeuft, ist ein `win` und zahlt 0,2 %. Die
Round-Trip-Kosten liegen in diesem Preisbereich bei rund 10 % — der Trade ist
also trotzdem ein Verlust. Eine Win rate von 98 % bei 1 % profitablen Trades
ist ein voellig normales Ergebnis und bedeutet: das Modell trifft die
Richtung, aber die Bewegungen tragen die Kosten nicht.

### Das Kostenmodell (`CostConfig`)

Handelskosten sind **kein** fester Prozentsatz. Sie werden in absoluten
Preiseinheiten notiert (ein Spread sind so und so viele Ticks), eine Position
ist aber nur `Tokenpreis` pro Share wert. Der relative Kostenanteil skaliert
damit mit `1 / Tokenpreis`:

```
cost(tp) = ( spread_abs + fee_legs * fee_rate * min(tp, 1-tp) ) / tp
```

| Tokenpreis | Round-Trip-Kosten (Default) |
|---|---|
| 0.50 | 10 % |
| 0.20 | 13 % |
| 0.10 | 18 % |
| 0.05 | 28 % |
| 0.02 | 58 % |
| 0.01 | 108 % |

Gemessen an 120 aktiven Orderbuechern (Median-Spread nach Preisniveau):

| min(p, 1-p) | Spread absolut | Spread / Preis |
|---|---|---|
| 0.35 – 0.50 | 0.0270 | 63 % |
| 0.20 – 0.35 | 0.0200 | 8 % |
| 0.10 – 0.20 | 0.0370 | 28 % |
| 0.05 – 0.10 | 0.0160 | 23 % |
| 0.02 – 0.05 | 0.0020 | 6 % |
| 0.00 – 0.02 | 0.0010 | 40 % |

**Kalibrierung:** `spread_abs` ist der einzige empirische Eingabewert und
streut stark (Quartile ueber die 120 Buecher: 0.001 / 0.010 / 0.039). Der
Default ist der Median. Setze ihn auf das, was du in *deinen* Maerkten
tatsaechlich siehst — die `1/tp`-Form gilt unabhaengig von der Konstante.

`fee_rate` ist Polymarkets eigener Satz (Gamma `feeSchedule.rate`, nur Taker):
Gebuehr pro Share = `rate * min(p, 1-p)`. `fee_legs` ist die Anzahl der Legs,
die du als Taker ueberquerst — 2 fuer rein und raus als Taker, 1 wenn du als
Maker aussteigst, 0 fuer eine reine Maker-Strategie.

Trades, deren Kosten `max_cost` (Default 1.0 = der ganze Positionswert)
uebersteigen, werden als nicht handelbar uebersprungen und unter `Skipped`
ausgewiesen — statt sie als Beinahe-Totalverlust zu verbuchen.

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
  Strategy         Dir     Cover     AUC   Brier  WinRate Profit%  Ret/Trd    Cost     ROI  Sharpe  Trades
  ---------------------------------------------------------------------------------------------------------
  baseline         follow  100.0%  0.5000  0.249   51.2%    1.1%  +0.412%   9.87%   -9.5%  -0.012   45823
  random           follow  100.0%  0.5001  0.250   51.2%    0.9%  +0.401%   9.91%   -9.5%  -0.015   18350
  momentum         follow   34.5%  0.5312  0.246   53.4%    2.7%  +0.688%   9.44%   -8.8%   0.071     892
  reversion        AGAINST  12.3%  0.5187  0.248   52.1%    1.8%  +0.551%  11.20%  -10.7%   0.031     123
  volume           follow    8.2%  0.5421  0.243   55.7%    3.4%  +0.914%   9.12%   -8.2%   0.112     234
  closing          follow    5.1%  0.5634  0.238   58.9%    5.9%  +1.203%   8.90%   -7.7%   0.141      67
  contrarian       AGAINST  18.7%  0.5023  0.251   51.3%    1.2%  +0.470%  14.51%  -14.0%   0.022     456
```

`Ret/Trd` ist die mittlere realisierte Preisbewegung pro Trade **vor** Kosten,
`Cost` sind die mittleren Round-Trip-Kosten. Liegt `Ret/Trd` unter `Cost`, ist
die Strategie strukturell unprofitabel, egal wie gut ihre AUC aussieht.

Dass `contrarian` die hoechsten Kosten hat, ist kein Zufall: die
AGAINST-Strategien halten das komplementaere Token, das haeufig das billigere
und damit relativ teurer zu handelnde ist.

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

### Welche API benutzt wird

Trades kommen von `https://data-api.polymarket.com/trades` — der oeffentlichen
Handelshistorie. **Nicht** von `clob.polymarket.com/trades`: das ist der
authentifizierte Endpunkt, der die *eigenen* Trades des API-Key-Inhabers
liefert und ohne Credentials mit `401 Unauthorized` antwortet.

Abgefragt wird pro **Markt** (`conditionId`), nicht pro Token. Die Token-ID auf
der Kommandozeile wird ueber die Gamma-API zum Markt aufgeloest. Das ist
zwingend: das Training aggregiert beide Tokens eines Marktes in einen Bucket,
also ist `yes_ratio` der YES-Anteil aller Fills. Faehrt man nur ein Token ab,
ist `yes_ratio` konstant und das Feature, auf dem das gesamte Label beruht,
ist tot.

### Schritt 3.2a: Direkter API-Abruf (aktive Maerkte)

Funktioniert bei Maerkten mit genuegend Handelsaktivitaet (~300+ Trades pro Stunde):

```bash
python live_bid.py \
    --token-id <TOKEN_ID_AUS_MARKETS_CSV> \
    --model model.txt \
    --threshold 0.6
```

**Wichtig:** `--bucket-minutes` und `--low-memory` muessen zum Training passen.
Sonst berechnet das Live-Skript andere Features als das Modell gelernt hat.
`live_bid.py` warnt, wenn ein Modell-Feature live nicht erzeugt werden kann.

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
  Loaded. Features: 92
[2/5] Resolving market...
  Market : Will X happen before Y?
  Closes : 2026-06-01 00:00:00
  Your token is the YES side
[3/5] Fetching trades from data-api (last 5.0 h)...
  API returned 847 trades in window
  Usable trades: 847
[4/5] Building features (training pipeline)...
  Buckets: 60 x 5 min (gap-filled, current bucket excluded)
  Scoring bucket: 2026-02-25 14:25:00
[5/5] Predicting...

=======================================================
  Scored bucket            : 2026-02-25 14:25:00
  Bucket yes_ratio         : 0.32
  Mean price (P(YES))      : 0.4180
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
├── sweep_horizon.py        [Phase 0]  Traegt sich die Haltedauer ueberhaupt?
│
├── evaluate_model.py       [Phase 2]  Modell auswerten ohne Neutraining
├── benchmark_strategies.py [Phase 2b] Klassische Strategien vs. Modell vergleichen
│
├── collect_trades.py       [Phase 3] Daemon: Live-Trades → SQLite
├── live_bid.py             [Phase 3] Live-Gebot pruefen mit model.txt
├── paper_trades.py         [Phase 3] Paper-Trading-Simulator mit Logging
│
├── requirements.txt        Python-Abhaengigkeiten
│
├── pipeline/
│   ├── data_loader.py      Daten laden (CSV/Parquet/SQLite), Preis → P(YES)
│   ├── aggregation.py      Trades → 5-Min-Buckets
│   ├── gap_handler.py      Luecken erkennen + behandeln
│   ├── features.py         92 Features berechnen
│   ├── labeling.py         Win/Loss Labels + realisierte Renditen
│   ├── splitter.py         Train/Val/Test aufteilen (zeitbasierter Purge)
│   ├── model.py            LightGBM Training + Inference
│   ├── monitor.py          Live-Dashboard
│   ├── evaluation.py       Metriken + Backtest
│   ├── polymarket_api.py   Gamma- + data-api-Zugriff (gemeinsam genutzt)
│   ├── live_features.py    Live-Features ueber die Trainings-Codepfade
│   └── results_logger.py   Historisches Ergebnis-Log
└── tests/
    ├── test_pipeline_e2e.py           E2E-Test mit Fake-Daten
    ├── test_data_loader.py            Preis-Normalisierung
    └── test_streaming_equivalence.py  Batching == Einzelmarkt
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
| `--context-buckets` | `60` | Vorlauf-Buckets an jeder Chunk-Grenze. **Muss mindestens so gross sein wie das laengste Feature-Fenster** (Default 48), sonst sind Rolling-/Lag-Features an jeder Grenze falsch. `train_chunked.py` warnt. |
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
| `--token-ids` | — | Eine oder mehrere Token-IDs (Leerzeichen getrennt). Jede wird zu ihrem Markt aufgeloest; gesammelt werden **beide** Tokens |
| `--db` | `trades.db` | Pfad zur SQLite-Datenbank |
| `--poll-interval` | `60` | Sekunden zwischen API-Abfragen |
| `--keep-days` | `7` | Tage Handelshistorie in der DB behalten |
| `--backfill-hours` | `12` | Wie viel Historie beim ersten Poll geholt wird |

Eine `trades.db` aus einer aelteren Version hat ein anderes Schema (pro Token,
`is_yes` immer 1). `collect_trades.py` und `live_bid.py` verweigern die Arbeit
damit, statt die unbrauchbaren Werte stillschweigend zu benutzen — die Datei
loeschen und neu sammeln.

### paper_trades.py

Simuliert Trades und prueft nach dem Label-Fenster, ob die Entscheidung
richtig war. Die PnL-Rechnung benutzt die tatsaechliche Preisbewegung; der
Ausgang wird gegen den Preis im faelligen Bucket geprueft, nicht gegen den
Preis zum Zeitpunkt des Checks.

```bash
# Terminal 1: Daten sammeln
python collect_trades.py --token-ids <TOKEN> --db trades.db

# Terminal 2: simuliert traden
python paper_trades.py --token-ids <TOKEN> --trades-db trades.db --model model.txt

# Terminal 3: Zwischenstand
python paper_trades.py --report --paper-db paper_trades.db
```

| Parameter | Default | Beschreibung |
|---|---|---|
| `--token-ids` | — | Token-IDs zum Tracken (ausser bei `--report`) |
| `--model` | `model.txt` | Pfad zum trainierten Modell |
| `--threshold` | `0.6` | P(win)-Schwellenwert fuer BID |
| `--stake` | `100.0` | Simulierter Einsatz pro Trade in USD |
| `--fee-rate` | `0.02` | Round-Trip-Kosten als Anteil vom Einsatz |
| `--bucket-minutes` | `5` | Muss zum Training passen |
| `--forward-window` | `6` | Label-Fenster in Buckets |
| `--low-memory` | — | Setzen, wenn mit `--low-memory` trainiert wurde |
| `--trades-db` | `trades.db` | DB von collect_trades.py |
| `--paper-db` | `paper_trades.db` | Log-DB fuer Entscheidungen |
| `--report` | — | Nur Report anzeigen, nicht traden |

### live_bid.py

| Parameter | Default | Beschreibung |
|---|---|---|
| `--token-id` | — | Polymarket Token-ID (aus markets.csv token1/token2) |
| `--model` | `model.txt` | Pfad zum trainierten Modell |
| `--threshold` | `0.6` | P(win)-Schwellenwert fuer BID (60%) |
| `--history-buckets` | `60` | Anzahl Buckets Historie (60 = 5 Stunden) |
| `--bucket-minutes` | `5` | **Muss zum Training passen** |
| `--low-memory` | — | Setzen, wenn mit `--low-memory` trainiert wurde |
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
zusammen. Das reduziert Rauschen und macht die Daten handhabbar. Ein Bucket
enthaelt die Trades **beider** Tokens eines Marktes.

### Preiskonvention: alles ist P(YES)

In `orderFilled.csv` ist `price` der Preis des jeweils gehandelten Tokens —
token2-Zeilen tragen also den NO-Preis. `pipeline/data_loader.py` rechnet die
NO-Seite auf `1 - price` um, bevor aggregiert wird.

Ohne das waere `mean_price` eine Mischung beider Seiten: ein Markt bei
YES = 0.80 mit ausgeglichenem Handelsmix landet bei 0.50, und der Wert bewegt
sich, sobald sich das YES/NO-Verhaeltnis verschiebt — auch wenn der Markt
stillsteht. Das Label ("Preis ist gestiegen") wuerde dann genau dieses Artefakt
messen. Nach der Normalisierung heisst jede Preisspalte dasselbe: P(YES).
`yes_ratio` traegt den Handelsmix weiterhin als eigenes Feature.

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

Der Gap wird in **Wandzeit** gemessen, nicht in Zeilen: Label-Fenster
(`forward_window_buckets x bucket_minutes`, Default 30 Min) plus
`SplitConfig.split_gap_minutes` (Default 60), also 90 Minuten.

Das ist wichtiger als es klingt. Die Zeilen der Feature-Matrix sind ueber
tausende Maerkte verschachtelt — ein Gap von "12 Zeilen" waren bei 5 Maerkten
noch 70 Minuten, bei 100 Maerkten nur noch 5 Minuten und bei realistischer
Marktzahl praktisch null. Die letzten Trainings-Labels reichten damit in den
Validierungszeitraum hinein und haben AUC und ROI out-of-sample geschoent.

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

`win` sagt nur, **ob** sich der Preis um mindestens 0.001 in die richtige
Richtung bewegt hat — nicht, um wie viel. Fuer den Backtest berechnet
`pipeline/labeling.py` deshalb zusaetzlich die tatsaechlich realisierte
Rendite der Position:

- YES-Seite: kaufen zu `p`, verkaufen zu `p'` → `(p' - p) / p`
- NO-Seite: kaufen zu `1 - p`, verkaufen zu `1 - p'` → `(p - p') / (1 - p)`

Die beiden Seiten haben unterschiedliche Nenner, weil eine NO-Share `1 - p`
kostet. Diese Spalten (`trade_return`, `trade_return_opp`) sind **keine**
Features — sie enthalten die Zukunft und sind vom Feature-Set ausgeschlossen.

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
