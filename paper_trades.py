#!/usr/bin/env python3
"""
paper_trader.py — Simuliertes Trading mit Logging

Läuft alle 5 Minuten, ruft das Modell ab, loggt BID-Entscheidungen
und prüft nach 30 Minuten ob die Entscheidung richtig war.

Usage:
    # Starten (läuft dauerhaft):
    python paper_trader.py \
        --token-ids <TOKEN_A> <TOKEN_B> \
        --db trades.db \
        --model model.txt \
        --threshold 0.6 \
        --stake 100.0

    # Report anzeigen (separates Terminal):
    python paper_trader.py --report --paper-db paper_trades.db

Schema (SQLite paper_trades.db):
    decisions   — jede BID-Entscheidung
    outcomes    — Ergebnis nach 30 Minuten
"""

import argparse
import signal
import sqlite3
import sys
import time
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

# ── Konstanten (müssen mit Training übereinstimmen) ───────────────────────────
BUCKET_MINUTES   = 5
FORWARD_BUCKETS  = 6          # 6 × 5min = 30 Minuten Label-Fenster
FEE_RATE         = 0.02
WHALE_THRESHOLD  = 1000.0
LAG_BUCKETS      = [1, 2, 3, 6, 12]
ROLLING_WINDOWS  = [6, 12, 24, 48]
MIN_BUCKETS      = 60         # Mindest-History für Features
CLOB_API         = "https://clob.polymarket.com"
GAMMA_API        = "https://gamma-api.polymarket.com"


# ── SQLite Setup ──────────────────────────────────────────────────────────────

def init_paper_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,          -- Entscheidungszeitpunkt (UTC)
            token_id      TEXT    NOT NULL,
            question      TEXT,
            direction     TEXT    NOT NULL,           -- YES oder NO
            p_win         REAL    NOT NULL,           -- Modell-Wahrscheinlichkeit
            threshold     REAL    NOT NULL,
            entry_price   REAL,                       -- Preis bei Entscheidung
            yes_ratio     REAL,
            stake         REAL    NOT NULL,
            bid           INTEGER NOT NULL,           -- 1=BID, 0=NO BID
            outcome_due   TEXT    NOT NULL,           -- wann prüfen (ts + 30min)
            outcome_checked INTEGER DEFAULT 0,        -- 0=ausstehend, 1=geprüft
            won           INTEGER,                    -- 1=gewonnen, 0=verloren, NULL=ausstehend
            pnl           REAL                        -- realisierter PnL
        )
    """)
    conn.commit()
    return conn


# ── Polymarket API ────────────────────────────────────────────────────────────

def _get(url, params=None):
    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests")
        sys.exit(1)
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_recent_trades(token_id: str, limit: int = 1000) -> list:
    data = _get(f"{CLOB_API}/trades", params={"token_id": token_id, "limit": limit})
    if isinstance(data, dict):
        return data.get("data", [])
    return data


def fetch_current_price(token_id: str) -> float | None:
    """Aktuellen Marktpreis vom CLOB holen."""
    try:
        data = _get(f"{CLOB_API}/book", params={"token_id": token_id})
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if bids and asks:
            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])
            return (best_bid + best_ask) / 2
        elif bids:
            return float(bids[0]["price"])
        elif asks:
            return float(asks[0]["price"])
    except Exception:
        pass
    return None


def fetch_market_info(token_id: str) -> dict:
    try:
        data = _get(f"{GAMMA_API}/markets", params={"clob_token_ids": token_id})
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            markets = data.get("markets", data.get("data", []))
            if markets:
                return markets[0]
    except Exception as e:
        print(f"  Warning: Markt-Metadata nicht verfügbar: {e}")
    return {}


# ── Feature Engineering (identisch zu live_bid.py) ───────────────────────────

def parse_and_aggregate(raw_trades: list, token_id: str) -> pl.DataFrame | None:
    rows = []
    for t in raw_trades:
        ts_raw = t.get("timestamp") or t.get("created_at") or t.get("time")
        if ts_raw is None:
            continue
        if isinstance(ts_raw, (int, float)):
            ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc).replace(tzinfo=None)
        else:
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                continue

        price_raw = t.get("price")
        size_raw  = t.get("size") or t.get("amount")
        if price_raw is None or size_raw is None:
            continue

        price        = float(price_raw)
        token_amount = float(size_raw)
        usd_amount   = price * token_amount

        outcome = str(t.get("outcome", "")).upper()
        if outcome in ("YES", "1"):
            is_yes = True
        elif outcome in ("NO", "0"):
            is_yes = False
        else:
            is_yes = str(t.get("token_id", "")) == str(token_id)

        rows.append({"timestamp": ts, "price": price,
                     "usd_amount": usd_amount, "token_amount": token_amount,
                     "is_yes": is_yes})

    if not rows:
        return None

    trades = (pl.DataFrame(rows)
              .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
              .sort("timestamp"))

    bucket_dur = f"{BUCKET_MINUTES}m"
    bucketed = (
        trades
        .with_columns(pl.col("timestamp").dt.truncate(bucket_dur).alias("bucket_time"))
        .group_by("bucket_time")
        .agg([
            pl.len().alias("trade_count"),
            pl.col("usd_amount").sum().alias("total_usd"),
            pl.col("token_amount").sum().alias("total_tokens"),
            pl.col("price").first().alias("open_price"),
            pl.col("price").last().alias("close_price"),
            pl.col("price").max().alias("high_price"),
            pl.col("price").min().alias("low_price"),
            pl.col("price").mean().alias("mean_price"),
            (pl.col("price") * pl.col("usd_amount")).sum().alias("_pxu"),
            pl.col("is_yes").mean().alias("yes_ratio"),
            (pl.col("usd_amount") > WHALE_THRESHOLD).sum().alias("whale_count"),
            pl.col("usd_amount").filter(pl.col("usd_amount") > WHALE_THRESHOLD)
              .sum().alias("whale_usd"),
        ])
        .sort("bucket_time")
        .with_columns([
            (pl.col("_pxu") / pl.col("total_usd")).fill_nan(None).alias("vwap"),
            (pl.col("close_price") - pl.col("open_price")).alias("momentum"),
        ])
        .drop("_pxu")
    )
    return bucketed


def build_features_live(bucketed: pl.DataFrame, market: dict, now: datetime) -> pl.DataFrame:
    df = bucketed.clone()

    # Lag Features
    lag_cols = ["mean_price", "total_usd", "trade_count", "momentum", "yes_ratio"]
    exprs = []
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in LAG_BUCKETS:
            exprs.append(pl.col(col).shift(lag).alias(f"{col}_lag{lag}"))
    if exprs:
        df = df.with_columns(exprs)

    for lag in LAG_BUCKETS:
        lc = f"mean_price_lag{lag}"
        if lc in df.columns:
            df = df.with_columns([
                (pl.col("mean_price") - pl.col(lc)).alias(f"price_change_lag{lag}"),
                ((pl.col("mean_price") - pl.col(lc)) / pl.col(lc)).fill_nan(None)
                  .alias(f"price_return_lag{lag}"),
            ])

    # Rolling Features
    for w in ROLLING_WINDOWS:
        df = df.with_columns([
            pl.col("mean_price").rolling_mean(w).alias(f"price_ma{w}"),
            pl.col("mean_price").rolling_std(w).alias(f"price_std{w}"),
            pl.col("total_usd").rolling_sum(w).alias(f"volume_sum{w}"),
            pl.col("trade_count").rolling_mean(w).alias(f"trade_count_ma{w}"),
            pl.col("momentum").rolling_mean(w).alias(f"momentum_ma{w}"),
        ])
    for w in ROLLING_WINDOWS:
        ma = f"price_ma{w}"
        df = df.with_columns(
            ((pl.col("mean_price") - pl.col(ma)) / pl.col(ma)).fill_nan(None)
              .alias(f"price_vs_ma{w}")
        )

    # Market Features
    yes_price  = None
    no_price   = None
    op = market.get("outcomePrices")
    if op:
        try:
            if isinstance(op, str):
                import json
                op = json.loads(op)
            yes_price = float(op[0])
            no_price  = float(op[1]) if len(op) > 1 else None
        except Exception:
            pass

    volume    = _sf(market.get("volume") or market.get("volumeNum"))
    liquidity = _sf(market.get("liquidity") or market.get("liquidityNum"))
    end_str   = market.get("endDate") or market.get("end_date_iso")

    exprs = []
    if yes_price  is not None: exprs.append(pl.lit(yes_price).alias("yes_price"))
    if no_price   is not None: exprs.append(pl.lit(no_price).alias("no_price"))
    if volume     is not None: exprs.append(pl.lit(volume).alias("volume"))
    if liquidity  is not None: exprs.append(pl.lit(liquidity).alias("liquidity"))
    if exprs:
        df = df.with_columns(exprs)

    if "yes_price" in df.columns and "no_price" in df.columns:
        df = df.with_columns(
            (pl.col("yes_price") - pl.col("no_price")).abs().alias("market_spread")
        )
        p = pl.col("yes_price").clip(1e-9, 1 - 1e-9)
        df = df.with_columns(
            (-(p * p.log(base=2) + (1-p)*(1-p).log(base=2))).alias("market_entropy")
        )

    if end_str:
        try:
            close_dt = datetime.fromisoformat(str(end_str).replace("Z", "+00:00")).replace(tzinfo=None)
            df = df.with_columns(
                pl.lit((close_dt - now).total_seconds() / 86400).alias("days_to_close")
            )
        except Exception:
            pass

    # Cross Features
    if "yes_price" in df.columns:
        df = df.with_columns(
            (pl.col("mean_price") - pl.col("yes_price")).alias("entry_vs_market")
        )
    if "liquidity" in df.columns:
        df = df.with_columns(
            (pl.col("total_usd") / pl.col("liquidity")).fill_nan(None)
              .alias("trade_size_vs_liquidity")
        )
    if "volume" in df.columns:
        df = df.with_columns(
            (pl.col("total_usd") / pl.col("volume")).fill_nan(None)
              .alias("volume_concentration")
        )
    if "whale_count" in df.columns:
        df = df.with_columns(
            (pl.col("whale_count").cast(pl.Float64) / pl.col("trade_count").cast(pl.Float64))
              .fill_nan(0.0).alias("whale_ratio")
        )
        df = df.with_columns(
            (pl.col("whale_ratio") * pl.col("momentum")).alias("whale_momentum")
        )
    if "days_to_close" in df.columns:
        df = df.with_columns(
            (pl.col("momentum") / (pl.col("days_to_close") + 1)).fill_nan(None)
              .alias("momentum_per_day")
        )

    # Zeit Features
    df = df.with_columns([
        pl.col("bucket_time").dt.hour().alias("hour"),
        pl.col("bucket_time").dt.weekday().alias("day_of_week"),
    ])
    df = df.with_columns([
        (pl.col("hour").cast(pl.Float64) * 2 * math.pi / 24).sin().alias("hour_sin"),
        (pl.col("hour").cast(pl.Float64) * 2 * math.pi / 24).cos().alias("hour_cos"),
        (pl.col("day_of_week").cast(pl.Float64) * 2 * math.pi / 7).sin().alias("dow_sin"),
        (pl.col("day_of_week").cast(pl.Float64) * 2 * math.pi / 7).cos().alias("dow_cos"),
        (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("is_weekend"),
    ])

    return df


def _sf(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_feature_row(df: pl.DataFrame, booster: lgb.Booster) -> np.ndarray:
    model_features = booster.feature_name()
    last = df.tail(1)
    row = []
    for feat in model_features:
        if feat in last.columns:
            v = last[feat][0]
            row.append(float(v) if v is not None else float("nan"))
        else:
            row.append(float("nan"))
    return np.array([row], dtype=np.float32)


# ── Outcome-Prüfung ───────────────────────────────────────────────────────────

def check_pending_outcomes(paper_conn: sqlite3.Connection, trades_db: str | None) -> None:
    """Prüft alle offenen Entscheidungen deren outcome_due in der Vergangenheit liegt."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pending = paper_conn.execute("""
        SELECT id, token_id, direction, entry_price, stake, outcome_due
        FROM decisions
        WHERE bid=1 AND outcome_checked=0 AND outcome_due <= ?
    """, (now_str,)).fetchall()

    if not pending:
        return

    print(f"\n[Outcome-Check] {len(pending)} offene Entscheidung(en) prüfen...")

    for row_id, token_id, direction, entry_price, stake, outcome_due in pending:
        current_price = None

        # Aus SQLite DB lesen falls verfügbar
        if trades_db and Path(trades_db).exists():
            try:
                tc = sqlite3.connect(trades_db)
                r = tc.execute("""
                    SELECT price FROM trades
                    WHERE token_id=? ORDER BY timestamp DESC LIMIT 1
                """, (token_id,)).fetchone()
                tc.close()
                if r:
                    current_price = r[0]
            except Exception:
                pass

        # Fallback: live API
        if current_price is None:
            current_price = fetch_current_price(token_id)

        if current_price is None or entry_price is None:
            print(f"  ID {row_id}: Kein aktueller Preis verfügbar — später erneut prüfen")
            continue

        # Gewonnen?
        min_move = 0.001
        if direction == "YES":
            won = 1 if current_price > entry_price + min_move else 0
        else:
            won = 1 if current_price < entry_price - min_move else 0

        pnl = stake * (1.0 - FEE_RATE) if won else -stake

        paper_conn.execute("""
            UPDATE decisions
            SET outcome_checked=1, won=?, pnl=?
            WHERE id=?
        """, (won, pnl, row_id))
        paper_conn.commit()

        symbol = "✅ GEWONNEN" if won else "❌ VERLOREN"
        print(f"  ID {row_id}: {direction} | Entry {entry_price:.3f} → Jetzt {current_price:.3f} | {symbol} | PnL: ${pnl:+.2f}")


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(paper_db: str) -> None:
    conn = sqlite3.connect(paper_db)

    total_decisions = conn.execute("SELECT COUNT(*) FROM decisions WHERE bid=1").fetchone()[0]
    checked         = conn.execute("SELECT COUNT(*) FROM decisions WHERE bid=1 AND outcome_checked=1").fetchone()[0]
    pending         = total_decisions - checked
    wins            = conn.execute("SELECT COUNT(*) FROM decisions WHERE won=1").fetchone()[0]
    losses          = conn.execute("SELECT COUNT(*) FROM decisions WHERE won=0").fetchone()[0]
    total_pnl       = conn.execute("SELECT COALESCE(SUM(pnl),0) FROM decisions WHERE outcome_checked=1").fetchone()[0]
    skipped         = conn.execute("SELECT COUNT(*) FROM decisions WHERE bid=0").fetchone()[0]

    win_rate = wins / checked if checked > 0 else 0

    print("\n" + "="*60)
    print("  PAPER TRADING REPORT")
    print("="*60)
    print(f"  Entscheidungen gesamt : {total_decisions + skipped}")
    print(f"  Davon BID             : {total_decisions}")
    print(f"  Davon NO BID          : {skipped}")
    print(f"  Bereits geprüft       : {checked}")
    print(f"  Noch ausstehend       : {pending}")
    print(f"  Gewonnen              : {wins}")
    print(f"  Verloren              : {losses}")
    print(f"  Win Rate              : {win_rate:.1%}")
    print(f"  Gesamt PnL            : ${total_pnl:+,.2f}")
    if checked > 0:
        print(f"  Ø PnL pro Trade       : ${total_pnl/checked:+.2f}")

    # Letzte 10 Trades
    rows = conn.execute("""
        SELECT ts, question, direction, p_win, entry_price, won, pnl
        FROM decisions
        WHERE bid=1 AND outcome_checked=1
        ORDER BY ts DESC LIMIT 10
    """).fetchall()

    if rows:
        print(f"\n  Letzte {len(rows)} abgeschlossene Trades:")
        print(f"  {'Zeitpunkt':<22} {'Markt':<30} {'Dir':<4} {'P(win)':<8} {'Preis':<7} {'Ergebnis'}")
        print("  " + "-"*90)
        for ts, question, direction, p_win, entry_price, won, pnl in rows:
            q        = (question or "?")[:28]
            ergebnis = f"{'✅':>2} ${pnl:+.2f}" if won else f"{'❌':>2} ${pnl:+.2f}"
            print(f"  {ts:<22} {q:<30} {direction:<4} {p_win:.1%}   {entry_price or 0:.3f}  {ergebnis}")

    # Pro Token aufschlüsseln
    token_rows = conn.execute("""
        SELECT token_id, question,
               COUNT(*) as trades,
               SUM(won) as wins,
               SUM(pnl) as pnl
        FROM decisions
        WHERE bid=1 AND outcome_checked=1
        GROUP BY token_id
    """).fetchall()

    if token_rows:
        print(f"\n  Performance pro Markt:")
        print(f"  {'Markt':<40} {'Trades':>7} {'Win%':>6} {'PnL':>10}")
        print("  " + "-"*70)
        for token_id, question, trades, wins, pnl in token_rows:
            q  = (question or token_id[:20])[:38]
            wr = (wins or 0) / trades if trades > 0 else 0
            print(f"  {q:<40} {trades:>7} {wr:>6.1%} ${pnl or 0:>9.2f}")

    print("="*60)
    conn.close()


# ── Haupt-Trading-Loop ────────────────────────────────────────────────────────

def trading_loop(args, booster: lgb.Booster, paper_conn: sqlite3.Connection) -> None:
    print(f"\nPaper Trader gestartet")
    print(f"  Token(s)  : {len(args.token_ids)}")
    print(f"  Threshold : {args.threshold:.0%}")
    print(f"  Einsatz   : ${args.stake:.2f} pro Trade")
    print(f"  Log DB    : {args.paper_db}")
    print(f"  Interval  : alle {BUCKET_MINUTES} Minuten")
    print("  Ctrl+C zum Beenden\n")

    running = True
    def _stop(sig, frame):
        nonlocal running
        print("\nPaper Trader wird beendet...")
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running:
        now     = datetime.now(timezone.utc).replace(tzinfo=None)
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n[{now.strftime('%H:%M:%S UTC')}] Analysiere {len(args.token_ids)} Markt/Märkte...")

        # Offene Outcomes prüfen
        check_pending_outcomes(paper_conn, args.trades_db)

        for token_id in args.token_ids:
            print(f"\n  Token ...{token_id[-12:]}")

            # Trades holen — zuerst aus DB, dann API
            raw_trades = None
            if args.trades_db and Path(args.trades_db).exists():
                try:
                    tc     = sqlite3.connect(args.trades_db)
                    cutoff = int((now - timedelta(minutes=MIN_BUCKETS * BUCKET_MINUTES))
                                 .replace(tzinfo=timezone.utc).timestamp())
                    db_rows = tc.execute("""
                        SELECT timestamp, price, usd_amount, token_amount, is_yes
                        FROM trades WHERE token_id=? AND timestamp>=?
                        ORDER BY timestamp ASC
                    """, (token_id, cutoff)).fetchall()
                    tc.close()
                    if db_rows:
                        raw_trades = [{"timestamp": r[0], "price": r[1],
                                       "usd_amount": r[2], "token_amount": r[3],
                                       "is_yes": bool(r[4])} for r in db_rows]
                        print(f"    DB: {len(raw_trades)} Trades geladen")
                except Exception as e:
                    print(f"    DB Fehler: {e}")

            if not raw_trades:
                raw_trades = fetch_recent_trades(token_id, limit=1000)
                print(f"    API: {len(raw_trades)} Trades geladen")

            if not raw_trades:
                print("    Keine Trades verfügbar — überspringe")
                continue

            # Aggregieren + Features
            bucketed = parse_and_aggregate(raw_trades, token_id)
            if bucketed is None or len(bucketed) < 2:
                print("    Zu wenig Daten für Buckets")
                continue
            print(f"    Buckets: {len(bucketed)} × {BUCKET_MINUTES} min")

            if len(bucketed) < 12:
                print(f"    Warnung: Nur {len(bucketed)} Buckets — Features unvollständig")

            market      = fetch_market_info(token_id)
            question    = market.get("question", market.get("title", "?"))
            featured    = build_features_live(bucketed, market, now)

            # Prediction
            X           = get_feature_row(featured, booster)
            p_win       = float(booster.predict(X)[0])
            last        = featured.tail(1)
            yes_ratio   = float(last["yes_ratio"][0]) if "yes_ratio" in last.columns else 0.5
            entry_price = float(last["mean_price"][0]) if "mean_price" in last.columns else None
            direction   = "YES" if yes_ratio > 0.5 else "NO"
            bid         = 1 if p_win >= args.threshold else 0
            outcome_due = (now + timedelta(minutes=BUCKET_MINUTES * FORWARD_BUCKETS)
                           ).strftime("%Y-%m-%dT%H:%M:%SZ")

            # In DB speichern
            paper_conn.execute("""
                INSERT INTO decisions
                (ts, token_id, question, direction, p_win, threshold,
                 entry_price, yes_ratio, stake, bid, outcome_due)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (now_str, token_id, question[:100], direction, p_win,
                  args.threshold, entry_price, yes_ratio, args.stake, bid, outcome_due))
            paper_conn.commit()

            if bid:
                print(f"    *** BID {direction} *** P(win)={p_win:.1%} | Preis={entry_price:.3f} | Einsatz=${args.stake}")
            else:
                print(f"    NO BID — P(win)={p_win:.1%} < {args.threshold:.0%}")

        # Warte bis zum nächsten Bucket
        print(f"\n  Nächste Analyse in {BUCKET_MINUTES} Minuten...")
        for _ in range(BUCKET_MINUTES * 60):
            if not running:
                break
            time.sleep(1)

    # Abschluss-Report
    print_report(args.paper_db)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Simulator für Neuropoly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--token-ids",  nargs="+",
                        help="Polymarket Token-IDs zum Tracken")
    parser.add_argument("--model",      default="model.txt")
    parser.add_argument("--threshold",  type=float, default=0.6)
    parser.add_argument("--stake",      type=float, default=100.0,
                        help="Simulierter Einsatz pro Trade in USD (default: 100)")
    parser.add_argument("--trades-db",  default="trades.db",
                        help="SQLite DB von collect_trades.py (optional)")
    parser.add_argument("--paper-db",   default="paper_trades.db",
                        help="SQLite DB für Paper-Trading Log (default: paper_trades.db)")
    parser.add_argument("--report",     action="store_true",
                        help="Nur Report anzeigen, nicht traden")
    args = parser.parse_args()

    if args.report:
        print_report(args.paper_db)
        return

    if not args.token_ids:
        parser.error("--token-ids ist erforderlich (außer bei --report)")

    # Modell laden
    print(f"Lade Modell {args.model}...")
    if not Path(args.model).exists():
        print(f"ERROR: {args.model} nicht gefunden")
        sys.exit(1)
    booster = lgb.Booster(model_file=args.model)
    print(f"  {len(booster.feature_name())} Features geladen")

    # Paper DB initialisieren
    paper_conn = init_paper_db(args.paper_db)

    # Trading Loop
    trading_loop(args, booster, paper_conn)
    paper_conn.close()


if __name__ == "__main__":
    main()
