#!/usr/bin/env python3
"""
paper_trades.py — Simuliertes Trading mit Logging

Laeuft alle N Minuten, fragt das Modell, loggt BID-Entscheidungen und prueft
nach dem Label-Fenster (Default 30 Min) ob die Entscheidung richtig war.

Features werden ueber pipeline/live_features.py gebaut — also mit demselben
Code, der auch die Trainingsdaten erzeugt hat.  Die PnL-Rechnung benutzt die
tatsaechliche Preisbewegung, nicht eine Even-Money-Wette: eine Share, die von
0.500 auf 0.501 laeuft, zahlt 0,2 % — nicht 100 %.

Usage:
    # Starten (laeuft dauerhaft):
    python paper_trades.py \
        --token-ids <TOKEN_A> <TOKEN_B> \
        --trades-db trades.db \
        --model model.txt \
        --threshold 0.6 \
        --stake 100.0

    # Report anzeigen (separates Terminal):
    python paper_trades.py --report --paper-db paper_trades.db

Schema (SQLite paper_trades.db): Tabelle `decisions`
"""

import argparse
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from live_bid import fetch_trades_from_db
from pipeline.live_features import (
    align_to_model,
    bucket_age,
    build_live_features,
    explain_missing,
    history_window,
    is_stale,
)
from pipeline.polymarket_api import (
    MarketInfo,
    PolymarketAPIError,
    fetch_market,
    fetch_trades,
    normalize_trades,
)

MIN_PRICE_MOVE = 0.001   # muss zu LabelConfig.min_price_move passen


# ── SQLite Setup ──────────────────────────────────────────────────────────────

def init_paper_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            TEXT    NOT NULL,   -- Entscheidungszeitpunkt (UTC)
            token_id      TEXT    NOT NULL,
            condition_id  TEXT    NOT NULL,
            question      TEXT,
            direction     TEXT    NOT NULL,   -- YES oder NO
            p_win         REAL    NOT NULL,   -- Modell-Wahrscheinlichkeit
            threshold     REAL    NOT NULL,
            bucket_time   TEXT,               -- gescorter (abgeschlossener) Bucket
            entry_price   REAL,               -- P(YES) bei Entscheidung
            yes_ratio     REAL,
            stake         REAL    NOT NULL,
            fee_rate      REAL    NOT NULL,
            bid           INTEGER NOT NULL,   -- 1=BID, 0=NO BID
            outcome_due   TEXT    NOT NULL,   -- wann pruefen
            outcome_checked INTEGER DEFAULT 0,
            exit_price    REAL,               -- P(YES) am Faelligkeitszeitpunkt
            trade_return  REAL,               -- realisierte Rendite der Position
            won           INTEGER,            -- 1=gewonnen, 0=verloren
            pnl           REAL                -- realisierter PnL nach Kosten
        )
    """)
    conn.commit()
    return conn


# ── Outcome-Pruefung ──────────────────────────────────────────────────────────

def _price_at(
    trades: pl.DataFrame,
    at: datetime,
    bucket_minutes: int,
) -> float | None:
    """
    Mittlerer P(YES)-Preis des Buckets, der `at` enthaelt.

    Bewusst NICHT "der aktuelle Preis": wenn der Check verspaetet laeuft
    (Prozess schlief, API war weg), wuerde der aktuelle Preis ein voellig
    anderes Zeitfenster messen als das Label-Fenster, auf das die
    Entscheidung sich bezieht.
    """
    if trades.is_empty():
        return None
    start = at.replace(
        minute=(at.minute // bucket_minutes) * bucket_minutes,
        second=0, microsecond=0,
    )
    end = start + timedelta(minutes=bucket_minutes)
    window = trades.filter(
        (pl.col("timestamp") >= start) & (pl.col("timestamp") < end)
    )
    if window.is_empty():
        return None
    return float(window["price"].mean())


def check_pending_outcomes(
    paper_conn: sqlite3.Connection,
    markets: dict[str, MarketInfo],
    args,
    cfg: PipelineConfig,
) -> None:
    """Prueft alle offenen Entscheidungen deren outcome_due erreicht ist."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pending = paper_conn.execute("""
        SELECT id, condition_id, direction, entry_price, stake, fee_rate, outcome_due
        FROM decisions
        WHERE bid=1 AND outcome_checked=0 AND outcome_due <= ?
    """, (now_str,)).fetchall()

    if not pending:
        return

    print(f"\n[Outcome-Check] {len(pending)} offene Entscheidung(en) pruefen...")

    for row_id, cond_id, direction, entry, stake, fee_rate, due_str in pending:
        due = datetime.strptime(due_str, "%Y-%m-%dT%H:%M:%SZ")
        market = markets.get(cond_id)

        # Handelsdaten um den Faelligkeitszeitpunkt herum holen
        window_start = int(
            (due - timedelta(minutes=cfg.bucket.bucket_minutes * 2))
            .replace(tzinfo=timezone.utc).timestamp()
        )
        trades = _load_trades(cond_id, market, window_start, args)
        exit_price = _price_at(trades, due, cfg.bucket.bucket_minutes)

        if exit_price is None or entry is None:
            print(f"  ID {row_id}: kein Preis im Faelligkeits-Bucket "
                  f"({due_str}) — spaeter erneut pruefen")
            continue

        # Realisierte Rendite der tatsaechlich gehaltenen Seite.
        # Identisch zu pipeline/labeling.py:trade_return.
        entry_c = min(max(entry, 1e-6), 1.0 - 1e-6)
        if direction == "YES":
            ret = (exit_price - entry_c) / entry_c
            won = 1 if exit_price > entry_c + MIN_PRICE_MOVE else 0
        else:
            ret = (entry_c - exit_price) / (1.0 - entry_c)
            won = 1 if exit_price < entry_c - MIN_PRICE_MOVE else 0

        pnl = stake * (ret - fee_rate)

        paper_conn.execute("""
            UPDATE decisions
            SET outcome_checked=1, exit_price=?, trade_return=?, won=?, pnl=?
            WHERE id=?
        """, (exit_price, ret, won, pnl, row_id))
        paper_conn.commit()

        symbol = "GEWONNEN" if won else "VERLOREN"
        print(f"  ID {row_id}: {direction} | Entry {entry:.4f} -> Exit {exit_price:.4f} "
              f"| {symbol} | Rendite {ret:+.3%} | PnL ${pnl:+.2f}")


# ── Daten laden ───────────────────────────────────────────────────────────────

def _load_trades(
    condition_id: str,
    market: MarketInfo | None,
    since_unix: int,
    args,
) -> pl.DataFrame:
    """Trades aus der SQLite-DB, sonst von der data-api."""
    if args.trades_db and Path(args.trades_db).exists():
        try:
            df = fetch_trades_from_db(args.trades_db, condition_id, since_unix)
            if not df.is_empty():
                return df
        except (RuntimeError, sqlite3.Error) as exc:
            print(f"    DB-Fehler: {exc}")

    if market is None:
        return pl.DataFrame()
    try:
        raw = fetch_trades(condition_id, since_unix=since_unix, max_trades=5000)
    except PolymarketAPIError as exc:
        print(f"    API-Fehler: {exc}")
        return pl.DataFrame()
    return normalize_trades(raw, market)


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(paper_db: str) -> None:
    conn = sqlite3.connect(paper_db)

    total_decisions = conn.execute("SELECT COUNT(*) FROM decisions WHERE bid=1").fetchone()[0]
    checked = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE bid=1 AND outcome_checked=1").fetchone()[0]
    pending = total_decisions - checked
    wins = conn.execute("SELECT COUNT(*) FROM decisions WHERE won=1").fetchone()[0]
    losses = conn.execute("SELECT COUNT(*) FROM decisions WHERE won=0").fetchone()[0]
    profitable = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE outcome_checked=1 AND pnl > 0").fetchone()[0]
    total_pnl = conn.execute(
        "SELECT COALESCE(SUM(pnl),0) FROM decisions WHERE outcome_checked=1").fetchone()[0]
    mean_ret = conn.execute(
        "SELECT AVG(trade_return) FROM decisions WHERE outcome_checked=1").fetchone()[0]
    total_stake = conn.execute(
        "SELECT COALESCE(SUM(stake),0) FROM decisions WHERE outcome_checked=1").fetchone()[0]
    skipped = conn.execute("SELECT COUNT(*) FROM decisions WHERE bid=0").fetchone()[0]

    win_rate = wins / checked if checked > 0 else 0
    profit_rate = profitable / checked if checked > 0 else 0
    roi = total_pnl / total_stake if total_stake > 0 else 0

    print("\n" + "=" * 62)
    print("  PAPER TRADING REPORT")
    print("=" * 62)
    print(f"  Entscheidungen gesamt : {total_decisions + skipped}")
    print(f"  Davon BID             : {total_decisions}")
    print(f"  Davon NO BID          : {skipped}")
    print(f"  Bereits geprueft      : {checked}")
    print(f"  Noch ausstehend       : {pending}")
    print(f"  Richtige Richtung     : {wins}  ({win_rate:.1%})")
    print(f"  Davon profitabel      : {profitable}  ({profit_rate:.1%})   <- nach Kosten")
    print(f"  Verloren              : {losses}")
    print(f"  O Rendite pro Trade   : {(mean_ret or 0):+.3%}")
    print(f"  Gesamt PnL            : ${total_pnl:+,.2f}")
    print(f"  ROI auf Einsatz       : {roi:+.2%}")
    if checked > 0:
        print(f"  O PnL pro Trade       : ${total_pnl/checked:+.2f}")

    rows = conn.execute("""
        SELECT ts, question, direction, p_win, entry_price, exit_price,
               trade_return, won, pnl
        FROM decisions
        WHERE bid=1 AND outcome_checked=1
        ORDER BY ts DESC LIMIT 10
    """).fetchall()

    if rows:
        print(f"\n  Letzte {len(rows)} abgeschlossene Trades:")
        print(f"  {'Zeitpunkt':<21} {'Markt':<26} {'Dir':<4} {'P(win)':>7} "
              f"{'Entry':>7} {'Exit':>7} {'Rendite':>9} {'PnL':>9}")
        print("  " + "-" * 96)
        for ts, q, direction, p_win, entry, exit_p, ret, won, pnl in rows:
            mark = "+" if won else "-"
            print(f"  {ts:<21} {(q or '?')[:24]:<26} {direction:<4} {p_win:>6.1%} "
                  f"{entry or 0:>7.4f} {exit_p or 0:>7.4f} {(ret or 0):>+8.2%} "
                  f"{mark}${abs(pnl or 0):>7.2f}")

    token_rows = conn.execute("""
        SELECT condition_id, question, COUNT(*) AS trades,
               SUM(won) AS wins, SUM(pnl) AS pnl, AVG(trade_return) AS ret
        FROM decisions
        WHERE bid=1 AND outcome_checked=1
        GROUP BY condition_id
    """).fetchall()

    if token_rows:
        print(f"\n  Performance pro Markt:")
        print(f"  {'Markt':<40} {'Trades':>7} {'Win%':>6} {'O Rend':>9} {'PnL':>10}")
        print("  " + "-" * 78)
        for cond_id, q, trades, wins_m, pnl, ret in token_rows:
            name = (q or cond_id[:20])[:38]
            wr = (wins_m or 0) / trades if trades else 0
            print(f"  {name:<40} {trades:>7} {wr:>6.1%} {(ret or 0):>+8.2%} "
                  f"${pnl or 0:>9.2f}")

    print("=" * 62)
    conn.close()


# ── Haupt-Trading-Loop ────────────────────────────────────────────────────────

def trading_loop(args, cfg: PipelineConfig, booster: lgb.Booster,
                 markets: dict[str, MarketInfo], paper_conn: sqlite3.Connection) -> None:
    horizon = cfg.label.forward_window_buckets * cfg.bucket.bucket_minutes

    print(f"\nPaper Trader gestartet")
    print(f"  Maerkte    : {len(markets)}")
    print(f"  Threshold  : {args.threshold:.0%}")
    print(f"  Einsatz    : ${args.stake:.2f} pro Trade")
    print(f"  Kosten     : {args.fee_rate:.2%} pro Trade (Gebuehr + Spread)")
    print(f"  Label-Fenster: {horizon} min")
    print(f"  Log DB     : {args.paper_db}")
    print(f"  Interval   : alle {cfg.bucket.bucket_minutes} Minuten")
    print("  Ctrl+C zum Beenden\n")

    running = True

    def _stop(sig, frame):
        nonlocal running
        print("\nPaper Trader wird beendet...")
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    model_features = booster.feature_name()
    lookback = history_window(cfg, args.history_buckets)

    while running:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"\n[{now.strftime('%H:%M:%S UTC')}] Analysiere {len(markets)} Markt/Maerkte...")

        check_pending_outcomes(paper_conn, markets, args, cfg)

        for cond_id, market in markets.items():
            print(f"\n  {market.question[:52]}")

            since = int((now - lookback).replace(tzinfo=timezone.utc).timestamp())
            trades = _load_trades(cond_id, market, since, args)
            if trades.is_empty():
                print("    Keine Trades verfuegbar — ueberspringe")
                continue
            print(f"    Trades: {trades.height}")

            featured = build_live_features(trades, market, cfg, now=now)
            if featured.is_empty():
                print("    Kein abgeschlossener Bucket — ueberspringe")
                continue
            print(f"    Buckets: {featured.height} x {cfg.bucket.bucket_minutes} min")

            # A decision taken on a bucket that closed hours ago is not a
            # decision this simulator should score itself on — it would make
            # the paper-trading statistics look like live performance when
            # they are not.  Skip instead of logging.
            age = bucket_age(featured["bucket_time"][-1], now,
                             cfg.bucket.bucket_minutes)
            if is_stale(age, cfg.bucket.bucket_minutes):
                print(f"    Uebersprungen: letzter Bucket ist "
                      f"{age.total_seconds()/60:.0f} min alt (keine "
                      f"aktuelle Grundlage)")
                continue

            X, missing = align_to_model(featured, model_features)
            _, unexpected = explain_missing(missing)
            if unexpected:
                print(f"    WARNUNG: {len(unexpected)} Modell-Feature(s) fehlen live "
                      f"— passen --bucket-minutes / --low-memory zum Training?")
            p_win = float(booster.predict(X)[0])

            last = featured.tail(1)
            yes_ratio = float(last["yes_ratio"][0]) if last["yes_ratio"][0] is not None else 0.5
            entry_price = last["mean_price"][0]
            entry_price = float(entry_price) if entry_price is not None else None
            bucket_time = last["bucket_time"][0]
            direction = "YES" if yes_ratio > 0.5 else "NO"
            bid = 1 if p_win >= args.threshold else 0
            outcome_due = (now + timedelta(minutes=horizon)).strftime("%Y-%m-%dT%H:%M:%SZ")

            paper_conn.execute("""
                INSERT INTO decisions
                (ts, token_id, condition_id, question, direction, p_win, threshold,
                 bucket_time, entry_price, yes_ratio, stake, fee_rate, bid, outcome_due)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (now_str, market.token_yes, cond_id, market.question[:100], direction,
                  p_win, args.threshold, str(bucket_time), entry_price, yes_ratio,
                  args.stake, args.fee_rate, bid, outcome_due))
            paper_conn.commit()

            if bid:
                print(f"    *** BID {direction} *** P(win)={p_win:.1%} | "
                      f"Entry={entry_price:.4f} | Einsatz=${args.stake}")
            else:
                print(f"    NO BID — P(win)={p_win:.1%} < {args.threshold:.0%}")

        print(f"\n  Naechste Analyse in {cfg.bucket.bucket_minutes} Minuten...")
        for _ in range(cfg.bucket.bucket_minutes * 60):
            if not running:
                break
            time.sleep(1)

    print_report(args.paper_db)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Simulator fuer Neuropoly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--token-ids", nargs="+",
                        help="Polymarket Token-IDs zum Tracken")
    parser.add_argument("--model", default="model.txt")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--stake", type=float, default=100.0,
                        help="Simulierter Einsatz pro Trade in USD (default: 100)")
    parser.add_argument("--fee-rate", type=float, default=0.02,
                        help="Round-trip-Kosten als Anteil vom Einsatz (default: 0.02). "
                             "Bei ~1%% Preisbewegung dominiert dieser Wert das Ergebnis.")
    parser.add_argument("--bucket-minutes", type=int, default=5,
                        help="Muss zum Training passen (default: 5)")
    parser.add_argument("--forward-window", type=int, default=6,
                        help="Label-Fenster in Buckets (default: 6 = 30 min)")
    parser.add_argument("--history-buckets", type=int, default=60)
    parser.add_argument("--low-memory", action="store_true",
                        help="Setzen, wenn mit --low-memory trainiert wurde")
    parser.add_argument("--trades-db", default="trades.db",
                        help="SQLite DB von collect_trades.py (optional)")
    parser.add_argument("--paper-db", default="paper_trades.db",
                        help="SQLite DB fuer Paper-Trading Log")
    parser.add_argument("--report", action="store_true",
                        help="Nur Report anzeigen, nicht traden")
    args = parser.parse_args()

    if args.report:
        print_report(args.paper_db)
        return

    if not args.token_ids:
        parser.error("--token-ids ist erforderlich (ausser bei --report)")

    cfg = PipelineConfig()
    cfg.bucket.bucket_minutes = args.bucket_minutes
    cfg.label.forward_window_buckets = args.forward_window
    if args.low_memory:
        cfg.features.lag_buckets = [1, 3, 6]
        cfg.features.rolling_windows = [6, 12]
        cfg.features.cross_market_features = False

    print(f"Lade Modell {args.model}...")
    if not Path(args.model).exists():
        print(f"ERROR: {args.model} nicht gefunden")
        sys.exit(1)
    booster = lgb.Booster(model_file=args.model)
    print(f"  {len(booster.feature_name())} Features geladen")

    print(f"\nLoese {len(args.token_ids)} Token auf...")
    markets: dict[str, MarketInfo] = {}
    for tid in args.token_ids:
        try:
            info = fetch_market(tid)
        except PolymarketAPIError as exc:
            print(f"  ...{tid[-12:]}  ERROR: {exc}")
            continue
        if info is None or not info.condition_id:
            print(f"  ...{tid[-12:]}  nicht gefunden — ueberspringe")
            continue
        markets[info.condition_id] = info
        print(f"  ...{tid[-12:]}  -> {info.question[:52]}")

    if not markets:
        print("Kein Markt aufloesbar. Token-IDs pruefen.")
        sys.exit(1)

    paper_conn = init_paper_db(args.paper_db)
    trading_loop(args, cfg, booster, markets, paper_conn)
    paper_conn.close()


if __name__ == "__main__":
    main()
