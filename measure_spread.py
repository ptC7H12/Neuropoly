"""Messe den echten Spread in den Maerkten, die wir handeln wuerden."""
import sys, time; sys.path.insert(0, '/root/Neuropoly')
import polars as pl, requests
from concurrent.futures import ThreadPoolExecutor
import threading

R = "/root/Neuropoly/"
BOOK = "https://clob.polymarket.com/book"
_local = threading.local()

def sess():
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session(); _local.s = s
    return s

def fetch(row):
    """best bid/ask und Groesse an der Spitze fuer ein Token."""
    for attempt in range(4):
        try:
            r = sess().get(BOOK, params={"token_id": row["token1"]}, timeout=20)
            if r.status_code == 200:
                b = r.json()
                bids, asks = b.get("bids") or [], b.get("asks") or []
                if not bids or not asks:
                    return None
                bb = max(bids, key=lambda x: float(x["price"]))
                ba = min(asks, key=lambda x: float(x["price"]))
                bid, ask = float(bb["price"]), float(ba["price"])
                return {"market_id": row["market_id"], "segment": row["segment"],
                        "usd": row["usd"], "slug": row["market_slug"],
                        "bid": bid, "ask": ask, "spread": round(ask - bid, 6),
                        "mid": round((ask + bid) / 2, 6),
                        "bid_size": float(bb.get("size", 0)), "ask_size": float(ba.get("size", 0)),
                        "n_bids": len(bids), "n_asks": len(asks)}
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt); continue
            return None
        except Exception:
            time.sleep(2 ** attempt)
    return None

d = pl.read_parquet(R + "data/open_trainable.parquet")
print(f"Frage {d.height:,} Orderbuecher ab ...", flush=True)
rows = d.to_dicts()
out, t0 = [], time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, res in enumerate(ex.map(fetch, rows), 1):
        if res: out.append(res)
        if i % 250 == 0:
            print(f"  {i:,}/{len(rows):,}  ({len(out):,} mit Buch)  {time.time()-t0:.0f}s", flush=True)

b = pl.DataFrame(out)
b.write_parquet(R + "data/spreads.parquet")
print(f"\nErhalten: {b.height:,} von {d.height:,}  in {time.time()-t0:.0f}s")
print(f"-> data/spreads.parquet\n")

print(f"{'':<26}{'n':>7}{'Median':>9}{'p25':>8}{'p75':>8}{'p90':>8}")
print("-"*66)
def line(name, df):
    if df.height == 0: return
    s = df["spread"]
    print(f"{name:<26}{df.height:>7,}{s.median():>9.4f}{s.quantile(.25):>8.4f}"
          f"{s.quantile(.75):>8.4f}{s.quantile(.90):>8.4f}")
line("alle", b)
for seg in ("other", "politics"):
    line(f"  {seg}", b.filter(pl.col("segment") == seg))
print()
# volumengewichtet: so handelt man tatsaechlich
tot = float(b["usd"].sum())
vw = float((b["spread"] * b["usd"]).sum() / tot)
print(f"Volumengewichteter Spread : {vw:.4f}")
print(f"Einfacher Median          : {b['spread'].median():.4f}")
print(f"Annahme in allen Rechnungen: 0.0100")
