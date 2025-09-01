import os
import time
import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

from incr_tp1_engine import (
    INCR_EDGES_MIN,
    _fmt_incr_key,
    _first_touch_is_tp1,
    IncrTP1Engine,
)

LOG_DIR = "logs"
SIGNALS_CSV = os.path.join(LOG_DIR, "incr_signals.csv")
OUTCOMES_CSV = os.path.join(LOG_DIR, "incr_outcomes.csv")
STATS_JSON = "ptp1_stats.json"

FETCH_INTERVAL = 1.0  # seconds


class PriceFetcher:
    """Wrap ccxt to fetch last price for a symbol."""

    def __init__(self):
        if ccxt is None:
            self.exchange = None
        else:
            self.exchange = ccxt.binance()

    def last_price(self, symbol: str) -> Optional[float]:
        if not self.exchange:
            return None
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker.get("last"))
        except Exception:
            return None


def ensure_logs():
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(OUTCOMES_CSV):
        pd.DataFrame(
            columns=[
                "t_signal",
                "symbol",
                "direction",
                "grade",
                "tp1_ts",
                "sl_ts",
            ]
        ).to_csv(OUTCOMES_CSV, index=False)


def monitor_signal(row: pd.Series, fetcher: PriceFetcher) -> pd.Series:
    """Monitor price until TP1 or SL is touched; return outcome row."""

    t_signal = pd.to_datetime(row["t_signal"], unit="ms", utc=True)
    symbol = row["symbol"]
    direction = row["direction"]
    grade = row.get("grade", None)
    tp1 = float(row["tp1"])
    sl = float(row["sl"])

    while True:
        price = fetcher.last_price(symbol)
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        if price is not None:
            if direction == "long":
                if price >= tp1:
                    return pd.Series(
                        {
                            "t_signal": t_signal,
                            "symbol": symbol,
                            "direction": direction,
                            "grade": grade,
                            "tp1_ts": now,
                            "sl_ts": pd.NaT,
                        }
                    )
                if price <= sl:
                    return pd.Series(
                        {
                            "t_signal": t_signal,
                            "symbol": symbol,
                            "direction": direction,
                            "grade": grade,
                            "tp1_ts": pd.NaT,
                            "sl_ts": now,
                        }
                    )
            else:  # short
                if price <= tp1:
                    return pd.Series(
                        {
                            "t_signal": t_signal,
                            "symbol": symbol,
                            "direction": direction,
                            "grade": grade,
                            "tp1_ts": now,
                            "sl_ts": pd.NaT,
                        }
                    )
                if price >= sl:
                    return pd.Series(
                        {
                            "t_signal": t_signal,
                            "symbol": symbol,
                            "direction": direction,
                            "grade": grade,
                            "tp1_ts": pd.NaT,
                            "sl_ts": now,
                        }
                    )
        time.sleep(FETCH_INTERVAL)


def write_stats(csv_path: str, json_path: str):
    """Compute disjoint TP1 stats and write to JSON."""
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
    for col in ["t_signal", "tp1_ts", "sl_ts"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    buckets = {}
    for j in range(1, len(INCR_EDGES_MIN)):
        lo, hi = INCR_EDGES_MIN[j - 1], INCR_EDGES_MIN[j]
        key = _fmt_incr_key(lo, hi)
        buckets[key] = {"hits": 0, "trials": 0}

    for _, r in df.iterrows():
        t_tp1_m = (
            (r["tp1_ts"] - r["t_signal"]).total_seconds() / 60.0
            if pd.notna(r["tp1_ts"])
            else None
        )
        t_sl_m = (
            (r["sl_ts"] - r["t_signal"]).total_seconds() / 60.0
            if pd.notna(r["sl_ts"])
            else None
        )
        t_first = None
        if t_tp1_m is not None and t_sl_m is not None:
            t_first = min(t_tp1_m, t_sl_m)
        elif t_tp1_m is not None:
            t_first = t_tp1_m
        elif t_sl_m is not None:
            t_first = t_sl_m

        for j in range(1, len(INCR_EDGES_MIN)):
            lo, hi = INCR_EDGES_MIN[j - 1], INCR_EDGES_MIN[j]
            key = _fmt_incr_key(lo, hi)
            if t_first is not None and (t_first > lo) and (t_first <= hi):
                buckets[key]["trials"] += 1
                if _first_touch_is_tp1(t_tp1_m, t_sl_m) and t_tp1_m is not None and (t_tp1_m > lo) and (t_tp1_m <= hi):
                    buckets[key]["hits"] += 1
                break

    data = {
        "updated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "buckets": buckets,
    }
    with open(json_path, "w") as f:
        json.dump(data, f)


def main():
    ensure_logs()
    fetcher = PriceFetcher()
    processed = set()
    if os.path.exists(OUTCOMES_CSV):
        df_out = pd.read_csv(OUTCOMES_CSV)
        processed = set(df_out["t_signal"].astype(str))

    while True:
        if not os.path.exists(SIGNALS_CSV):
            time.sleep(5)
            continue
        df = pd.read_csv(SIGNALS_CSV)
        if "t_signal" not in df.columns:
            df = df.rename(columns={"ts": "t_signal", "dir": "direction"})
        for _, row in df.iterrows():
            key = str(row["t_signal"])
            if key in processed:
                continue
            outcome_row = monitor_signal(row, fetcher)
            outcome_row_df = outcome_row.to_frame().T
            if os.path.exists(OUTCOMES_CSV):
                outcome_row_df.to_csv(
                    OUTCOMES_CSV, mode="a", header=False, index=False
                )
            else:
                outcome_row_df.to_csv(OUTCOMES_CSV, index=False)
            processed.add(key)
            # update stats
            write_stats(OUTCOMES_CSV, STATS_JSON)
            # refresh engine
            engine = IncrTP1Engine(OUTCOMES_CSV)
            engine.refresh(force=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
