from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

INCR_EDGES_MIN = [0, 15, 60, 120, 240, 480, 720, 1440]

def _fmt_incr_key(lo:int, hi:int) -> str:
    if hi == 15: return "0–15m"
    if hi == 60: return "15–60m"
    if hi == 120: return "1–2h"
    if hi == 240: return "2–4h"
    if hi == 480: return "4–8h"
    if hi == 720: return "8–12h"
    if hi == 1440: return "12–24h"
    return f"{lo}-{hi}m"

INCR_LABELS = [_fmt_incr_key(INCR_EDGES_MIN[i-1], INCR_EDGES_MIN[i]) for i in range(1, len(INCR_EDGES_MIN))]

MIN_ROWS_FOR_SYMBOL_BUCKET = 30
MIN_ROWS_FOR_GRADE_BUCKET  = 30
BETA_PRIOR = (30, 20)  # Beta smoothing

@dataclass
class IncrProbRow:
    N: int
    probs: Dict[str, float]  # yüzdeler

def _minutes_between(a: pd.Timestamp, b: pd.Timestamp) -> Optional[float]:
    if pd.isna(a) or pd.isna(b):
        return None
    return (b - a).total_seconds() / 60.0

def _first_touch_is_tp1(t_tp1_m: Optional[float], t_sl_m: Optional[float]) -> bool:
    if t_tp1_m is None:
        return False
    if t_sl_m is None:
        return True
    return t_tp1_m < t_sl_m  # eşitlikte SL önce sayılır (konservatif)

def _beta_smooth(success:int, total:int, alpha0:int, beta0:int) -> float:
    fail = max(0, total - success)
    return (alpha0 + success) / (alpha0 + beta0 + success + fail) if total >= 0 else np.nan

def _safe_pct(x: float) -> float:
    return float(np.round(100.0 * x, 1)) if pd.notna(x) else np.nan

class IncrTP1Engine:
    """
    signals.csv'den yalnız ARTIMSAL (disjoint) TP1 olasılıklarını çıkarır.
    Gerekli kolonlar: t_signal, symbol, direction, grade (opsiyonel),
    tp1_ts, sl_ts (ilk dokunuşlar; UTC).
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._cache_mtime = None
        self.symbol_dir: Dict[Tuple[str,str], IncrProbRow] = {}
        self.grade_dir:  Dict[Tuple[str,str], IncrProbRow] = {}

    def _load(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(self.csv_path):
            return None
        df = pd.read_csv(self.csv_path)
        need = ["t_signal","symbol","direction","grade"]
        if any(col not in df.columns for col in need):
            return None
        for c in ["t_signal","tp1_ts","sl_ts"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
        return df

    def refresh(self, force: bool=False):
        mtime = os.path.getmtime(self.csv_path) if os.path.exists(self.csv_path) else None
        if (not force) and (mtime == self._cache_mtime):
            return
        df = self._load()
        if df is None or len(df) == 0:
            self.symbol_dir, self.grade_dir = {}, {}
            self._cache_mtime = mtime
            return

        df = df[df["direction"].isin(["long","short"])].copy()
        df["t_tp1_m"] = [_minutes_between(r["t_signal"], r.get("tp1_ts", pd.NaT)) for _, r in df.iterrows()]
        df["t_sl_m"]  = [_minutes_between(r["t_signal"], r.get("sl_ts",  pd.NaT)) for _, r in df.iterrows()]

        df["incr_bucket"] = None
        for i, r in df.iterrows():
            t1, ts = r["t_tp1_m"], r["t_sl_m"]
            if (t1 is None) or (not _first_touch_is_tp1(t1, ts)):
                continue
            for j in range(1, len(INCR_EDGES_MIN)):
                lo, hi = INCR_EDGES_MIN[j-1], INCR_EDGES_MIN[j]
                if (t1 > lo) and (t1 <= hi):
                    df.at[i, "incr_bucket"] = _fmt_incr_key(lo, hi)
                    break

        def aggregate(sub: pd.DataFrame) -> IncrProbRow:
            N = len(sub)
            a0, b0 = BETA_PRIOR
            probs = {}
            for key in INCR_LABELS:
                S = int((sub["incr_bucket"] == key).sum())
                probs[key] = _safe_pct(_beta_smooth(S, N, a0, b0))
            return IncrProbRow(N=N, probs=probs)

        sym_map: Dict[Tuple[str,str], IncrProbRow] = {}
        g = df.groupby(["symbol","direction"], observed=False)
        for (sym, d), sub in g:
            if len(sub) >= MIN_ROWS_FOR_SYMBOL_BUCKET:
                sym_map[(sym, d)] = aggregate(sub)

        grd_map: Dict[Tuple[str,str], IncrProbRow] = {}
        if "grade" in df.columns:
            g2 = df.groupby(["grade","direction"], observed=False)
            for (gr, d), sub in g2:
                if len(sub) >= MIN_ROWS_FOR_GRADE_BUCKET:
                    grd_map[(gr, d)] = aggregate(sub)

        self.symbol_dir, self.grade_dir = sym_map, grd_map
        self._cache_mtime = mtime

    def get_for(self, *, symbol:str, direction:str, grade:Optional[str]=None) -> Optional[IncrProbRow]:
        key1 = (symbol, direction)
        if key1 in self.symbol_dir:
            return self.symbol_dir[key1]
        if grade is not None:
            key2 = (grade, direction)
            if key2 in self.grade_dir:
                return self.grade_dir[key2]
        return None
