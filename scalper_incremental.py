#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scalper_incremental.py
- 15 sn'de bir evreni tarar
- Skor/öznitelikler/TP-SL hesaplar
- Sıkılaştırılabilir filtrelerle sinyal üretir
- Ayrık (disjoint) TP1 olasılıklarını (0–15m, 15–60m, 1–2h, 2–4h, 4–8h, 8–12h, 12–24h) sütun olarak gösterir
- Aday yoksa "neden elendi" istatistiğini yazdırır
- A ve A+ önerileri “Recommended” bloğunda verir
"""

import os
import sys
import time
import math
import json
import signal
import traceback
from datetime import datetime, timezone
from collections import Counter, defaultdict

# ---- 3. parti ----
try:
    import ccxt
except Exception as e:
    print("[FATAL] ccxt gerekli. Kurun: pip install ccxt")
    raise

try:
    from tabulate import tabulate
except Exception as e:
    print("[FATAL] tabulate gerekli. Kurun: pip install tabulate")
    raise

# ---- TP1 ayrık olasılık motoru ----
# Aynı klasörde incr_tp1_engine.py olmalı
INCR_LABELS = ["0–15m","15–60m","1–2h","2–4h","4–8h","8–12h","12–24h"]
try:
    from incr_tp1_engine import IncrTP1Engine
except Exception:
    # Motor yoksa graceful degrade (sütunlar '-' kalır)
    IncrTP1Engine = None

# ================== KONFİG ==================

CONFIG = {
    # Döngü
    "REFRESH_SECS": 15,

    # Evren
    "UNIVERSE": [
        "BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT","XRP/USDT:USDT","ADA/USDT:USDT",
        "LINK/USDT:USDT","LTC/USDT:USDT","SUI/USDT:USDT","DOGE/USDT:USDT","ARB/USDT:USDT",
        # GateIO'da olanlar – kullanıcı çıktılarına göre takma isimler:
        "OKB/USDT:USDT","CFX/USDT:USDT","ENA/USDT:USDT","PYTH/USDT:USDT","PI/USDT:USDT",
        # mizahi semboller de varsa (görsellerde vardı)
        "FARTCOIN/USDT:USDT","TRUMP/USDT:USDT","IP/USDT:USDT","TA/USDT:USDT",
    ],

    # Skor ağırlıkları
    "W": {
        "TrendAlign":       20,
        "VolatilitySweet":  15,
        "VolumeSpike1m":    20,
        "BreakoutQuality":  15,
        "OBIAligned":       10,  # isim doğru
        "LowSpreadScore":   10,
        "RSIImpulse":       10,
    },

    # Filtre/şartlar (test modunda biraz gevşek)
    "MIN_SCORE": 55,           # üretimde 55–60 arası kullan
    "RECOMMEND_MIN": 60,       # A ve üstü için öner
    "OBI_MIN": 0.10,           # hizalama için min |OBI|
    "REQUIRE_ALIGN": True,     # trend hizası (EMA) şart mı?
    "MAX_SPREAD_BPS": 3.0,     # bp
    "TOP_K": 10,               # tabloda en çok N

    # ATR tatlı aralığı (yumuşak değerlendirme)
    "ATR_SWEET_MIN": 0.15,     # %
    "ATR_SWEET_MAX": 0.80,     # %

    # TP-SL (ATR bazlı çoklama)
    "TP1_ATR_MULT": 1.0,
    "TP2_ATR_MULT": 2.0,
    "TP3_ATR_MULT": 3.0,
    "SL_ATR_MULT":  1.0,

    # Log / dosyalar
    "LOG_DIR": "logs",
    "SIGNALS_CSV": "logs/incr_signals.csv",
    "OUTCOMES_CSV": "logs/incr_outcomes.csv",  # motor bunu da kullanıyor
    "ENGINE_CACHE": "logs/incr_probs.json",    # opsiyonel cache
}

# ================== YARDIMCI ==================

def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def ensure_dirs():
    os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)

def nice_bp(x):
    return f"{x:.2f}bp"

def nice_pct(x):
    return f"{x:.2f}%"

def strip_sym(sym):
    # Görsel çıktı için "BTC/USDT:USDT" -> "BTC/USDT"
    return sym.replace(":USDT","")

def grade_from_score(score: float) -> str:
    if score >= 70: return "A+"
    if score >= 60: return "A"
    if score >= 55: return "B"
    return "C"

def clamp01(x):
    return max(0.0, min(1.0, x))

def ema(arr, period):
    if not arr or len(arr) < period:
        return None
    k = 2/(period+1)
    e = arr[0]
    for v in arr[1:]:
        e = v*k + e*(1-k)
    return e

def rsi(values, period=7):
    if len(values) < period+1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period+1):
        ch = values[-i] - values[-i-1]
        if ch >= 0: gains += ch
        else: losses -= ch
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100/(1+rs))

def atr_pct_from_1m(candles_1m, lookback_min=5):
    """
    Gate/ccxt 1m OHLCV: [ts, open, high, low, close, vol]
    5 bar üzerinden TR ortalaması / close * 100
    """
    n = min(len(candles_1m), lookback_min+1)
    if n < lookback_min+1:
        return None
    closes = [c[4] for c in candles_1m[-n:]]
    highs  = [c[2] for c in candles_1m[-n:]]
    lows   = [c[3] for c in candles_1m[-n:]]
    trs = []
    for i in range(1, n):
        h, l, pc = highs[i], lows[i], closes[i-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    atr = sum(trs)/len(trs)
    last_close = closes[-1]
    return (atr / last_close) * 100.0, last_close

def volume_spike_1m(candles_1m, ref=5):
    if len(candles_1m) < ref+1:
        return 0.0
    v_last = candles_1m[-1][5]
    v_ref = sum(c[5] for c in candles_1m[-(ref+1):-1]) / ref
    if v_ref <= 0: return 0.0
    return v_last / v_ref

def breakout_quality(candles_1m, atr_abs):
    """
    Son kapanışın son 20'nci dakikanın min/max'ına göre normalleşmiş breakout'u.
    """
    if len(candles_1m) < 21 or atr_abs is None or atr_abs <= 0:
        return 0.0
    closes = [c[4] for c in candles_1m]
    last = closes[-1]
    window = closes[-21:-1]
    hi = max(window); lo = min(window)
    # Üst kırılım
    up = max(0.0, last - hi) / atr_abs
    dn = max(0.0, lo - last) / atr_abs
    # İki yönde normalize: 0..1
    b = max(up, dn)
    return clamp01(b)

def orderbook_obi(ob, depth=10):
    """
    (bid_qty - ask_qty) / (bid_qty + ask_qty), 0..±1
    """
    bids = ob.get("bids", [])[:depth]
    asks = ob.get("asks", [])[:depth]
    bsum = sum([p*q for p,q in ((b[0], b[1]) for b in bids)])
    asum = sum([p*q for p,q in ((a[0], a[1]) for a in asks)])
    denom = bsum + asum
    if denom <= 0: return 0.0
    return (bsum - asum) / denom

def spread_bps(ob):
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks: return 999.0
    bid = bids[0][0]; ask = asks[0][0]
    mid = (bid + ask)/2.0
    if mid <= 0: return 999.0
    return (ask - bid)/mid * 10000.0

def volatility_sweet(atr_pct, lo, hi):
    """
    ATR sweet-spot: hedef aralıkta 1.0, dışına çıktıkça lineer düş.
    """
    if atr_pct is None: return 0.0
    if atr_pct <= lo:
        return clamp01(1.0 - (lo - atr_pct)/lo)  # lo'dan çok aşağıda ise düş
    if atr_pct >= hi:
        return clamp01(1.0 - (atr_pct - hi)/hi)  # hi'dan çok yukarıda ise düş
    return 1.0

def rsi_impulse(closes):
    r = rsi(closes, period=7)
    if r is None: return 0.0
    # 50 çevresine yakınlık ve momentum için kaba bir skala
    # 50'ye ne kadar uzaksa o kadar "impulse": 0..1
    return clamp01(abs(r - 50.0) / 50.0)

# ================== PUANLAMA ==================

def score_one_side(direction, feats, obi, spr_bps, W):
    """
    Yön bazlı skor. direction: "long" veya "short"
    feats:
      - ema9, ema21, ema50 (closes)
      - atr_pct, atr_abs, vol_spike, breakout, rsi_imp
    """
    trend_score = 0.0
    if feats["ema9"] and feats["ema21"] and feats["ema50"]:
        if direction == "long":
            trend_score = 1.0 if (feats["ema9"] > feats["ema21"] > feats["ema50"]) else 0.0
        else:
            trend_score = 1.0 if (feats["ema9"] < feats["ema21"] < feats["ema50"]) else 0.0

    vol_sweet = volatility_sweet(
        feats["atr_pct"], CONFIG["ATR_SWEET_MIN"], CONFIG["ATR_SWEET_MAX"]
    )

    # hacim spike (normalize)
    vol_spike_norm = clamp01(feats["vol_spike"] / 2.5)  # 2.5x ve üstü ~1.0

    # breakout
    breakout_norm = feats["breakout"]

    # OBI hizalı katkı (sadece yön lehine olan kısım)
    if direction == "long":
        obi_aligned = max(0.0, obi)
    else:
        obi_aligned = max(0.0, -obi)
    obi_norm = clamp01(obi_aligned)

    # spread
    low_spread = clamp01(1.0 - (spr_bps / CONFIG["MAX_SPREAD_BPS"]))

    # rsi impulse
    rsi_imp = feats["rsi_imp"]

    total = (
        W["TrendAlign"]      * trend_score +
        W["VolatilitySweet"] * vol_sweet +
        W["VolumeSpike1m"]   * vol_spike_norm +
        W["BreakoutQuality"] * breakout_norm +
        W["OBIAligned"]      * obi_norm +
        W["LowSpreadScore"]  * low_spread +
        W["RSIImpulse"]      * rsi_imp
    )
    detail = {
        "Trend": round(trend_score, 2),
        "ATR%": round(feats["atr_pct"] or 0.0, 2),
        "VolSpike": round(feats["vol_spike"], 2),
        "Breakout": round(breakout_norm, 2),
        "OBIAligned": round(obi_norm, 2),
        "LowSpread": round(low_spread, 2),
        "RSIImp": round(rsi_imp, 2),
    }
    return total, detail

def choose_direction(feats, obi, spr_bps, W):
    s_long, d_long = score_one_side("long", feats, obi, spr_bps, W)
    s_short, d_short = score_one_side("short", feats, obi, spr_bps, W)
    if s_long >= s_short:
        return "long", s_long, d_long
    else:
        return "short", s_short, d_short

# ================== TP/SL ==================

def tps_sl(last, atr_pct, direction):
    # ATR yüzde → mutlak ATR
    atr_abs = (atr_pct/100.0) * last if atr_pct is not None else last*0.002
    tp1 = CONFIG["TP1_ATR_MULT"] * atr_abs
    tp2 = CONFIG["TP2_ATR_MULT"] * atr_abs
    tp3 = CONFIG["TP3_ATR_MULT"] * atr_abs
    sl  = CONFIG["SL_ATR_MULT"]  * atr_abs

    if direction == "long":
        return last+tp1, last+tp2, last+tp3, last-sl, atr_abs/last*100.0
    else:
        return last-tp1, last-tp2, last-tp3, last+sl, atr_abs/last*100.0

# ================== MOTOR / BASKI ==================

class ProbAdapter:
    """
    incr_tp1_engine yoksa graceful degrade.
    prob_for(sym, grade) -> (probs_dict, N, src)
    """
    def __init__(self, log_dir):
        self.engine = None
        if IncrTP1Engine:
            try:
                self.engine = IncrTP1Engine(log_dir=log_dir)
            except Exception:
                self.engine = None

    def prob_for(self, sym, grade):
        if not self.engine:
            return None, None, None
        try:
            # grade ve sembole göre ayrık olasılıkları getir
            res = self.engine.prob_for(sym=sym, grade=grade)
            # beklenen: {"0–15m":0.12, "15–60m":0.18, ..., "N": 45, "src":"logs"}
            if not res:
                return None, None, None
            probs = {k:v for k,v in res.items() if k in INCR_LABELS}
            N = res.get("N", None)
            src = res.get("src", None)
            return probs, N, src
        except Exception:
            return None, None, None

def print_table(rows, title="DISJOINT TP1 PROB VIEW"):
    # Sütunlar
    header = [
        "Symbol","Score","Dir","ATR%5m","VolSpike","Breakout","OBI","Align","Spread",
        "Last","TP1","TP2","TP3","SL",
        *INCR_LABELS,
        "Fast%","N","Src","Grade"
    ]
    table = []
    for r in rows:
        table.append([
            strip_sym(r["sym"]), int(round(r["score"])), r["dir"],
            nice_pct(r["atr_pct"]) if r["atr_pct"] is not None else "-",
            f"{r['vol_spike']:.2f}x",
            f"{r['breakout']:.2f}",
            f"{r['obi']:+.3f}",
            "✔" if r["align_ok"] else "·",
            nice_bp(r["spread_bps"]),
            f"{r['last']:.6g}",
            f"{r['tp1']:.6g}",
            f"{r['tp2']:.6g}",
            f"{r['tp3']:.6g}",
            f"{r['sl']:.6g}",
            *( ("-" if r["probs"].get(lbl) is None else f"{r['probs'][lbl]*100:.1f}%") for lbl in INCR_LABELS ),
            ("-" if r["fast_prob"] is None else f"{r['fast_prob']*100:.1f}%"),
            "-" if r["N"] is None else r["N"],
            "-" if r["src"] is None else r["src"],
            r["grade"],
        ])
    print(utcnow_iso())
    print("+ " + title + " +")
    print(tabulate(table, headers=header, tablefmt="github"))
    print()

def print_recommended(rows):
    rec = [r for r in rows if r["score"] >= CONFIG["RECOMMEND_MIN"] and r["should_signal"]]
    if not rec:
        print("[RECOMMENDED (disjoint, FAST)] — uygun aday yok.")
        return
    header = [
        "Symbol","Score","Dir","ATR%5m","VolSpike","Breakout","OBI","Align","Spread",
        "Last","TP1","TP2","TP3","SL","Fast%","N","Grade"
    ]
    table=[]
    for r in rec:
        table.append([
            strip_sym(r["sym"]), int(round(r["score"])), r["dir"],
            nice_pct(r["atr_pct"]) if r["atr_pct"] is not None else "-",
            f"{r['vol_spike']:.2f}x",
            f"{r['breakout']:.2f}",
            f"{r['obi']:+.3f}",
            "✔" if r["align_ok"] else "·",
            nice_bp(r["spread_bps"]),
            f"{r['last']:.6g}",
            f"{r['tp1']:.6g}",
            f"{r['tp2']:.6g}",
            f"{r['tp3']:.6g}",
            f"{r['sl']:.6g}",
            ("-" if r["fast_prob"] is None else f"{r['fast_prob']*100:.1f}%"),
            "-" if r["N"] is None else r["N"],
            r["grade"],
        ])
    print("[RECOMMENDED ≥ A]")
    print(tabulate(table, headers=header, tablefmt="github"))
    print()

# ================== ANA DÖNGÜ ==================

def main():
    ensure_dirs()

    # ccxt exchange
    ex = ccxt.gateio({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
    })
    ex.load_markets()

    # TP1 ayrık motoru
    prob_engine = ProbAdapter(CONFIG["LOG_DIR"])

    # Reddedilme nedenleri sayacı
    reject_stats = Counter()

    # Sinyal CSV başlığı
    if not os.path.exists(CONFIG["SIGNALS_CSV"]):
        with open(CONFIG["SIGNALS_CSV"], "w", encoding="utf-8") as f:
            f.write("ts,symbol,dir,last,tp1,tp2,tp3,sl,score,grade,atr_pct,vol_spike,breakout,obi,spread_bps\n")

    universe = [s for s in CONFIG["UNIVERSE"] if s in ex.markets]

    print(f"[INFO] Scalper Incremental başlıyor. Evren={len(universe)} sembol. Döngü={CONFIG['REFRESH_SECS']} sn")
    time.sleep(1)

    while True:
        ts = utcnow_iso()
        print(f"[HB] {ts} cycle start")

        rows = []
        topk = CONFIG["TOP_K"]

        # Her döngü için istatistikleri sıfırla
        cycle_reject = Counter()

        for sym in universe:
            try:
                # 1) 1m mumlar
                candles = ex.fetch_ohlcv(sym, timeframe="1m", limit=60)
                if not candles or len(candles) < 25:
                    cycle_reject["ohlcv<25"] += 1
                    continue

                closes = [c[4] for c in candles]
                atr_pct, last = atr_pct_from_1m(candles, lookback_min=5)
                if atr_pct is None:
                    cycle_reject["atr=None"] += 1
                    continue

                # ATR absolute (TP/SL için)
                atr_abs = (atr_pct/100.0) * last

                # 2) orderbook
                ob = ex.fetch_order_book(sym, limit=20)
                obi = orderbook_obi(ob, depth=10)
                spr_bps = spread_bps(ob)

                # 3) özellikler
                vol_spike = volume_spike_1m(candles, ref=5)
                bq = breakout_quality(candles, atr_abs=atr_abs)
                ema9  = ema([c[4] for c in candles[-30:]], 9)
                ema21 = ema([c[4] for c in candles[-60:]], 21)
                ema50 = ema([c[4] for c in candles[-60:]], 50)
                rimp = rsi_impulse([c[4] for c in candles[-60:]])

                feats = {
                    "atr_pct": atr_pct,
                    "atr_abs": atr_abs,
                    "vol_spike": vol_spike,
                    "breakout": bq,
                    "ema9": ema9, "ema21": ema21, "ema50": ema50,
                    "rsi_imp": rimp,
                }

                # 4) yön & skor
                direction, score, detail = choose_direction(feats, obi, spr_bps, CONFIG["W"])

                # 5) trend hizası gerekli mi?
                align_ok = True
                if CONFIG["REQUIRE_ALIGN"]:
                    if direction == "long":
                        align_ok = True if (ema9 and ema21 and ema50 and ema9 > ema21 > ema50) else False
                    else:
                        align_ok = True if (ema9 and ema21 and ema50 and ema9 < ema21 < ema50) else False

                # 6) OBI minimum hizalama şartı
                obi_ok = ((direction == "long" and obi >= +CONFIG["OBI_MIN"]) or
                          (direction == "short" and obi <= -CONFIG["OBI_MIN"]))

                # 7) Filtreler
                reasons = []
                if score < CONFIG["MIN_SCORE"]: reasons.append(f"score<{CONFIG['MIN_SCORE']}")
                if not align_ok: reasons.append("!align")
                if not obi_ok: reasons.append(f"OBI|<{CONFIG['OBI_MIN']}")
                if spr_bps > CONFIG["MAX_SPREAD_BPS"]: reasons.append(f"spread>{CONFIG['MAX_SPREAD_BPS']}bp")
                if direction not in ("long","short"): reasons.append("dir=-")

                should_signal = (len(reasons) == 0)

                # 8) TP/SL
                tp1, tp2, tp3, sl, d_tp1_pct = tps_sl(last, atr_pct, direction)

                # 9) Olasılıklar (ayrık)
                probs, N, src = prob_engine.prob_for(strip_sym(sym), grade_from_score(score)) if prob_engine else (None, None, None)
                probs = probs or {}
                # FAST varsayımı: 0–15m + 15–60m
                fast_prob = None
                if probs:
                    p1 = probs.get("0–15m", 0.0) or 0.0
                    p2 = probs.get("15–60m", 0.0) or 0.0
                    fast_prob = p1 + p2

                row = {
                    "sym": sym,
                    "dir": direction,
                    "score": score,
                    "grade": grade_from_score(score),

                    "atr_pct": atr_pct,
                    "vol_spike": vol_spike,
                    "breakout": bq,
                    "obi": obi,
                    "align_ok": align_ok,
                    "spread_bps": spr_bps,

                    "last": last,
                    "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": sl,
                    "d_tp1_pct": d_tp1_pct,

                    "should_signal": should_signal,
                    "reasons": reasons,

                    "probs": probs,
                    "fast_prob": fast_prob,
                    "N": N,
                    "src": src,
                }
                rows.append(row)

                # reddedilme sayacı
                if reasons:
                    cycle_reject[",".join(sorted(reasons))] += 1

            except Exception as e:
                cycle_reject["exception"] += 1
                # İsteğe bağlı hata ayrıntısı:
                # print(f"[ERR] {sym}: {e}")

        # Skora göre sırala ve kırp
        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:topk]

        # Tablo bas
        print_table(rows, title="DISJOINT TP1 PROB VIEW")

        # Reddedilme nedenleri özeti
        if cycle_reject:
            top5 = cycle_reject.most_common(5)
            print("[DBG] Top reject reasons:", "; ".join(f"{k} x{v}" for k,v in top5))
        else:
            print("[DBG] Hiç reddedilen aday yok (filtreler geçilmiş).")

        # Recommended (A ve üstü + filtreleri geçen)
        print_recommended(rows)

        # Sinyalleri CSV'ye yaz (A ve üstü + should_signal)
        for r in rows:
            if r["should_signal"] and r["score"] >= CONFIG["RECOMMEND_MIN"]:
                with open(CONFIG["SIGNALS_CSV"], "a", encoding="utf-8") as f:
                    f.write(",".join(map(str, [
                        utcnow_iso(), strip_sym(r["sym"]), r["dir"], f"{r['last']:.8f}",
                        f"{r['tp1']:.8f}", f"{r['tp2']:.8f}", f"{r['tp3']:.8f}", f"{r['sl']:.8f}",
                        f"{r['score']:.2f}", r["grade"],
                        f"{r['atr_pct']:.4f}", f"{r['vol_spike']:.4f}", f"{r['breakout']:.4f}",
                        f"{r['obi']:.4f}", f"{r['spread_bps']:.4f}",
                    ])) + "\n")

        # Uyku
        time.sleep(CONFIG["REFRESH_SECS"])


# ================== ÇALIŞTIR ==================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Durduruldu.")
    except Exception as e:
        print("[FATAL] Çalışma hatası:", e)
        traceback.print_exc()
        sys.exit(1)
