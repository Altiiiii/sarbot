# mvp_scalper_balanced_v2.py
# ==========================
# Balanced scalper (GateIO swap USDT): 15s snapshot, 1m/5m/15m bar tabanlı özellikler
# Gerekenler:
#   pip install ccxt pandas numpy aiohttp pytz tabulate
#
# Özellikler:
# - Likidite ve spread filtresi
# - EMA50/200 (5m & 15m) trend hizası
# - ATR%(5m) "sweet spot" puanı
# - 1m hacim spike, 1m BB genişleme/breakout, RSI(1m) 50 kesişim ivmesi
# - Orderbook imbalance (OBI) ve YÖNLE HİZALAMA filtresi (OBI_ALIGN_MIN)
# - P(TP1) online tahminlemesi: kapanan sinyallerden yürüyen başarı oranı
# - Sinyalleri/sonuçları CSV’ye yazar; stats JSON’a kalıcı
# - Telegram bildirimi (opsiyonel, TG_TOKEN ve TG_CHAT_ID ile)

import os, time, json, asyncio, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from collections import deque
import numpy as np
import pandas as pd
from tabulate import tabulate
import ccxt.async_support as ccxt

# ================== CONFIG ==================
CONFIG = {
    # Borsa
    "EXCHANGE_ID": "gateio",
    "MARKET_TYPE": "swap",         # USDT-M perpetual
    "QUOTE": "USDT",

    # Evren filtresi
    "MIN_24H_VOL_USD": 15_000_000,
    "MAX_SPREAD_BPS": 4.0,         # 0.04%
    "MAX_SYMBOLS": 20,

    # Döngü / veri çekim aralıkları
    "SNAPSHOT_SEC": 15,            # 15 sn
    "LOOKBACK_1M": 220,
    "LOOKBACK_5M": 220,
    "LOOKBACK_15M": 220,

    # Özellik ve skor
    "ORDERBOOK_LIMIT": 5,
    "VOL_SPIKE_WINDOW": 20,
    "ATR_PERIOD": 14,
    "EMA_FAST": 50,
    "EMA_SLOW": 200,
    "BB_PERIOD": 20,
    "BB_STD": 2.0,
    "OBI_TOPK": 10,

    # Sinyal eşiği ve filtreler
    "WATCH_THRESHOLD": 55,
    "SIGNAL_THRESHOLD": 60,
    "OBI_ALIGN_MIN": 0.10,         # LONG için OBI >= +0.10, SHORT için OBI <= -0.10

    # Sinyal tekrarını sınırlama
    "COOLDOWN_MIN": 8,

    # Çıkış seti
    "STOP_MULT": 1.2,
    "TP_MULTS": (1, 2, 3),

    # P(TP1) online tahmin penceresi
    "PTP1_WINDOW": 500,            # son N kapalı sinyal

    # Log dosyaları
    "LOG_DIR": "./logs",
    "SIGNALS_CSV": "./signals_log.csv",
    "OUTCOMES_CSV": "./outcomes_log.csv",
    "STATS_JSON": "./ptp1_stats.json",

    # Telegram (opsiyonel)
    "ENABLE_TELEGRAM": True,
}

TELEGRAM_TOKEN = os.getenv("TG_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TG_CHAT_ID", "")

# ============== LOGGING =====================
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)
logger = logging.getLogger("scalper")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = RotatingFileHandler(os.path.join(CONFIG["LOG_DIR"], "scalper.log"), maxBytes=2_000_000, backupCount=3)
fh.setFormatter(fmt)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

# Pandas ayarları ve future warning kontrolü
pd.options.mode.copy_on_write = True
pd.options.mode.chained_assignment = None

# =============== HELPERS ====================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=False).mean()
    ma_down = down.ewm(com=period - 1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def bbands(series: pd.Series, period: int = 20, std: float = 2.0):
    mid = series.rolling(period).mean()
    sd = series.rolling(period).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    width = (upper - lower) / (mid + 1e-12)
    return upper, mid, lower, width

def vwap_window(df: pd.DataFrame, window: int = 30) -> float:
    sub = df.tail(window)
    pv = (sub['close'] * sub['volume']).sum()
    vv = sub['volume'].sum() + 1e-12
    return pv / vv

def extract_quote_volume_usd(tkr: dict) -> float:
    qv = tkr.get('quoteVolume')
    if qv is not None:
        return float(qv)
    bv = tkr.get('baseVolume')
    last = tkr.get('last') or tkr.get('close')
    if bv is not None and last is not None:
        return float(bv) * float(last)
    info = tkr.get('info') or {}
    for key in ['quote_volume', 'quoteVolume']:
        if key in info:
            try: return float(info[key])
            except: pass
    return 0.0

def spread_bps(tkr: dict) -> float:
    bid = tkr.get('bid')
    ask = tkr.get('ask')
    if not (bid and ask and bid > 0):
        return float('inf')
    mid = (bid + ask) / 2
    return ((ask - bid) / mid) * 1e4

async def send_telegram(text: str):
    if not (CONFIG["ENABLE_TELEGRAM"] and TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        import aiohttp
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=payload, timeout=10) as resp:
                await resp.text()
    except Exception as e:
        logger.warning(f"Telegram gönderilemedi: {e}")

def ohlcv_to_df(ohlcv: list) -> pd.DataFrame:
    if not ohlcv:
        return pd.DataFrame(columns=['ts','open','high','low','close','volume'])
    arr = np.array(ohlcv, dtype=float)
    df = pd.DataFrame(arr, columns=['ts','open','high','low','close','volume'])
    return df

def volatility_sweet_score(atr_pct: float) -> float:
    # Tatlı bölge tepe ~ %1.0; ±1.4 tolerans
    if atr_pct <= 0.1 or atr_pct >= 3.5:
        return 0.0
    ideal = 1.0
    spread = 1.4
    dist = abs(atr_pct - ideal) / spread
    return max(0.0, 1.0 - dist)

def orderbook_imbalance(ob: dict) -> float:
    if not ob: return 0.0
    bid_qty = sum(x[1] for x in ob.get('bids', []))
    ask_qty = sum(x[1] for x in ob.get('asks', []))
    denom = (bid_qty + ask_qty) + 1e-12
    return (bid_qty - ask_qty) / denom

def get_price_tick(markets: dict, symbol: str) -> float | None:
    m = markets.get(symbol) or {}
    prec = (m.get('precision') or {}).get('price')
    if isinstance(prec, int): return 10 ** (-prec)
    if isinstance(prec, float) and prec > 0: return float(prec)
    info = m.get('info') or {}
    for key in ('priceIncrement', 'tickSize', 'price_tick', 'price_step'):
        v = info.get(key)
        try:
            v = float(v)
            if v and v > 0: return v
        except: pass
    return None

def calc_exits(entry: float, atr_pct_5m: float, side: str,
               stop_mult: float = 1.2, tp_mults=(1,2,3), tick: float | None = None):
    ratio = float(atr_pct_5m) / 100.0
    atr_abs = max(1e-12, entry * ratio)
    def q(x: float) -> float:
        if tick and tick > 0: return round(x / tick) * tick
        if entry >= 1000: return round(x, 1)
        if entry >= 100:  return round(x, 2)
        if entry >= 10:   return round(x, 3)
        if entry >= 1:    return round(x, 4)
        return round(x, 6)
    s = (side or "").strip().lower()
    if s == "long":
        raw_tps = [entry + m * atr_abs for m in tp_mults]
        sl = entry - stop_mult * atr_abs
    elif s == "short":
        raw_tps = [entry - m * atr_abs for m in tp_mults]
        sl = entry + stop_mult * atr_abs
    else:
        return {"entry": q(entry), "atr_abs": atr_abs, "sl": None, "tps": []}
    tps = [q(x) for x in raw_tps]
    sl = q(sl)
    e = q(entry)
    if s == "long" and tps and tps[0] <= e:
        tps = [q(entry + m * abs(atr_abs)) for m in tp_mults]
    if s == "short" and tps and tps[0] >= e:
        tps = [q(entry - m * abs(atr_abs)) for m in tp_mults]
    return {"entry": e, "atr_abs": atr_abs, "sl": sl, "tps": tps}

def obi_ok_for_dir(direction: str, obi: float, thr: float = 0.10) -> bool:
    if direction == "long":  return obi >= thr
    if direction == "short": return obi <= -thr
    return False

# ============== STATE ===============

class DataHub:
    def __init__(self):
        self.candles = {}            # dict[symbol][tf] -> DataFrame
        self.last_signal_at = {}     # cooldown
        self.open_signals = []       # sinyal sonuç izlemesi

    def get_df(self, symbol: str, tf: str) -> pd.DataFrame:
        return self.candles.get(symbol, {}).get(tf, pd.DataFrame())

    def upsert_df(self, symbol: str, tf: str, df_new: pd.DataFrame):
        d = self.candles.setdefault(symbol, {})
        df_old = d.get(tf)
        if df_old is None or df_old.empty:
            d[tf] = df_new
            return
        merged = pd.concat([df_old, df_new]).drop_duplicates(subset=['ts']).sort_values('ts')
        d[tf] = merged

    def can_alert(self, symbol: str, now_ts: float, cooldown_min: int) -> bool:
        last = self.last_signal_at.get(symbol)
        if last is None: return True
        return (now_ts - last) >= cooldown_min * 60

    def mark_alert(self, symbol: str, now_ts: float):
        self.last_signal_at[symbol] = now_ts

# ============== DATA FETCHERS ==============

async def fetch_universe(exchange: ccxt.Exchange) -> list[str]:
    markets = await exchange.load_markets()
    syms = []
    for sym, m in markets.items():
        if m.get('type') == CONFIG["MARKET_TYPE"] and m.get('quote') == CONFIG["QUOTE"] and m.get('linear'):
            syms.append(sym)
    return syms

async def fetch_tickers_filtered(exchange: ccxt.Exchange, universe: list[str]) -> dict:
    tickers = await exchange.fetch_tickers(universe)
    rows = []
    for sym, tkr in tickers.items():
        vol = extract_quote_volume_usd(tkr)
        spr = spread_bps(tkr)
        if vol >= CONFIG["MIN_24H_VOL_USD"] and spr <= CONFIG["MAX_SPREAD_BPS"]:
            rows.append((sym, vol, spr, tkr))
    rows.sort(key=lambda x: x[1], reverse=True)
    selected = rows[:CONFIG["MAX_SYMBOLS"]]
    return {sym: tkr for (sym, _v, _s, tkr) in selected}

async def fetch_ohlcvs(exchange: ccxt.Exchange, symbols: list[str], timeframe: str, lookback: int) -> dict:
    out = {}
    for sym in symbols:
        try:
            o = await exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=lookback)
            out[sym] = o
            await asyncio.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.warning(f"{sym} {timeframe} fetch_ohlcv hata: {e}")
    return out

async def fetch_orderbooks(exchange: ccxt.Exchange, symbols: list[str], limit: int = 5) -> dict:
    out = {}
    for sym in symbols:
        try:
            ob = await exchange.fetch_order_book(sym, limit=limit)
            out[sym] = ob
            await asyncio.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            logger.warning(f"{sym} orderbook hata: {e}")
    return out

# ============== FEATURES & SCORER ==============

def compute_features(symbol: str, hub: DataHub):
    df1 = hub.get_df(symbol, '1m').copy()
    df5 = hub.get_df(symbol, '5m').copy()
    df15 = hub.get_df(symbol, '15m').copy()
    if any(df.empty for df in [df1, df5, df15]):
        return None
    for df in (df1, df5, df15):
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)

    # Trend: EMA50 vs EMA200 (5m & 15m)
    df5['ema_fast'] = ema(df5['close'], CONFIG["EMA_FAST"])
    df5['ema_slow'] = ema(df5['close'], CONFIG["EMA_SLOW"])
    df15['ema_fast'] = ema(df15['close'], CONFIG["EMA_FAST"])
    df15['ema_slow'] = ema(df15['close'], CONFIG["EMA_SLOW"])

    trend5_up = bool(df5['ema_fast'].iloc[-1] > df5['ema_slow'].iloc[-1])
    trend15_up = bool(df15['ema_fast'].iloc[-1] > df15['ema_slow'].iloc[-1])
    trend5_dn = bool(df5['ema_fast'].iloc[-1] < df5['ema_slow'].iloc[-1])
    trend15_dn = bool(df15['ema_fast'].iloc[-1] < df15['ema_slow'].iloc[-1])

    trend_dir = None
    trend_align = 0.0
    if trend5_up and trend15_up:
        trend_dir = "long"; trend_align = 1.0
    elif trend5_dn and trend15_dn:
        trend_dir = "short"; trend_align = 1.0
    else:
        trend_align = 0.0

    # ATR%(5m)
    df5['atr'] = atr(df5, CONFIG["ATR_PERIOD"])
    atr_pct_5m = float(df5['atr'].iloc[-1] / (df5['close'].iloc[-1] + 1e-12) * 100)

    # 1m hacim spike
    vol_med = float(df1['volume'].tail(CONFIG["VOL_SPIKE_WINDOW"]).median() or 0.0)
    vol_last = float(df1['volume'].iloc[-1] or 0.0)
    vol_spike = vol_last / (vol_med + 1e-12)

    # BB genişleme (1m)
    _u, _m, _l, width = bbands(df1['close'], CONFIG["BB_PERIOD"], CONFIG["BB_STD"])
    bb_width = float(width.iloc[-1] if not width.isna().iloc[-1] else 0.0)
    width_med = float(width.tail(CONFIG["BB_PERIOD"] * 3).median() if not width.tail(CONFIG["BB_PERIOD"]*3).isna().all() else 0.0)
    breakout_quality = max(0.0, min(bb_width / (width_med + 1e-12), 3.0))

    # RSI(1m) 50 kesişim ivmesi
    r = rsi(df1['close'], 14)
    r1, r2 = float(r.iloc[-1]), float(r.iloc[-2])
    rsi_impulse = 0.0
    crossed_up = (r2 < 50.0 and r1 >= 50.0)
    crossed_dn = (r2 > 50.0 and r1 <= 50.0)
    if crossed_up or crossed_dn:
        rsi_impulse = min(1.0, abs(r1 - 50.0) / 10.0)

    # VWAP (1m pencere)
    vwap = vwap_window(df1, 30)
    price_last = float(df1['close'].iloc[-1])
    above_vwap = price_last >= vwap

    return {
        "trend_dir": trend_dir,            # "long"/"short"/None
        "trend_align": trend_align,        # 0..1
        "atr_pct_5m": atr_pct_5m,          # %
        "vol_spike": vol_spike,            # x
        "breakout_quality": breakout_quality, # 0..3
        "rsi_impulse": rsi_impulse,        # 0..1
        "above_vwap": above_vwap,
        "price": price_last,
    }

def volatility_score_wrapper(atr_pct_5m: float) -> float:
    return volatility_sweet_score(atr_pct_5m)

def score_symbol(sym: str, feats: dict, tkr: dict, obi: float, spr_bps: float) -> tuple[float, dict]:
    # Ağırlıklar
    W = {
        "TrendAlign": 20,
        "VolatilitySweet": 15,
        "VolumeSpike1m": 20,
        "BreakoutQuality": 15,
        "OBIAligned": 10,
        "LowSpreadScore": 10,
        "RSIImpulse": 10,
    }
    trend_score = feats["trend_align"]
    vol_sweet = volatility_score_wrapper(feats["atr_pct_5m"])
    vol_spike_norm = min(1.0, feats["vol_spike"] / 2.5)
    breakout_norm = min(1.0, feats["breakout_quality"] / 2.0)

    # OBI yön hizalama: sadece doğru işaret katkı verir
    if feats.get("trend_dir") == "long":
        obi_aligned = max(0.0, obi)   # + yön
    elif feats.get("trend_dir") == "short":
        obi_aligned = max(0.0, -obi)  # - yön
    else:
        obi_aligned = 0.0
    obi_norm = min(1.0, obi_aligned)

    low_spread = max(0.0, 1.0 - (spr_bps / CONFIG["MAX_SPREAD_BPS"]))
    rsi_imp = feats["rsi_impulse"]

    total = (
        W["TrendAlign"] * trend_score
        + W["VolatilitySweet"] * vol_sweet
        + W["VolumeSpike1m"] * vol_spike_norm
        + W["BreakoutQuality"] * breakout_norm
        + W["OBIAligned"] * obi_norm
        + W["LowSpreadScore"] * low_spread
        + W["RSIImpulse"] * rsi_imp
    )
    detail = {
        "Breakout": round(breakout_norm, 2),
        "OBIAligned": round(obi_norm, 2),
    }
    return total, detail

def grade_from(score: float, aligned: bool, spread_bps_val: float,
               breakout_norm: float, vol_spike: float, atr_pct: float) -> str:
    if not aligned or spread_bps_val > CONFIG["MAX_SPREAD_BPS"]:
        return "C"
    base = "C"
    if score >= 68: base = "A+"
    elif score >= 60: base = "A"
    elif score >= 50: base = "B"
    else: base = "C"

    # Aşırı koşulları cezalandır
    if atr_pct < 0.12 or atr_pct > 1.50:
        if base == "A+": base = "A"
        elif base == "A": base = "B"

    if breakout_norm < 0.25 and base in ("A+", "A"):
        base = "B"

    return base

# ============== P(TP1) ONLINE STATS ==============

def load_stats(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

def save_stats(path: str, stats: dict):
    try:
        with open(path, "w") as f:
            json.dump(stats, f)
    except Exception as e:
        logger.warning(f"stats yazılamadı: {e}")

def update_ptp1_stats(stats: dict, grade: str, success: bool, window: int):
    g = stats.setdefault(grade, {"wins": 0, "loss": 0, "deque": []})
    # deque: 1=win, 0=loss (timeout dahil)
    dq = deque(g.get("deque", []), maxlen=window)
    dq.append(1 if success else 0)
    g["deque"] = list(dq)
    g["wins"] = int(sum(dq))
    g["loss"] = int(len(dq) - g["wins"])
    stats[grade] = g

def ptp1_of_grade(stats: dict, grade: str) -> tuple[str, int]:
    g = stats.get(grade)
    if not g: return "-", 0
    total = int(g.get("wins", 0) + g.get("loss", 0))
    if total <= 0: return "-", 0
    p = g["wins"] / total
    return f"{p*100:.0f}%", total

# ============== OUTCOME CHECKER ==================

def append_csv(path: str, row: dict, header: list):
    exists = os.path.exists(path)
    try:
        df = pd.DataFrame([row], columns=header)
        df.to_csv(path, mode="a", index=False, header=not exists)
    except Exception as e:
        logger.warning(f"CSV yazılamadı {path}: {e}")

def register_signal(hub: DataHub, symbol: str, direction: str, entry: float,
                    tp1: float, sl: float, grade: str):
    row = {
        "ts": int(time.time()*1000),
        "symbol": symbol,
        "dir": direction,
        "entry": entry,
        "tp1": tp1,
        "sl": sl,
        "grade": grade,
    }
    hub.open_signals.append({
        "ts": row["ts"],
        "symbol": symbol,
        "dir": direction,
        "entry": entry,
        "tp1": tp1,
        "sl": sl,
        "grade": grade,
        "timeout_s": 45*60,  # 45 dk
        "resolved": False,
    })
    append_csv(CONFIG["SIGNALS_CSV"], row,
               header=["ts","symbol","dir","entry","tp1","sl","grade"])

def try_resolve_signals(hub: DataHub, stats: dict):
    now_ms = int(time.time()*1000)
    for s in hub.open_signals:
        if s.get("resolved"): continue
        sym = s["symbol"]
        df = hub.get_df(sym, "1m")
        if df.empty: continue
        # Sinyalden SONRAKİ barlardan kontrol
        bars = df[df["ts"] >= s["ts"]].copy()
        if bars.empty: continue
        # İlk barı atla (aynı bar içi sıralama bilinmiyor)
        bars = bars.iloc[1:].reset_index(drop=True)
        hit = None
        for _, row in bars.iterrows():
            high = float(row["high"]); low = float(row["low"])
            if s["dir"] == "long":
                tp_hit = high >= s["tp1"]
                sl_hit = low <= s["sl"]
            else:
                tp_hit = low <= s["tp1"]
                sl_hit = high >= s["sl"]
            if tp_hit and sl_hit:
                # Aynı barda ikisi de — sıralama belirsiz → çözmeden devam et
                continue
            if tp_hit: hit = "TP1"; break
            if sl_hit: hit = "SL";  break

        timeout = (now_ms - s["ts"]) >= s["timeout_s"]*1000
        if hit or timeout:
            result = "TP1" if hit == "TP1" else ("SL" if hit == "SL" else "TIMEOUT")
            success = (result == "TP1")
            append_csv(CONFIG["OUTCOMES_CSV"], {
                "ts": now_ms, "symbol": sym, "dir": s["dir"],
                "entry": s["entry"], "tp1": s["tp1"], "sl": s["sl"],
                "grade": s["grade"], "result": result
            }, header=["ts","symbol","dir","entry","tp1","sl","grade","result"])
            update_ptp1_stats(stats, s["grade"], success, CONFIG["PTP1_WINDOW"])
            s["resolved"] = True

# ============== MAIN LOOP ==============

async def main():
    logger.info("Scalper başlıyor...")
    stats = load_stats(CONFIG["STATS_JSON"])

    exch = getattr(ccxt, CONFIG["EXCHANGE_ID"])({
        "options": {"defaultType": CONFIG["MARKET_TYPE"]},
        "enableRateLimit": True,
    })
    hub = DataHub()

    markets = await exch.load_markets()
    universe = [sym for sym, m in markets.items()
                if m.get("type") == CONFIG["MARKET_TYPE"]
                and m.get("quote") == CONFIG["QUOTE"]
                and m.get("linear")]
    if not universe:
        logger.error("Evren bulunamadı.")
        return
    logger.info(f"Evren ({CONFIG['MARKET_TYPE']} / {CONFIG['QUOTE']}): {len(universe)} sembol")

    last_1m_update = last_5m_update = last_15m_update = 0.0

    while True:
        loop_start = time.time()

        # 1) Likit evren + spread filtresi → top-N
        try:
            tickers_sel = await fetch_tickers_filtered(exch, universe)
        except Exception as e:
            logger.warning(f"fetch_tickers hata: {e}")
            await asyncio.sleep(2)
            continue

        symbols = list(tickers_sel.keys())

        # 2) OHLCV güncellemeleri
        now = time.time()
        if now - last_1m_update >= 55:
            o1 = await fetch_ohlcvs(exch, symbols, "1m", CONFIG["LOOKBACK_1M"])
            for sym, ohl in o1.items():
                hub.upsert_df(sym, "1m", ohlcv_to_df(ohl))
            last_1m_update = now

        if now - last_5m_update >= 290:
            o5 = await fetch_ohlcvs(exch, symbols, "5m", CONFIG["LOOKBACK_5M"])
            for sym, ohl in o5.items():
                hub.upsert_df(sym, "5m", ohlcv_to_df(ohl))
            last_5m_update = now

        if now - last_15m_update >= 890:
            o15 = await fetch_ohlcvs(exch, symbols, "15m", CONFIG["LOOKBACK_15M"])
            for sym, ohl in o15.items():
                hub.upsert_df(sym, "15m", ohlcv_to_df(ohl))
            last_15m_update = now

        # 3) Ön skor (OBI olmadan)
        prelim = []
        for sym, tkr in tickers_sel.items():
            feats = compute_features(sym, hub)
            if not feats: continue
            spr = spread_bps(tkr)
            tmp_score, _ = score_symbol(sym, feats, tkr, obi=0.0, spr_bps=spr)
            prelim.append((sym, tmp_score, feats, tkr))
        prelim.sort(key=lambda x: x[1], reverse=True)

        # 4) İlk 10 için orderbook ve final skor
        top_for_obi = [sym for (sym, _s, _f, _t) in prelim[:CONFIG["OBI_TOPK"]]]
        orderbooks = await fetch_orderbooks(exch, top_for_obi, CONFIG["ORDERBOOK_LIMIT"])

        final_rows = []  # (sym, score, detail, feats, tkr, obi, spr)
        for sym, tmp_score, feats, tkr in prelim:
            ob = orderbooks.get(sym)
            obi = orderbook_imbalance(ob) if ob else 0.0
            spr = spread_bps(tkr)
            score, detail = score_symbol(sym, feats, tkr, obi=obi, spr_bps=spr)
            final_rows.append((sym, score, detail, feats, tkr, obi, spr))
        final_rows.sort(key=lambda x: x[1], reverse=True)
        top10 = final_rows[:10]

        # 5) Konsol tablosu
        table = []
        recommended = []  # A/A+ olanlar

        for (sym, score, detail, feats, tkr, obi, spr) in top10:
            direction = (feats["trend_dir"] or "-").lower()
            aligned = (direction in ("long","short")) and obi_ok_for_dir(direction, obi, CONFIG["OBI_ALIGN_MIN"])

            tick = get_price_tick(exch.markets, sym)
            last = feats["price"]

            tp1 = tp2 = tp3 = sl = "-"
            if direction in ("long", "short"):
                exits = calc_exits(entry=last, atr_pct_5m=feats["atr_pct_5m"], side=direction,
                                   stop_mult=CONFIG["STOP_MULT"], tp_mults=CONFIG["TP_MULTS"], tick=tick)
                if len(exits["tps"]) == 3:
                    tp1_val, tp2_val, tp3_val = exits["tps"]
                    tp1, tp2, tp3 = (f"{tp1_val:.6g}", f"{tp2_val:.6g}", f"{tp3_val:.6g}")
                    sl = f"{exits['sl']:.6g}"
                else:
                    tp1_val = None
            else:
                tp1_val = None

            breakout_norm = min(1.0, feats["breakout_quality"] / 2.0)
            grade = grade_from(score, aligned, spr, breakout_norm, feats["vol_spike"], feats["atr_pct_5m"])

            # CONF ikonları
            conf = []
            if feats["vol_spike"] >= 1.5: conf.append("⚡")
            if breakout_norm >= 0.6:      conf.append("📈")
            if direction == "long" and feats["above_vwap"]: conf.append("Ⓥ")
            if direction == "short" and not feats["above_vwap"]: conf.append("Ⓥ")
            if feats["atr_pct_5m"] < 0.12 or feats["atr_pct_5m"] > 1.50: conf.append("Ⓡ")
            conf_str = " ".join(conf) if conf else "·"

            # P(TP1) (grade bazlı)
            p_str, n_cnt = ptp1_of_grade(stats, grade)

            # ΔTP1%
            prox = "-"
            if tp1_val is not None:
                if direction == "long":
                    prox_val = (tp1_val - last) / last * 100
                else:
                    prox_val = (last - tp1_val) / last * 100
                prox = f"{prox_val:.2f}%"

            row = [
                sym.replace(":USDT",""),
                f"{score:.0f}",
                direction if direction != "-" else "-",
                f"{feats['atr_pct_5m']:.2f}%",
                f"{feats['vol_spike']:.2f}x",
                f"{detail['Breakout']:.2f}",
                f"{obi:+.3f}",
                "✔" if aligned else "·",
                f"{spr:.2f}bp",
                f"{last:.6g}",
                tp1, tp2, tp3,
                sl,
                prox,
                grade,
                p_str,
                n_cnt
            ]
            table.append(row)

            # --- SİNYAL KARARI ---
            should_signal = (
                grade in ("A","A+") and
                score >= CONFIG["SIGNAL_THRESHOLD"] and
                aligned and
                spr <= CONFIG["MAX_SPREAD_BPS"] and
                tp1_val is not None
            )

            # OBI minimum hizalama (kontra işlemi engelle)
            if should_signal and not obi_ok_for_dir(direction, obi, CONFIG["OBI_ALIGN_MIN"]):
                should_signal = False
                conf_str = (conf_str + " Ⓧ").strip()

            if should_signal and hub.can_alert(sym, time.time(), CONFIG["COOLDOWN_MIN"]):
                # Sinyali kaydet ve bildir
                register_signal(hub, sym, direction, last, float(tp1), float(sl), grade)
                hub.mark_alert(sym, time.time())

                msg = (
                    f"✅ SİNYAL ({grade}) {direction.upper()}\n"
                    f"{sym}\n"
                    f"Score {score:.0f} | Spread {spr:.2f}bp | OBI {obi:+.3f}\n"
                    f"ATR%5m {feats['atr_pct_5m']:.2f}% | VolSpike {feats['vol_spike']:.2f}x | Breakout {detail['Breakout']:.2f}\n"
                    f"Last {last:.6g} | TP1 {tp1} | TP2 {tp2} | TP3 {tp3} | SL {sl}"
                )
                asyncio.create_task(send_telegram(msg))

                if grade in ("A","A+"):
                    recommended.append(row)

        # Çıktı
        print("\n" + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
        print(tabulate(
            table,
            headers=["Symbol","Score","Dir","ATR%5m","VolSpike","Breakout","OBI","Align","Spread","Last",
                     "TP1","TP2","TP3","SL","ΔTP1%","Grade","P(TP1)","N"],
            tablefmt="pretty"
        ))

        # Önerilenler (A/A+)
        rec = [r for r in table if r[15] in ("A","A+")]
        if rec:
            print("\n[RECOMMENDED ≥ A]")
            print(tabulate(
                rec,
                headers=["Symbol","Score","Dir","ATR%5m","VolSpike","Breakout","OBI","Align","Spread","Last",
                         "TP1","TP2","TP3","SL","ΔTP1%","Grade","P(TP1)","N"],
                tablefmt="pretty"
            ))

        # 6) Açık sinyallerin sonuçlarını değerlendir
        try_resolve_signals(hub, stats)
        save_stats(CONFIG["STATS_JSON"], stats)

        # 7) Balanced Mode uyku
        elapsed = time.time() - loop_start
        sleep_for = max(0.0, CONFIG["SNAPSHOT_SEC"] - elapsed)
        await asyncio.sleep(sleep_for)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Çıkılıyor...")
