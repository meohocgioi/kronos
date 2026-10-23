#!/usr/bin/env python3
"""Run a Kronos price forecast locally (GPU if available) for crypto, stocks, forex, indices or futures.

Examples:
    python kronos_local.py BINANCE:BTCUSDT.P --interval 5m --horizon 1h
    python kronos_local.py HOSE:HPG --interval 1d --horizon 1w
    python kronos_local.py NASDAQ:AAPL --interval 15m --horizon 4h --paths 100
    python kronos_local.py FX:EURUSD --interval 1h --horizon 1d --model base

Symbols use TradingView style EXCHANGE:TICKER. Crypto exchanges are read through ccxt
(a trailing .P means the USDT/USDC perpetual); everything else comes from Yahoo Finance.
A bare Yahoo ticker (HPG.VN, AAPL, EURUSD=X, GC=F, ^GSPC) also works. No API keys needed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("KRONOS_CODE_DIR") or str(HERE / "Kronos"))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

MODELS = {
    "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}
EXCHANGES = {"BINANCE": "binance", "BYBIT": "bybit", "OKX": "okx", "COINBASE": "coinbase",
             "KRAKEN": "kraken", "BITGET": "bitget", "MEXC": "mexc", "KUCOIN": "kucoin", "GATEIO": "gateio"}
QUOTES = ["USDT", "USDC", "FDUSD", "USD", "EUR", "BTC", "ETH"]
UNITS = {"m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}

# TradingView exchange prefix -> Yahoo Finance ticker suffix
YAHOO_SUFFIX = {"HOSE": ".VN", "TSX": ".TO", "TSXV": ".V", "LSE": ".L", "XETR": ".DE", "FWB": ".F", "TSE": ".T",
                "ASX": ".AX", "NSE": ".NS", "BSE": ".BO", "SSE": ".SS", "SZSE": ".SZ", "KRX": ".KS", "SGX": ".SI",
                "TWSE": ".TW", "SET": ".BK", "IDX": ".JK", "BMFBOVESPA": ".SA", "SIX": ".SW", "MIL": ".MI",
                "BME": ".MC", "OMXSTO": ".ST"}
US_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSEARCA", "ARCA", "BATS", "OTC"}
FX_FEEDS = {"FX", "OANDA", "FX_IDC", "FOREXCOM", "SAXO", "PEPPERSTONE", "ICMARKETS", "FXCM", "CAPITALCOM", "TVC"}
INDEX_ALIASES = {"SPX": "^GSPC", "NDX": "^NDX", "DJI": "^DJI", "VIX": "^VIX", "VNINDEX": "^VNINDEX.VN",
                 "NI225": "^N225", "HSI": "^HSI", "DAX": "^GDAXI", "UKX": "^FTSE", "DXY": "DX-Y.NYB"}
# Spot metals/oil aren't on Yahoo; the front-month future stands in for them
STAND_INS = {"XAUUSD": ("GC=F", "gold futures"), "GOLD": ("GC=F", "gold futures"), "XAGUSD": ("SI=F", "silver futures"),
             "SILVER": ("SI=F", "silver futures"), "USOIL": ("CL=F", "WTI crude futures"), "WTICOUSD": ("CL=F", "WTI crude futures"),
             "UKOIL": ("BZ=F", "Brent crude futures"), "BCOUSD": ("BZ=F", "Brent crude futures")}
# app interval -> (Yahoo interval, how far back Yahoo serves it)
YAHOO_IV = {"1m": ("1m", "7d"), "5m": ("5m", "60d"), "15m": ("15m", "60d"), "30m": ("30m", "60d"),
            "1h": ("60m", "730d"), "4h": ("60m", "730d"), "1d": ("1d", "10y"), "1w": ("1wk", "max")}


@dataclass
class Market:
    df: pd.DataFrame   # timestamps are the exchange's local wall-clock time, tz-naive
    source: str        # shown to the user, e.g. "Yahoo Finance HPG.VN"
    tz: str            # e.g. "UTC" or "UTC+07:00"
    always_open: bool  # True for 24/7 crypto; otherwise future candles follow the trading hours seen in history


def seconds(s: str) -> int:
    m = re.fullmatch(r"(\d+)([mhdwM])", s.strip() if s[-1] == "M" else s.strip().lower())
    if not m:
        raise SystemExit(f"Bad duration '{s}'. Use forms like 1m, 5m, 15m, 1h, 4h, 1d, 1w.")
    return int(m.group(1)) * UNITS[m.group(2)]


def parse_symbol(tv: str) -> tuple[ccxt.Exchange, str]:
    if ":" not in tv:
        raise SystemExit("Write the pair as EXCHANGE:TICKER, e.g. BINANCE:BTCUSDT.P")
    ex_name, ticker = tv.upper().split(":", 1)
    if ex_name not in EXCHANGES:
        raise SystemExit(f"{ex_name} is not a crypto exchange this app reads directly")
    perp = ticker.endswith(".P")
    ticker = ticker.removesuffix(".P")
    quote = next((q for q in QUOTES if ticker.endswith(q) and len(ticker) > len(q)), None)
    if not quote:
        raise SystemExit(f"Can't split {ticker} into base/quote (known quotes: {', '.join(QUOTES)})")
    base = ticker[: -len(quote)]
    ex = getattr(ccxt, EXCHANGES[ex_name])({"enableRateLimit": True})
    return ex, f"{base}/{quote}:{quote}" if perp else f"{base}/{quote}"


def fetch(ex: ccxt.Exchange, sym: str, tf: str, n: int) -> pd.DataFrame:
    ex.load_markets()
    if sym not in ex.markets:
        raise SystemExit(f"{ex.id} has no market {sym}")
    tf_ms = seconds(tf) * 1000
    since = ex.milliseconds() - (n + 2) * tf_ms
    rows: list = []
    while True:
        batch = ex.fetch_ohlcv(sym, tf, since=since, limit=min(n, 1000))
        if not batch:
            break
        rows += [r for r in batch if not rows or r[0] > rows[-1][0]]
        since = batch[-1][0] + tf_ms
        if len(batch) < 2 or since > ex.milliseconds():
            break
    if len(rows) < 64:
        raise SystemExit(f"Only {len(rows)} candles returned for {sym} {tf}")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["ts"], unit="ms")
    return df.iloc[:-1].tail(n).reset_index(drop=True)  # drop the still-forming candle


def to_yahoo(tv: str) -> tuple[str, str]:
    """TradingView-style symbol -> (Yahoo ticker, note shown next to it)."""
    if ":" not in tv:
        return tv, ""
    ex, tk = tv.split(":", 1)
    if ex == "YAHOO":
        return tk, ""
    if tk in INDEX_ALIASES:
        return INDEX_ALIASES[tk], ""
    if tk in STAND_INS and (ex in FX_FEEDS or ex in {"COMEX", "NYMEX"}):
        y, what = STAND_INS[tk]
        return y, f"{what}, standing in for {tk}"
    if re.fullmatch(r"[A-Z0-9]+[12]!", tk):  # continuous futures like GC1!, ES1!
        return tk[:-2] + "=F", "front-month future"
    if ex in FX_FEEDS:
        return tk + "=X", ""
    if ex in US_EXCHANGES:
        return tk.replace(".", "-"), ""
    if ex == "HKEX":
        return tk.zfill(4) + ".HK", ""
    if ex in YAHOO_SUFFIX:
        return tk + YAHOO_SUFFIX[ex], ""
    if ex in ("HNX", "UPCOM"):
        raise SystemExit(f"Yahoo Finance doesn't carry {ex} stocks. Export the chart from TradingView and use a CSV file instead.")
    raise SystemExit(f"Unknown exchange {ex}. Use a crypto exchange ({', '.join(EXCHANGES)}), a stock exchange "
                     f"(HOSE, NASDAQ, NYSE, LSE, TSE, HKEX, ...), FX:EURUSD, or type a Yahoo Finance ticker such as "
                     f"HPG.VN, AAPL, EURUSD=X, GC=F or ^GSPC.")


def fetch_yahoo(ticker: str, tf: str, n: int) -> tuple[pd.DataFrame, str]:
    import yfinance as yf

    if tf not in YAHOO_IV:
        raise SystemExit(f"Interval {tf} isn't available for stocks/forex. Use one of {', '.join(YAHOO_IV)}.")
    yiv, period = YAHOO_IV[tf]
    t = yf.Ticker(ticker)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # yfinance's own pandas deprecation noise
        d = t.history(period=period, interval=yiv, auto_adjust=False, actions=False)
    if d is None or d.empty:
        raise SystemExit(f"Yahoo Finance returned no {tf} candles for {ticker}. Check the ticker on finance.yahoo.com.")
    tz = d.index.tz
    off = d.index[-1].strftime("%z") or "+0000"
    d.index = d.index.tz_localize(None)  # exchange-local wall clock
    d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    if tf == "4h":
        d = d.resample("4h", origin="start_day").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    if tf == "1d" and pd.isna(d["close"].iloc[-1]):
        # Yahoo often leaves today's daily close blank for a while after the session; use the last price
        px = (t.history_metadata or {}).get("regularMarketPrice")
        if px:
            i = d.index[-1]
            d.loc[i, "close"] = px
            d.loc[i, "high"] = max(d.loc[i, "high"], px)
            d.loc[i, "low"] = min(d.loc[i, "low"], px)
    d = d.dropna(subset=["open", "high", "low", "close"])
    d["volume"] = d["volume"].fillna(0.0)

    # drop the still-forming candle
    now = pd.Timestamp.now(tz=tz).tz_localize(None)
    end = d.index[-1] + pd.Timedelta(seconds=seconds(tf))
    if tf == "1d":
        try:
            reg_end = t.history_metadata["currentTradingPeriod"]["regular"]["end"]
            # newer yfinance hands back a tz-aware Timestamp, older ones raw epoch seconds
            reg_end = pd.Timestamp(reg_end) if isinstance(reg_end, datetime) else pd.Timestamp(reg_end, unit="s", tz="UTC")
            end = reg_end.tz_convert(tz).tz_localize(None)
            if end.date() != d.index[-1].date():
                end = d.index[-1]  # today's session is already over
        except Exception:
            pass
    if end > now:
        d = d.iloc[:-1]

    d = d.tail(n).reset_index(names="timestamps")
    if len(d) < 64:
        raise SystemExit(f"Yahoo Finance has only {len(d)} {tf} candles for {ticker}; Kronos needs at least 64. "
                         f"Try another interval, or export the chart from TradingView and use a CSV file.")
    return d, f"UTC{off[:3]}:{off[3:]}"


def load_market(symbol: str, tf: str, n: int) -> Market:
    sym = symbol.strip().upper()
    if ":" in sym and sym.split(":", 1)[0] in EXCHANGES:
        ex, msym = parse_symbol(sym)
        return Market(fetch(ex, msym, tf, n), f"{ex.id} {msym}", "UTC", True)
    ticker, note = to_yahoo(sym)
    df, tz = fetch_yahoo(ticker, tf, n)
    return Market(df, f"Yahoo Finance {ticker}" + (f" ({note})" if note else ""), tz, False)


def plan_future(ts: pd.Series, iv: int, hz: int, always_open: bool) -> pd.DatetimeIndex:
    """Timestamps of the candles to forecast.

    24/7 markets: horizon / interval candles. Markets with trading hours: future candles only land on
    weekday + time-of-day slots seen in the history (so nights, weekends and lunch breaks are skipped).
    There a horizon under a day counts trading time (next 1h = the next hour the market is open) and a
    horizon of a day or more counts trading days (1w = 5 sessions for a Mon-Fri market).
    """
    if hz < iv:
        raise SystemExit("Pick a forecast length at least one candle long.")
    last = ts.iloc[-1]
    if always_open or iv >= 604800:
        steps = max(1, round(hz / iv))
        return pd.date_range(last + pd.Timedelta(seconds=iv), periods=steps, freq=f"{iv}s")
    recent = ts.iloc[-3000:]
    days = set(recent.dt.weekday)
    if iv >= 86400:
        ok = lambda t: t.weekday() in days
        per_day = 1
    else:
        slots = set(zip(recent.dt.weekday, recent.dt.hour, recent.dt.minute))
        ok = lambda t: (t.weekday(), t.hour, t.minute) in slots
        per_day = int(recent.dt.date.value_counts().median())
    if hz < 86400:
        steps = max(1, round(hz / iv))
    else:
        steps = max(1, round(hz / 86400 * len(days) / 7)) * per_day
    out, t, step = [], last, pd.Timedelta(seconds=iv)
    for _ in range(steps * 5000 + 10000):
        if len(out) == steps:
            break
        t += step
        if ok(t):
            out.append(t)
    return pd.DatetimeIndex(out)


def load_csv(path: str, n: int) -> pd.DataFrame:
    """CSV with a time column (unix seconds/ms or ISO date) plus open, high, low, close[, volume].
    A TradingView 'Export chart data' CSV works as-is."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("timestamps", "time", "date", "datetime", "ts") if c in df.columns), None)
    if not tcol or not {"open", "high", "low", "close"} <= set(df.columns):
        raise SystemExit("CSV needs a time column and open, high, low, close columns")
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        df["timestamps"] = pd.to_datetime(t, unit="ms" if t.iloc[-1] > 1e11 else "s")
    else:
        df["timestamps"] = pd.to_datetime(t, utc=True).dt.tz_localize(None)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["timestamps", "open", "high", "low", "close", "volume"]].sort_values("timestamps").tail(n).reset_index(drop=True)


def write_html(path: Path, title: str, df: pd.DataFrame, future: pd.DatetimeIndex, paths: np.ndarray) -> None:
    t = lambda s: int(pd.Timestamp(s).tz_localize("UTC").timestamp())
    hist = df.tail(300)
    data = {
        "title": title,
        "candles": [{"time": t(r.timestamps), "open": r.open, "high": r.high, "low": r.low, "close": r.close} for r in hist.itertuples()],
        "future": [t(x) for x in future],
        "p10": np.percentile(paths, 10, 0).tolist(), "p50": np.median(paths, 0).tolist(), "p90": np.percentile(paths, 90, 0).tolist(),
        "last": float(df["close"].iloc[-1]), "lastT": t(df["timestamps"].iloc[-1]),
    }
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{margin:0;background:#0f1218;color:#e6e9ef;font:14px system-ui}}h1{{font-size:16px;margin:12px 16px}}#c{{height:85vh}}</style>
<h1>{title}</h1><div id="c"></div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>const d={json.dumps(data)};const ch=LightweightCharts.createChart(document.getElementById('c'),{{autoSize:true,
layout:{{background:{{color:'#0f1218'}},textColor:'#8a93a3'}},grid:{{vertLines:{{color:'#262c37'}},horzLines:{{color:'#262c37'}}}},
timeScale:{{timeVisible:true}}}});ch.addCandlestickSeries({{upColor:'#26c08b',downColor:'#f0616d',wickUpColor:'#26c08b',wickDownColor:'#f0616d',borderVisible:false}}).setData(d.candles);
const j=a=>[{{time:d.lastT,value:d.last}},...d.future.map((t,i)=>({{time:t,value:a[i]}}))];
for(const [k,w,s] of [['p10',1,2],['p90',1,2],['p50',2,0]])ch.addLineSeries({{color:'#6b9bff',lineWidth:w,lineStyle:s,title:k}}).setData(j(d[k]));
ch.timeScale().setVisibleRange({{from:d.lastT-(d.future.at(-1)-d.lastT)*3,to:d.future.at(-1)}});</script>""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol", help="TradingView-style pair, e.g. BINANCE:BTCUSDT.P")
    ap.add_argument("--interval", default="5m", help="candle size: 1m 5m 15m 30m 1h 4h 1d 1w (default 5m)")
    ap.add_argument("--horizon", default="1h", help="how far ahead to predict, e.g. 15m 1h 4h 1d (default 1h)")
    ap.add_argument("--paths", type=int, default=50, help="number of sampled forecast paths (default 50)")
    ap.add_argument("--model", choices=MODELS, default="small", help="Kronos model size (default small)")
    ap.add_argument("--lookback", type=int, default=None, help="history candles fed to the model (default = model context)")
    ap.add_argument("--device", default=None, help="cuda, cuda:1, mps or cpu (default: auto)")
    ap.add_argument("--html", default=None, help="also write an interactive chart to this .html file")
    ap.add_argument("--csv", default=None, help="read candles from this CSV instead of the exchange")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    iv, hz = seconds(a.interval), seconds(a.horizon)
    model_id, tok_id, ctx = MODELS[a.model]
    lookback = min(a.lookback or ctx, ctx)
    device = a.device or ("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if a.csv:
        mk = Market(load_csv(a.csv, lookback), f"CSV {Path(a.csv).name}", "UTC", False)
    else:
        print(f"[data] {a.symbol} {a.interval} x {lookback} candles...", file=sys.stderr)
        mk = load_market(a.symbol, a.interval, lookback)
    df, src = mk.df, mk.source
    future = plan_future(df["timestamps"], iv, hz, mk.always_open)
    steps = len(future)

    print(f"[model] Kronos-{a.model} on {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.startswith("cuda") else ""), file=sys.stderr)
    torch.manual_seed(a.seed)
    pred = KronosPredictor(Kronos.from_pretrained(model_id), KronosTokenizer.from_pretrained(tok_id), device=device, max_context=ctx)

    x = df[["open", "high", "low", "close", "volume"]]
    x_ts = df["timestamps"]
    y_ts = pd.Series(future)

    t0 = time.time()
    paths = []
    chunk = 64 if device != "cpu" else 8  # paths per GPU batch; lower if you run out of VRAM
    for i in range(0, a.paths, chunk):
        k = min(chunk, a.paths - i)
        out = pred.predict_batch([x] * k, [x_ts] * k, [y_ts] * k, pred_len=steps, T=1.0, top_p=0.9, sample_count=1, verbose=False)
        paths += [o["close"].to_numpy() for o in out]
    paths = np.array(paths)
    print(f"[model] {a.paths} paths x {steps} steps in {time.time() - t0:.1f}s", file=sys.stderr)

    last = float(df["close"].iloc[-1])
    end = paths[:, -1]
    p = lambda q: np.percentile(end, q)
    pct = lambda v: f"{(v / last - 1) * 100:+.2f}%"
    dec = 1 if last >= 1000 else 2 if last >= 100 else 4 if last >= 0.01 else 6
    f = lambda v: f"{v:,.{dec}f}"
    up = int((end > last).sum())

    print(f"\n{a.symbol}  ({src})  {a.interval} candles, forecast next {a.horizon} = {steps} candles, Kronos-{a.model}, {a.paths} paths")
    print(f"Last closed candle : {df['timestamps'].iloc[-1]} {mk.tz}  close {f(last)}")
    print(f"{'Median at +' + a.horizon:<19}: {f(np.median(end))}  ({pct(np.median(end))})")
    print(f"10-90% range       : {f(p(10))} - {f(p(90))}  ({pct(p(10))} to {pct(p(90))})")
    print(f"Paths ending up    : {up}/{a.paths}")
    print(f"\n  time ({mk.tz})     p10            median         p90")
    every = max(1, math.ceil(steps / 12))
    for i in list(range(every - 1, steps, every)) + ([steps - 1] if (steps - 1) % every != every - 1 else []):
        col = paths[:, i]
        print(f"  {future[i]:%m-%d %H:%M}   {f(np.percentile(col, 10)):>13}  {f(np.median(col)):>13}  {f(np.percentile(col, 90)):>13}")
    print("\nResearch output, not financial advice.")

    if a.html:
        write_html(Path(a.html), f"{a.symbol} · {a.interval} · next {a.horizon} · Kronos-{a.model}", df, future, paths)
        print(f"Chart written to {Path(a.html).resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
