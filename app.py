#!/usr/bin/env python3
"""Kronos Forecast — local web UI. Start with the launcher (Start Kronos.bat / start.command / start.sh)."""
from __future__ import annotations

import io
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import updater
from kronos_local import MODELS, Kronos, KronosPredictor, KronosTokenizer, Market, load_csv, load_market, plan_future, seconds

HERE = Path(__file__).resolve().parent
INDEX = HERE / "ui" / "index.html"
DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE.startswith("cuda") else {"mps": "Apple GPU (Metal)", "cpu": "CPU"}[DEVICE]
MAX_STEPS = 500
SHOW_PATHS = 40
PORT_FILE = HERE / ".kronos-port"
RESTART_CODE = 3  # tells the launcher to start the app again (after a Kronos update)

_predictors: dict[str, KronosPredictor] = {}
_lock = threading.Lock()  # one forecast at a time keeps GPU memory predictable
_server: ThreadingHTTPServer | None = None
_restart = threading.Event()
_status_cache: dict = {"kronos": (0.0, None), "app": (0.0, None)}


def predictor(name: str) -> KronosPredictor:
    if name not in _predictors:
        model_id, tok_id, ctx = MODELS[name]
        print(f"[model] loading Kronos-{name} on {DEVICE}...", flush=True)
        _predictors[name] = KronosPredictor(Kronos.from_pretrained(model_id), KronosTokenizer.from_pretrained(tok_id), device=DEVICE, max_context=ctx)
    return _predictors[name]


class UserError(Exception):
    pass


def forecast(req: dict) -> dict:
    sym = str(req.get("symbol", "")).strip().upper()
    interval, horizon = req.get("interval", "5m"), req.get("horizon", "1h")
    n_paths = max(4, min(int(req.get("paths", 50)), 500))
    name = req.get("model", "small")
    if name not in MODELS:
        raise UserError(f"Unknown model {name}")
    iv, hz = seconds(interval), seconds(horizon)
    if hz < iv:
        raise UserError(f"'Predict next {horizon}' is shorter than one {interval} candle.")
    ctx = MODELS[name][2]
    lookback = min(int(req.get("lookback") or 512), ctx)

    t0 = time.time()
    if req.get("csv"):
        mk = Market(load_csv(io.StringIO(req["csv"]), lookback), "your CSV file", "UTC", False)
    else:
        mk = load_market(sym, interval, lookback)
    df = mk.df
    t_data = time.time() - t0

    future = plan_future(df["timestamps"], iv, hz, mk.always_open)
    steps = len(future)
    if steps > MAX_STEPS:
        raise UserError(f"That's {steps} candles ahead; the limit is {MAX_STEPS}. Pick a bigger candle interval or a shorter forecast.")
    x = df[["open", "high", "low", "close", "volume"]]
    x_ts = df["timestamps"]
    y_ts = pd.Series(future)

    with _lock:
        pred = predictor(name)
        torch.manual_seed(int(req.get("seed", 0)))
        t1 = time.time()
        paths, chunk = [], 64 if DEVICE != "cpu" else 8
        for i in range(0, n_paths, chunk):
            k = min(chunk, n_paths - i)
            out = pred.predict_batch([x] * k, [x_ts] * k, [y_ts] * k, pred_len=steps, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            paths += [o["close"].to_numpy() for o in out]
        t_model = time.time() - t1
    paths = np.array(paths)

    ts = lambda s: int(pd.Timestamp(s).tz_localize("UTC").timestamp())
    pick = np.linspace(0, n_paths - 1, min(SHOW_PATHS, n_paths)).astype(int)
    return {
        "symbol": sym or "CSV", "source": mk.source, "tz": mk.tz, "always_open": mk.always_open, "interval": interval, "horizon": horizon, "steps": int(steps),
        "model": f"Kronos-{name}", "device": DEVICE_NAME, "paths_n": n_paths,
        "timing": {"data": round(t_data, 2), "model": round(t_model, 2)},
        "candles": [{"time": ts(r.timestamps), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume} for r in df.itertuples()],
        "future": [ts(t) for t in future],
        "p10": np.percentile(paths, 10, 0).tolist(), "p25": np.percentile(paths, 25, 0).tolist(),
        "p50": np.median(paths, 0).tolist(), "p75": np.percentile(paths, 75, 0).tolist(), "p90": np.percentile(paths, 90, 0).tolist(),
        "ends": paths[:, -1].tolist(),
        "sample_paths": paths[pick].tolist(),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._json(200, {"device": DEVICE_NAME, "gpu": DEVICE != "cpu", "models": list(MODELS), "loaded": list(_predictors),
                             "kronos": updater.version_of(updater.KRONOS_DIR), "app": updater.app_info().get("version")})
        elif self.path.startswith(("/api/kronos/status", "/api/app/status")):
            kind = self.path.split("/")[2]
            at, data = _status_cache[kind]
            if "refresh=1" in self.path or not data or time.time() - at > 600:  # GitHub allows 60 API checks an hour
                data = updater.status() if kind == "kronos" else updater.app_status()
                _status_cache[kind] = (time.time(), data)
            self._json(200, data)
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path in ("/api/kronos/update", "/api/kronos/undo", "/api/app/update", "/api/app/undo"):
            _, _, kind, action = self.path.split("/")
            return self._change(kind, action)
        if self.path != "/api/forecast":
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            print(f"[run] {req.get('symbol') or 'CSV'} {req.get('interval')} -> next {req.get('horizon')} ({req.get('model')}, {req.get('paths')} paths)", flush=True)
            res = forecast(req)
            print(f"[run] done: data {res['timing']['data']}s, model {res['timing']['model']}s", flush=True)
            self._json(200, res)
        except (UserError, SystemExit) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # exchange/network errors from ccxt land here
            traceback.print_exc()
            msg = str(e).split("\n")[0][:300]
            name = type(e).__name__
            if "451" in msg or "restricted location" in msg.lower():
                msg = "This exchange blocks your country (HTTP 451). Try BYBIT: or OKX:, or upload a CSV exported from TradingView."
            elif name in ("NetworkError", "RequestTimeout", "ExchangeNotAvailable"):
                msg = f"Couldn't reach the exchange ({name}). Check your internet connection or try another exchange."
            elif name == "BadSymbol":
                msg = f"The exchange doesn't list that pair: {msg}"
            self._json(500, {"error": msg})


    def _change(self, kind: str, action: str) -> None:
        run = {("kronos", "update"): updater.update, ("kronos", "undo"): updater.undo,
               ("app", "update"): updater.app_update, ("app", "undo"): updater.app_undo}[(kind, action)]
        try:
            with _lock:  # waits for a running forecast to finish
                res = run()
        except updater.UpdateError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})
        self._json(200, {**res, "restarting": True})
        _restart.set()
        threading.Thread(target=_server.shutdown, daemon=True).start()


def free_port(start: int = 8765) -> int:
    for p in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise SystemExit("No free port found")


def main() -> None:
    global _server
    port = free_port(int(os.environ.get("KRONOS_PORT") or 8765))
    _server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    PORT_FILE.write_text(str(port))
    updater.JUST_UPDATED.unlink(missing_ok=True)  # started fine, so the launcher needn't roll back
    url = f"http://127.0.0.1:{port}/"
    ver = updater.version_of(updater.KRONOS_DIR)
    print(f"\n  Kronos Forecast {updater.app_info().get('version')} is running on {DEVICE_NAME}  (Kronos code {ver[:7] if ver else 'version unknown'})")
    print(f"  Open {url} in your browser" + ("." if os.environ.get("KRONOS_NO_BROWSER") else " (it should open by itself)."))
    print("  Keep this window open while you use it. Close it to stop.\n", flush=True)
    if not os.environ.get("KRONOS_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        _server.serve_forever()
    except KeyboardInterrupt:
        pass
    _server.server_close()
    sys.exit(RESTART_CODE if _restart.is_set() else 0)


if __name__ == "__main__":
    main()
