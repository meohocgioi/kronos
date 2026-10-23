# Kronos Forecast: a local app for your GPU

A point-and-click app that runs the [Kronos](https://github.com/shiyu-coder/Kronos) forecasting model on your own computer, using your GPU. It forecasts **crypto, stocks, forex, indices and futures**. Type a symbol such as `BINANCE:BTCUSDT.P`, `HOSE:HPG` or `NASDAQ:AAPL`, choose the candle interval and how far ahead to predict, then press **Run forecast**. You get an interactive chart with the model's forecast range.

You never need to type commands. A launcher you double-click handles setup and starting the app.

## Updates

The **Updates** badge at the top right of the page checks GitHub each time the page opens. It covers two things:

- **The app itself** (this repository, [meohocgioi/kronos](https://github.com/meohocgioi/kronos)).
- **The Kronos model code** (the authors' repository, [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)).

The badge says **"v1.3.0 · up to date"** when there's nothing new. It turns blue (**"Update available"**) when either one has a newer version. Click it to see what changed, then press **Update the app** or **Update Kronos**.

An update never breaks a working setup:
1. The new version is downloaded into a separate folder and test-run with a small forecast.
2. Only if that works does the app switch over and restart, which takes about a minute. The page reloads by itself.
3. If the test fails, the update is refused and nothing changes.
4. If an updated app ever fails to start, the launcher puts the previous version back by itself.
5. The previous version is kept, and **Undo last update** in the same panel goes back to it.

The model weights update on their own: each time a model loads, it checks Hugging Face for newer weights.

**The start files are never auto-updated.** `Start Kronos (Windows).bat` and the other start scripts are left alone, because Windows misbehaves when a running .bat file is rewritten. They rarely change; if one ever does, the release notes will say so.

**Coming from a version before 1.3.0?** Older versions have no app update button, so update by hand once: download the ZIP from GitHub (**Code → Download ZIP**) and unzip it over your existing folder, letting it replace the files. Your `.venv` and `Kronos` folders aren't in the ZIP, so they stay as they are. After that, the button handles updates.

If the Kronos section shows your installed version as "unknown", your install predates version tracking. Updating Kronos once fixes that.

## What you need

- **A GPU.** Either an NVIDIA graphics card with a recent driver, or an Apple Silicon Mac. Without one the app still works on the CPU, just more slowly.
- **Python 3.10 or newer**, from <https://www.python.org/downloads/>. **On Windows, tick "Add python.exe to PATH"** on the first screen of the installer.
- **About 4 GB of free disk** and an internet connection for the first start.

## Start it

1. Download the app from <https://github.com/meohocgioi/kronos> (**Code → Download ZIP**) and unzip it anywhere, for example your Desktop.
2. Double-click the launcher for your system:
   - **Windows:** `Start Kronos (Windows).bat`
   - **Mac:** `Start Kronos (Mac).command`. If macOS says it can't be opened, right-click it, choose **Open**, then **Open** again.
   - **Linux:** `start-linux.sh`
3. **The first start takes 5–15 minutes.** A window shows the progress while the launcher:
   - creates a private Python environment inside the folder,
   - checks for an NVIDIA GPU and installs the matching GPU version of PyTorch (about 2–3 GB),
   - installs the other packages and downloads the Kronos code from GitHub.
4. Your browser opens **http://127.0.0.1:8765**. The top-right corner should say **Running on** followed by your GPU's name.

Later starts take a few seconds. **Keep the black launcher window open** while you use it; closing it stops the app.

## What you can forecast

Type symbols the way TradingView writes them. The app works out where to get the candles:

| Market | Examples | Data comes from |
|---|---|---|
| **Crypto** | `BINANCE:BTCUSDT.P` (perpetual), `BYBIT:RLCUSDT.P`, `OKX:ETHUSDT.P`, `COINBASE:BTCUSD` | The exchange itself (Binance, Bybit, OKX, Coinbase, Kraken, Bitget, MEXC, KuCoin, Gate.io) |
| **Vietnam (HOSE)** | `HOSE:HPG`, `HOSE:FPT`, `HOSE:VCB`, `HOSE:VNM` | Yahoo Finance |
| **US stocks** | `NASDAQ:AAPL`, `NASDAQ:NVDA`, `NYSE:BRK.B` | Yahoo Finance |
| **Other stock exchanges** | `TSE:7203` (Tokyo), `HKEX:700`, `LSE:VOD`, `XETR:SAP`, `TSX:SHOP`, `ASX:BHP`, `NSE:RELIANCE`, `KRX:005930`, `SGX:D05`, `TWSE:2330` | Yahoo Finance |
| **Forex** | `FX:EURUSD`, `OANDA:USDJPY`, `FX:GBPUSD` | Yahoo Finance |
| **Indices** | `SP:SPX`, `NASDAQ:NDX`, `DJ:DJI`, `TVC:NI225`, `TVC:HSI`, `XETR:DAX`, `TVC:UKX`, `TVC:DXY` | Yahoo Finance |
| **Futures and metals** | `COMEX:GC1!`, `NYMEX:CL1!`, `CME_MINI:ES1!`, `OANDA:XAUUSD` | Yahoo Finance |

You can also type a **Yahoo Finance ticker** directly, such as `HPG.VN`, `GC=F`, `^GSPC` or `EURUSD=X`. Look one up on finance.yahoo.com.

**Not available from Yahoo:**
- **HNX and UPCOM stocks.** For these, export the chart from TradingView (chart menu → Export chart data) and use **Use CSV file…**.
- **VN-Index.** Yahoo's data for it is too sparse; use a TradingView CSV export.
- **Spot gold and silver (`XAUUSD`, `XAGUSD`).** The app uses gold or silver futures instead and says so next to the chart title. They track spot closely.

**How far back the data goes.** Yahoo serves 1-minute candles for the last 7 days, 5–30 minute candles for the last 60 days, and hourly candles for about 2 years. Kronos uses the latest 512 candles either way.

### Trading hours

Stocks, forex and futures don't trade around the clock, so their forecasts only cover the hours the market is open:
- **The forecast skips closed time.** Nights, weekends and lunch breaks are left out. For HOSE, the forecast runs 9:15–11:30, then 13:00–14:45.
- **Under a day, "Predict next" counts trading time.** "Next 4h" means the next 4 hours the market is open. If it's closed now, the forecast starts at the next open.
- **A day or more counts trading sessions.** "Next 1w" is 5 sessions, and "next 30d" is about 21.
- **Times are the exchange's local time.** For example, UTC+07:00 for HOSE.

Crypto still forecasts 24/7 in UTC.

## Using the app

| Control | What it does |
|---|---|
| **Symbol** | See the table above. Symbols you've used appear as quick buttons. |
| **Candle interval** | 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w |
| **Predict next** | How far ahead to forecast. The limit is 500 candles; the app tells you if a choice is too long. |
| **Model** | Kronos-mini, Kronos-small (the default; a good balance) or Kronos-base (larger and slower, and may be more accurate). Each model downloads once. |
| **Paths** | How many separate forecast paths to draw. More paths give a steadier range. On a GPU, 100–200 is quick. |
| **Re-run every candle** | Runs the forecast again each time a new candle closes. |
| **Use CSV file…** | Forecast from a file instead, for example a TradingView "Export chart data" CSV. |

**Reading the results:**
- **Median:** the middle forecast.
- **10th–90th range:** 80% of the forecast paths end inside it.
- **Paths ending higher:** close to 50% means the model sees no direction.

The badge next to the title sums it up: "Leans up", "Leans down" or "No clear direction". This is research output, not financial advice.

## Troubleshooting

**The top-right corner says "Running on CPU" but I have an NVIDIA card.** Update your NVIDIA driver, delete the `.venv` folder inside `kronos-local`, and double-click the launcher again.

**"Yahoo Finance returned no candles" or "only N candles".** The ticker may be wrong, or Yahoo has little history for it at that interval. Check the symbol on finance.yahoo.com, try a larger interval, or use a TradingView CSV export.

**"This exchange blocks your country (HTTP 451)".** Binance blocks some countries, including the US. Use `BYBIT:` or `OKX:` for the same pair.

**"Python was not found".** Install Python from python.org. On Windows, tick **Add python.exe to PATH**, then double-click the launcher again.

**The page says "Lost contact with the local server".** The launcher window was closed. Double-click the launcher again.

**I want to start over.** Delete the `.venv` and `Kronos` folders inside `kronos-local` and double-click the launcher.

## Files

| | |
|---|---|
| `Start Kronos (…)` | The launchers you double-click |
| `launcher.py` | First-run setup (and package updates), then starts the app |
| `app.py` | The local web server that runs Kronos on your GPU |
| `updater.py` | Checks GitHub for newer versions of the app and the Kronos code, test-runs them, and swaps them in |
| `app.json` | The app's version number, release notes, and the repository it updates from |
| `ui/index.html` | The page you see in the browser |
| `kronos_local.py` | Shared code for data download and the model. You can also run it from a terminal, e.g. `python kronos_local.py HOSE:HPG --interval 1d --horizon 1w` |

Everything runs on your computer. The app only goes online to download candles (from the crypto exchange or Yahoo Finance), plus the one-time downloads of the model and packages.

## Publishing an update (for whoever maintains this repository)

Installed copies update from the `main` branch of this repository:
1. Make the changes.
2. In `app.json`, raise `version` (for example `1.3.0` → `1.3.1`) and add an entry at the top of `changes` with a few plain-language notes. Users see these notes before they update.
3. Push to `main`. Every installed copy sees the update the next time its page opens, or within a few minutes, since GitHub caches the file briefly.

A push that doesn't raise `version` isn't offered as an update. The repository must stay **public**, because installed copies download from it without logging in.
