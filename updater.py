"""Check for and install newer versions of the Kronos model code and of this app, both from GitHub.

Every update is downloaded next to what's installed, test-run, and only then put in place.
The replaced version is kept (Kronos_previous / _app_previous) so it can be restored.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KRONOS_DIR = HERE / "Kronos"
NEW_DIR = HERE / "Kronos_new"
PREV_DIR = HERE / "Kronos_previous"
VERSION_FILE = ".kronos-version"
REPO = "shiyu-coder/Kronos"
BRANCH = "master"
GIT_URL = f"https://github.com/{REPO}"
ZIP_URL = f"{GIT_URL}/archive/refs/heads/{BRANCH}.zip"

# Runs in a fresh Python with the candidate code: the calls app.py makes must still work.
SMOKE_TEST = r"""
import inspect, sys
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd, torch
from model import Kronos, KronosTokenizer, KronosPredictor
need = ["df_list", "x_timestamp_list", "y_timestamp_list", "pred_len", "T", "top_p", "sample_count", "verbose"]
have = inspect.signature(KronosPredictor.predict_batch).parameters
missing = [p for p in need if p not in have]
assert not missing, "predict_batch no longer takes: " + ", ".join(missing)
p = KronosPredictor(Kronos.from_pretrained("NeoQuasar/Kronos-small"),
                    KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base"), device="cpu", max_context=512)
ts = pd.Series(pd.date_range("2024-01-01", periods=120, freq="1h"))
c = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 120))
df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.full(120, 1000.0)})
fut = pd.Series(pd.date_range(ts.iloc[-1] + pd.Timedelta(hours=1), periods=3, freq="1h"))
out = p.predict_batch([df] * 2, [ts] * 2, [fut] * 2, pred_len=3, T=1.0, top_p=0.9, sample_count=1, verbose=False)
assert len(out) == 2 and len(out[0]) == 3 and np.isfinite(out[0]["close"]).all(), "forecast output looks wrong"
print("ok")
"""


class UpdateError(Exception):
    pass


def _git() -> str | None:
    return shutil.which("git")


def _api(path: str) -> dict:
    req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/{path}",
                                 headers={"Accept": "application/vnd.github+json", "User-Agent": "kronos-local"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def version_of(folder: Path) -> str | None:
    if (folder / ".git").exists() and _git():
        r = subprocess.run([_git(), "-C", str(folder), "rev-parse", "HEAD"], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    f = folder / VERSION_FILE
    return f.read_text().strip() or None if f.exists() else None


def latest() -> dict:
    """Newest commit on GitHub: {sha, date, message}. Uses the API, or git when the API is unreachable."""
    try:
        c = _api(f"commits/{BRANCH}")
        return {"sha": c["sha"], "date": c["commit"]["committer"]["date"], "message": c["commit"]["message"].splitlines()[0]}
    except Exception as api_err:
        if _git():
            r = subprocess.run([_git(), "ls-remote", GIT_URL, f"refs/heads/{BRANCH}"], capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                return {"sha": r.stdout.split()[0], "date": "", "message": ""}
        raise UpdateError(f"Couldn't reach GitHub to check for updates ({api_err}).")


def status() -> dict:
    local = version_of(KRONOS_DIR)
    out = {"local": local, "can_undo": (PREV_DIR / "model").exists(), "previous": version_of(PREV_DIR)}
    try:
        new = latest()
    except UpdateError as e:
        return {**out, "error": str(e), "update_available": False}
    out.update(latest=new, update_available=local != new["sha"], behind=None, changes=[])
    if local and local != new["sha"]:
        try:
            cmp = _api(f"compare/{local}...{BRANCH}")
            out["behind"] = cmp.get("ahead_by")
            out["changes"] = [{"sha": c["sha"][:7], "date": c["commit"]["committer"]["date"],
                               "message": c["commit"]["message"].splitlines()[0]} for c in reversed(cmp.get("commits", []))][:20]
        except Exception:
            pass  # still updatable; we just can't list what changed
    return out


def _download(dest: Path) -> str | None:
    shutil.rmtree(dest, ignore_errors=True)
    if _git() and subprocess.call([_git(), "clone", "--depth", "1", "--branch", BRANCH, GIT_URL, str(dest)]) == 0:
        return version_of(dest)
    shutil.rmtree(dest, ignore_errors=True)
    try:
        sha = latest()["sha"]
    except UpdateError:
        sha = None
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "kronos-local"})
    try:
        data = urllib.request.urlopen(req, timeout=180).read()
    except Exception as e:
        raise UpdateError(f"Couldn't download Kronos from GitHub ({e}).")
    tmp = HERE / "_kronos_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = z.namelist()[0].split("/")[0]
        z.extractall(tmp)
    (tmp / root).rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    if sha:
        (dest / VERSION_FILE).write_text(sha)
    return sha


def _swap(incoming: Path, keep_as: Path) -> None:
    """Move KRONOS_DIR to keep_as and incoming into its place, restoring the original if anything fails."""
    shutil.rmtree(keep_as, ignore_errors=True)
    try:
        KRONOS_DIR.rename(keep_as)
    except OSError as e:
        raise UpdateError(f"Couldn't move the current Kronos folder ({e}). Close any window or editor that has it open and try again.")
    try:
        incoming.rename(KRONOS_DIR)
    except OSError as e:
        keep_as.rename(KRONOS_DIR)
        raise UpdateError(f"Couldn't put the new version in place ({e}). Nothing was changed.")


def update() -> dict:
    print("[update] downloading the latest Kronos code...", flush=True)
    sha = _download(NEW_DIR)
    print("[update] test-running a small forecast with the new code...", flush=True)
    r = subprocess.run([sys.executable, "-c", SMOKE_TEST, str(NEW_DIR)], capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or "ok" not in r.stdout:
        shutil.rmtree(NEW_DIR, ignore_errors=True)
        lines = [l for l in (r.stderr or r.stdout).strip().splitlines() if l.strip()]
        raise UpdateError("The new Kronos code doesn't work with this app yet, so your current version was kept. "
                          f"Details: {lines[-1] if lines else 'unknown error'}")
    _swap(NEW_DIR, PREV_DIR)
    print(f"[update] now on Kronos {sha[:7] if sha else '(unknown version)'}", flush=True)
    return {"ok": True, "sha": sha}


def undo() -> dict:
    if not (PREV_DIR / "model").exists():
        raise UpdateError("There's no earlier version to go back to.")
    swap_tmp = HERE / "Kronos_undo"
    shutil.rmtree(swap_tmp, ignore_errors=True)
    PREV_DIR.rename(swap_tmp)
    try:
        _swap(swap_tmp, PREV_DIR)  # current -> Kronos_previous, so undo can be undone too
    except UpdateError:
        if swap_tmp.exists():
            swap_tmp.rename(PREV_DIR)
        raise
    sha = version_of(KRONOS_DIR)
    print(f"[update] back on Kronos {sha[:7] if sha else '(unknown version)'}", flush=True)
    return {"ok": True, "sha": sha}


# ---------------------------------------------------------------- this app

APP_JSON = HERE / "app.json"
APP_PREV = HERE / "_app_previous"
APP_NEW = HERE / "_app_new"
JUST_UPDATED = HERE / ".app-just-updated"  # the launcher rolls back if the app fails to start while this exists
ADDED_LIST = ".added-files"
# Never touched by an app update: the user's environment, the Kronos code, update scratch space, and the
# start scripts (a Windows .bat that is rewritten while it runs misbehaves).
KEEP = {".venv", "Kronos", "Kronos_previous", "Kronos_new", "Kronos_undo", "_kronos_tmp", "_app_previous", "_app_new",
        "__pycache__", ".git", ".github", ".kronos-port", ".app-just-updated",
        "Start Kronos (Windows).bat", "Start Kronos (Mac).command", "start-linux.sh"}

APP_SMOKE_TEST = r"""
import os, sys
new, kronos = sys.argv[1], sys.argv[2]
os.environ["KRONOS_CODE_DIR"] = kronos
sys.path.insert(0, new)
import numpy as np
import app  # the new version's code
c = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 150))
rows = "\n".join(f"{1704067200 + i * 3600},{v},{v + 1},{v - 1},{v},1000" for i, v in enumerate(c))
r = app.forecast({"csv": "time,open,high,low,close,volume\n" + rows, "interval": "1h", "horizon": "3h", "model": "small", "paths": 4})
assert r["steps"] == 3 and len(r["p50"]) == 3 and len(r["candles"]) > 100, "forecast output looks wrong"
assert os.path.exists(os.path.join(new, "ui", "index.html")), "the page is missing"
print("ok")
"""


def _vtuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "0"))[:3] or (0,)


def app_info() -> dict:
    try:
        return json.loads(APP_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"version": "0", "repo": "", "branch": "main"}


def _manifest_url(info: dict) -> str:
    return info.get("manifest_url") or f"https://raw.githubusercontent.com/{info['repo']}/{info.get('branch', 'main')}/app.json"


def _zip_url(info: dict) -> str:
    return info.get("zip_url") or f"https://github.com/{info['repo']}/archive/refs/heads/{info.get('branch', 'main')}.zip"


def _get(url: str, timeout: int = 20) -> bytes:
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(f"{url}{sep}t={int(time.time())}" if url.endswith(".json") else url,
                                 headers={"User-Agent": "kronos-local", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _app_files(root: Path) -> list[Path]:
    """App files under root, relative to it, skipping everything in KEEP."""
    out = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and not any(part in KEEP for part in rel.parts) and not p.name.endswith((".pyc", ".zip")):
            out.append(rel)
    return out


def app_status() -> dict:
    mine = app_info()
    out = {"local": mine.get("version"), "can_undo": APP_PREV.exists(), "repo": mine.get("repo")}
    if APP_PREV.exists():
        try:
            out["previous"] = json.loads((APP_PREV / "app.json").read_text(encoding="utf-8")).get("version")
        except Exception:
            out["previous"] = None
    if not mine.get("repo") and not mine.get("manifest_url"):
        return {**out, "update_available": False, "error": "This copy of the app doesn't say where its updates are published."}
    try:
        remote = json.loads(_get(_manifest_url(mine)))
    except urllib.error.HTTPError as e:
        hint = " The repository may not exist yet or may be private." if e.code == 404 else ""
        return {**out, "update_available": False, "error": f"Couldn't read app updates from GitHub (HTTP {e.code}).{hint}"}
    except Exception as e:
        return {**out, "update_available": False, "error": f"Couldn't reach GitHub to check for app updates ({e})."}
    newer = _vtuple(remote.get("version")) > _vtuple(mine.get("version"))
    notes = [c for c in remote.get("changes", []) if _vtuple(c.get("version")) > _vtuple(mine.get("version"))]
    return {**out, "latest": remote.get("version"), "update_available": newer, "changes": notes[:10]}


def app_update() -> dict:
    mine = app_info()
    print("[app update] downloading the latest app version...", flush=True)
    try:
        data = _get(_zip_url(mine), timeout=120)
    except Exception as e:
        raise UpdateError(f"Couldn't download the app update ({e}).")
    shutil.rmtree(APP_NEW, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(APP_NEW)
    roots = [p.parent for p in APP_NEW.rglob("app.json")]
    if not roots:
        shutil.rmtree(APP_NEW, ignore_errors=True)
        raise UpdateError("The download doesn't look like this app (no app.json in it).")
    src = min(roots, key=lambda p: len(p.parts))
    new_ver = json.loads((src / "app.json").read_text(encoding="utf-8")).get("version")

    print(f"[app update] test-running a small forecast with version {new_ver}...", flush=True)
    r = subprocess.run([sys.executable, "-c", APP_SMOKE_TEST, str(src), str(KRONOS_DIR)],
                       capture_output=True, text=True, timeout=600, cwd=str(src))
    if r.returncode != 0 or "ok" not in r.stdout:
        shutil.rmtree(APP_NEW, ignore_errors=True)
        lines = [l for l in (r.stderr or r.stdout).strip().splitlines() if l.strip()]
        raise UpdateError(f"Version {new_ver} didn't pass its test run, so the app wasn't changed. "
                          f"Details: {lines[-1] if lines else 'unknown error'}")

    # back up what's installed, then copy the new files over it
    shutil.rmtree(APP_PREV, ignore_errors=True)
    incoming = _app_files(src)
    added = []
    for rel in set(_app_files(HERE)) | set(incoming):
        if (HERE / rel).exists():
            (APP_PREV / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HERE / rel, APP_PREV / rel)
        elif rel in incoming:
            added.append(rel.as_posix())
    (APP_PREV / ADDED_LIST).write_text("\n".join(added))
    for rel in incoming:
        (HERE / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, HERE / rel)
    shutil.rmtree(APP_NEW, ignore_errors=True)
    JUST_UPDATED.write_text(new_ver or "")
    print(f"[app update] installed version {new_ver}", flush=True)
    return {"ok": True, "version": new_ver}


def restore_previous_app() -> str | None:
    """Put the backed-up app files back. Kept free of anything but the standard library: the launcher calls it."""
    if not APP_PREV.exists():
        return None
    for name in (APP_PREV / ADDED_LIST).read_text().splitlines() if (APP_PREV / ADDED_LIST).exists() else []:
        if name.strip():
            (HERE / name).unlink(missing_ok=True)
    for rel in _app_files(APP_PREV):
        if rel.name == ADDED_LIST:
            continue
        (HERE / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(APP_PREV / rel, HERE / rel)
    shutil.rmtree(APP_PREV, ignore_errors=True)
    JUST_UPDATED.unlink(missing_ok=True)
    return app_info().get("version")


def app_undo() -> dict:
    if not APP_PREV.exists():
        raise UpdateError("There's no earlier app version to go back to.")
    ver = restore_previous_app()
    print(f"[app update] back on version {ver}", flush=True)
    return {"ok": True, "version": ver}
