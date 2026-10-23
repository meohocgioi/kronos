#!/usr/bin/env python3
"""First run: builds a private Python environment with GPU PyTorch and downloads Kronos.
Every run: starts the Kronos Forecast web app and opens it in your browser.
Uses only the Python standard library, so it runs on a fresh Python install."""
from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import venv
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
MARKER = VENV / ".kronos-installed"
KRONOS_DIR = HERE / "Kronos"
KRONOS_GIT = "https://github.com/shiyu-coder/Kronos"
KRONOS_ZIP = KRONOS_GIT + "/archive/refs/heads/master.zip"
# Newest first; the first one pip can install for your Python is used.
CUDA_BUILDS = ["cu128", "cu126", "cu124"]
IS_WIN = os.name == "nt"
PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def say(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def pip(*args: str) -> bool:
    return subprocess.call([str(PY), "-m", "pip", "install", "--disable-pip-version-check", *args]) == 0


def has_nvidia() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        p = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(p) if p.exists() else None
    return bool(exe) and subprocess.call([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def req_hash() -> str:
    return hashlib.sha256((HERE / "requirements.txt").read_bytes()).hexdigest()[:16]


def has_torch() -> bool:
    return PY.exists() and subprocess.call([str(PY), "-c", "import torch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def install() -> None:
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is needed. Install it from python.org and try again.")
    if not PY.exists():
        say("Creating a private Python environment in .venv (one time)")
        venv.create(VENV, with_pip=True)
    pip("--upgrade", "pip")

    mac = platform.system() == "Darwin"
    if has_torch():
        say("PyTorch is already installed, keeping it")
        ok = True
    elif mac:
        say("Installing PyTorch for Mac (uses the Apple GPU)")
        ok = pip("torch")
    elif has_nvidia():
        ok = False
        for build in CUDA_BUILDS:
            say(f"NVIDIA GPU found. Installing PyTorch with CUDA ({build}). This is a large download, about 2-3 GB")
            if pip("torch", "--index-url", f"https://download.pytorch.org/whl/{build}"):
                ok = True
                break
    else:
        say("No NVIDIA GPU found. Installing the CPU version of PyTorch (forecasts will be slower)")
        ok = pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    if not ok:
        sys.exit("PyTorch failed to install. Check your internet connection and try again.")

    say("Installing the other packages")
    if not pip("-r", str(HERE / "requirements.txt")):
        sys.exit("Package install failed. Check your internet connection and try again.")

    if not (KRONOS_DIR / "model").exists():
        say("Downloading the Kronos model code from GitHub")
        shutil.rmtree(KRONOS_DIR, ignore_errors=True)
        git = shutil.which("git")
        if not (git and subprocess.call([git, "clone", "--depth", "1", KRONOS_GIT, str(KRONOS_DIR)]) == 0):
            req = urllib.request.Request(KRONOS_ZIP, headers={"User-Agent": "kronos-local-launcher"})
            try:
                data = urllib.request.urlopen(req, timeout=180).read()
            except Exception as e:
                sys.exit(f"Couldn't download Kronos from GitHub ({e}). Check your internet connection and try again.")
            tmp = HERE / "_kronos_tmp"
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                root = z.namelist()[0].split("/")[0]
                z.extractall(tmp)
            (tmp / root).rename(KRONOS_DIR)
            shutil.rmtree(tmp, ignore_errors=True)
            try:  # remember which version this is, so the app can tell when GitHub has a newer one
                req = urllib.request.Request("https://api.github.com/repos/shiyu-coder/Kronos/commits/master",
                                             headers={"User-Agent": "kronos-local-launcher"})
                import json
                (KRONOS_DIR / ".kronos-version").write_text(json.load(urllib.request.urlopen(req, timeout=15))["sha"])
            except Exception:
                pass

    check = subprocess.run([str(PY), "-c", "import torch;print(torch.cuda.is_available() or torch.backends.mps.is_available())"], capture_output=True, text=True)
    if check.stdout.strip() != "True" and (mac or has_nvidia()):
        say("Warning: PyTorch installed but can't see your GPU. Update your NVIDIA driver, then delete the .venv folder and start again.")
    MARKER.write_text(req_hash())
    say("Setup finished")


# Kept inside the launcher, so a broken app update can't break the way back.
APP_PREV = HERE / "_app_previous"
JUST_UPDATED = HERE / ".app-just-updated"
APP_KEEP = {".venv", "Kronos", "Kronos_previous", "Kronos_new", "Kronos_undo", "_kronos_tmp", "_app_previous", "_app_new",
            "__pycache__", ".git", ".github", ".kronos-port", ".app-just-updated",
            "Start Kronos (Windows).bat", "Start Kronos (Mac).command", "start-linux.sh"}


def restore_previous_app() -> None:
    added = APP_PREV / ".added-files"
    for name in (added.read_text().splitlines() if added.exists() else []):
        if name.strip():
            (HERE / name).unlink(missing_ok=True)
    for p in APP_PREV.rglob("*"):
        rel = p.relative_to(APP_PREV)
        if p.is_file() and p.name != ".added-files" and not any(part in APP_KEEP for part in rel.parts):
            (HERE / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, HERE / rel)
    shutil.rmtree(APP_PREV, ignore_errors=True)
    JUST_UPDATED.unlink(missing_ok=True)


def main() -> None:
    os.chdir(HERE)
    if not MARKER.exists() or MARKER.read_text().strip() != req_hash() or not (KRONOS_DIR / "model").exists():
        install()
    env = {**os.environ, "PYTHONWARNINGS": os.environ.get("PYTHONWARNINGS", "ignore")}
    while True:
        rc = subprocess.call([str(PY), str(HERE / "app.py")], env=env)
        if rc not in (0, 3) and JUST_UPDATED.exists() and APP_PREV.exists():
            say("The updated app didn't start, so going back to the previous version")
            restore_previous_app()
        elif rc != 3:  # 3 = the app asked to restart after an update
            sys.exit(rc)
        # Files changed: re-check packages (a new app version may need one) before starting again
        if not MARKER.exists() or MARKER.read_text().strip() != req_hash():
            install()
        say("Restarting the app")
        env["KRONOS_NO_BROWSER"] = "1"  # the page that asked for the update reloads itself
        try:
            env["KRONOS_PORT"] = (HERE / ".kronos-port").read_text().strip()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
