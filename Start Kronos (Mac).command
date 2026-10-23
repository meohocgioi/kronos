#!/bin/bash
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and double-click this file again."
  read -n 1 -s -r -p "Press any key to close"; exit 1
fi
python3 launcher.py
