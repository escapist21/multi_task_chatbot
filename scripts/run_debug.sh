#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
export DEBUG=1
export PYTHONUNBUFFERED=1
cd "${ROOT_DIR}"
if [ -d venv ]; then
  source venv/bin/activate
fi
exec python main.py
