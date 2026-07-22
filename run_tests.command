#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Run run_mac.command once before running the tests."
  read -r -p "Press Enter to close..."
  exit 1
fi

source .venv/bin/activate
python -m pytest -q
read -r -p "Tests completed. Press Enter to close..."
