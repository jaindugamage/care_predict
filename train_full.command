#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Run run_mac.command once before full training."
  read -r -p "Press Enter to close..."
  exit 1
fi

source .venv/bin/activate
python prepare_project.py --mode full
read -r -p "Full training completed. Press Enter to close..."
