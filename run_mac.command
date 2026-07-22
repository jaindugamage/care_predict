#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

pause_on_error() {
  status=$?
  echo
  echo "CarePredict stopped because a command failed."
  echo "Check the error shown above."
  read -r -p "Press Enter to close..." || true
  exit "$status"
}
trap pause_on_error ERR

REQUIRED_ITEMS=(
  "app.py"
  "prepare_project.py"
  "requirements.txt"
  "data/raw/diabetic_data.csv"
  "src"
)

for item in "${REQUIRED_ITEMS[@]}"; do
  if [ ! -e "$item" ]; then
    echo "Missing required item: $item"
    read -r -p "Press Enter to close..." || true
    exit 1
  fi
done

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)
PY
then
  echo "CarePredict requires Python 3.12 or Python 3.13."
  "$PYTHON_BIN" --version 2>/dev/null || true
  read -r -p "Press Enter to close..." || true
  exit 1
fi

if [ -d ".venv" ] && ! .venv/bin/python - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)
PY
then
  rm -rf .venv
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

REQ_HASH=$(python - <<'PY'
from pathlib import Path
import hashlib
print(hashlib.sha256(Path("requirements.txt").read_bytes()).hexdigest())
PY
)
INSTALLED_HASH=""
[ -f .venv/.requirements_hash ] && INSTALLED_HASH=$(cat .venv/.requirements_hash)

if [ "$REQ_HASH" != "$INSTALLED_HASH" ]; then
  python -m pip install --prefer-binary -r requirements.txt
  python scripts/check_environment.py
  echo "$REQ_HASH" > .venv/.requirements_hash
else
  python scripts/check_environment.py
fi

if [ ! -f artifacts/intake_model.joblib ] || [ ! -f artifacts/discharge_model.joblib ]; then
  echo "Training the quick models for the first run..."
  python prepare_project.py --mode quick
fi

python -m streamlit run app.py
