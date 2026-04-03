#!/usr/bin/env bash
set -euo pipefail

# ====== Edit these paths on your server ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-railseg}"
CONFIG_PATH="${CONFIG_PATH:-configs/railsem19/segformer_b0_rs19_512x512_500it.py}"
WORK_DIR="${WORK_DIR:-runs/rs19/segformer_b0_512x512_500it_server_smoke}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
	echo "[ERR] PROJECT_ROOT not found: $PROJECT_ROOT"
	echo "[TIP] set PROJECT_ROOT explicitly, e.g. PROJECT_ROOT=$PWD ./scripts/server_smoke_test_500it.sh"
	exit 1
fi

cd "$PROJECT_ROOT"

if [[ ! -f "$CONFIG_PATH" ]]; then
	echo "[ERR] CONFIG_PATH not found: $CONFIG_PATH"
	exit 1
fi

if [[ ! -f "src/train_mmseg.py" ]]; then
	echo "[ERR] src/train_mmseg.py not found under PROJECT_ROOT=$PROJECT_ROOT"
	exit 1
fi

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Enable custom dataset import: datasets.rs19_mmseg_dataset
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

mkdir -p "$WORK_DIR"

LOG_FILE="$WORK_DIR/smoke_test_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] CONFIG_PATH=$CONFIG_PATH"
echo "[INFO] WORK_DIR=$WORK_DIR"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] DRY_RUN=$DRY_RUN"

if [[ "$DRY_RUN" == "1" ]]; then
	echo "[OK] dry-run checks passed"
	echo "[TIP] remove DRY_RUN or set DRY_RUN=0 to actually start smoke test"
	exit 0
fi

# Single-GPU launch
nohup python src/train_mmseg.py "$CONFIG_PATH" --work-dir "$WORK_DIR" > "$LOG_FILE" 2>&1 &

echo "[OK] smoke test started in background"
echo "[OK] log: $LOG_FILE"
echo "[TIP] monitor: tail -f $LOG_FILE"
echo "[TIP] pass criteria: training starts, runs to 500 iters, and saves checkpoint/best mIoU"
