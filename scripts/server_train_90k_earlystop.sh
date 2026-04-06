#!/usr/bin/env bash
set -euo pipefail

# ====== Edit these paths on your server ======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-railseg}"
CONFIG_PATH="${CONFIG_PATH:-configs/railsem19/segformer_b0_rs19_512x512_90000it_earlystop.py}"
WORK_DIR="${WORK_DIR:-runs/rs19/segformer_b0_512x512_100000it_earlystop}"
DRY_RUN="${DRY_RUN:-0}"

# Optional checkpoints
RESUME_CKPT="${RESUME_CKPT:-}"
LOAD_CKPT="${LOAD_CKPT:-}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
	echo "[ERR] PROJECT_ROOT not found: $PROJECT_ROOT"
	echo "[TIP] set PROJECT_ROOT explicitly, e.g. PROJECT_ROOT=$PWD ./scripts/server_train_90k_earlystop.sh"
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

if [[ -n "$RESUME_CKPT" && ! -f "$RESUME_CKPT" ]]; then
	echo "[ERR] RESUME_CKPT not found: $RESUME_CKPT"
	exit 1
fi

if [[ -n "$LOAD_CKPT" && ! -f "$LOAD_CKPT" ]]; then
	echo "[ERR] LOAD_CKPT not found: $LOAD_CKPT"
	exit 1
fi

if [[ -n "$RESUME_CKPT" && -n "$LOAD_CKPT" ]]; then
	echo "[ERR] set only one of RESUME_CKPT or LOAD_CKPT"
	exit 1
fi

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Important: enable custom dataset / hooks imports
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

mkdir -p "$WORK_DIR"

LOG_FILE="$WORK_DIR/server_train_100k_earlystop_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] CONFIG_PATH=$CONFIG_PATH"
echo "[INFO] WORK_DIR=$WORK_DIR"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] DRY_RUN=$DRY_RUN"
if [[ -n "$RESUME_CKPT" ]]; then
	echo "[INFO] RESUME_CKPT=$RESUME_CKPT"
fi
if [[ -n "$LOAD_CKPT" ]]; then
	echo "[INFO] LOAD_CKPT=$LOAD_CKPT"
fi

if [[ "$DRY_RUN" == "1" ]]; then
	echo "[OK] dry-run checks passed"
	echo "[TIP] remove DRY_RUN or set DRY_RUN=0 to actually start training"
	exit 0
fi

if [[ -n "$RESUME_CKPT" ]]; then
	nohup python src/train_mmseg.py "$CONFIG_PATH" --work-dir "$WORK_DIR" --resume-from "$RESUME_CKPT" > "$LOG_FILE" 2>&1 &
elif [[ -n "$LOAD_CKPT" ]]; then
	nohup python src/train_mmseg.py "$CONFIG_PATH" --work-dir "$WORK_DIR" --load-from "$LOAD_CKPT" > "$LOG_FILE" 2>&1 &
else
	nohup python src/train_mmseg.py "$CONFIG_PATH" --work-dir "$WORK_DIR" > "$LOG_FILE" 2>&1 &
fi

echo "[OK] 100k early-stop training started in background"
echo "[OK] log: $LOG_FILE"
echo "[TIP] monitor: tail -f $LOG_FILE"
echo "[TIP] early stop rule: mIoU improvement < 0.05 for 5 consecutive validations"
