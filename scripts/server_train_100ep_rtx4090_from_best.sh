#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-railseg}"
CONFIG_PATH="${CONFIG_PATH:-configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py}"
WORK_DIR="${WORK_DIR:-runs/rs19/segformer_b0_512x512_100ep_rtx4090_from_best}"
DRY_RUN="${DRY_RUN:-0}"

# Default: load from current 50ep best checkpoint.
# You can override by setting LOAD_CKPT env.
LOAD_CKPT="${LOAD_CKPT:-runs/rs19/segformer_b0_512x512_50ep_rtx4090/best_mIoU_iter_81600.pth}"
RESUME_CKPT="${RESUME_CKPT:-}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[ERR] PROJECT_ROOT not found: $PROJECT_ROOT"
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

if [[ -n "$LOAD_CKPT" && ! -f "$LOAD_CKPT" ]]; then
  echo "[ERR] LOAD_CKPT not found: $LOAD_CKPT"
  exit 1
fi

if [[ -n "$RESUME_CKPT" && ! -f "$RESUME_CKPT" ]]; then
  echo "[ERR] RESUME_CKPT not found: $RESUME_CKPT"
  exit 1
fi

if [[ -n "$RESUME_CKPT" && -n "$LOAD_CKPT" ]]; then
  echo "[ERR] set only one of RESUME_CKPT or LOAD_CKPT"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

mkdir -p "$WORK_DIR"
LOG_FILE="$WORK_DIR/server_train_100ep_rtx4090_from_best_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] CONFIG_PATH=$CONFIG_PATH"
echo "[INFO] WORK_DIR=$WORK_DIR"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] DRY_RUN=$DRY_RUN"
[[ -n "$LOAD_CKPT" ]] && echo "[INFO] LOAD_CKPT=$LOAD_CKPT"
[[ -n "$RESUME_CKPT" ]] && echo "[INFO] RESUME_CKPT=$RESUME_CKPT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[OK] dry-run checks passed"
  exit 0
fi

if [[ -n "$RESUME_CKPT" ]]; then
  nohup python src/train_mmseg.py "$CONFIG_PATH" --work-dir "$WORK_DIR" --resume-from "$RESUME_CKPT" > "$LOG_FILE" 2>&1 &
elif [[ -n "$LOAD_CKPT" ]]; then
  nohup python src/train_mmseg.py "$CONFIG_PATH" --work-dir "$WORK_DIR" --load-from "$LOAD_CKPT" > "$LOG_FILE" 2>&1 &
else
  nohup python src/train_mmseg.py "$CONFIG_PATH" --work-dir "$WORK_DIR" > "$LOG_FILE" 2>&1 &
fi

echo "[OK] 100-epoch continuation training started in background"
echo "[OK] log: $LOG_FILE"
echo "[TIP] monitor: tail -f $LOG_FILE"
