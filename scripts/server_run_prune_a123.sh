#!/usr/bin/env bash

'''
conda run -n railseg python /root/bishe/src/global_prune.py --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py --checkpoint runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth --pruning-ratio 0.1 --target-stages 3 4 --max-target-layers 0 --shape 512 512 --device cuda:0 --pruned-checkpoint /root/bishe/checkpoints/A3_stage34_ffn_r10_ft10k_pruned.pth --output-json /root/bishe/exports/A3_stage34_ffn_r10_ft10k.json --enable-finetune --finetune-iters 10000 --finetune-lr 1e-05 --finetune-weight-decay 0.01 --finetune-eval-interval 200 --finetune-log-interval 50 --finetune-save-best /root/bishe/checkpoints/A3_stage34_ffn_r10_ft10k_best.pth --finetune-save-last /root/bishe/checkpoints/A3_stage34_ffn_r10_ft10k_last.pth --skip-latency
'''
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-railseg}"
DRY_RUN="${DRY_RUN:-0}"

CONFIG_PATH="${CONFIG_PATH:-configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-exports}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "[ERR] PROJECT_ROOT not found: $PROJECT_ROOT"
  exit 1
fi

cd "$PROJECT_ROOT"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERR] CONFIG_PATH not found: $CONFIG_PATH"
  exit 1
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "[ERR] CHECKPOINT_PATH not found: $CHECKPOINT_PATH"
  exit 1
fi
if [[ ! -f "src/run_prune_a123.py" ]]; then
  echo "[ERR] src/run_prune_a123.py not found"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR" "$CHECKPOINTS_DIR"
LOG_FILE="$OUTPUT_DIR/prune_a123_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] CONFIG_PATH=$CONFIG_PATH"
echo "[INFO] CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "[INFO] DEVICE=$DEVICE"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] CHECKPOINTS_DIR=$CHECKPOINTS_DIR"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] DRY_RUN=$DRY_RUN"

CMD=(python -u src/run_prune_a123.py
  --config "$CONFIG_PATH"
  --checkpoint "$CHECKPOINT_PATH"
  --device "$DEVICE"
  --conda-env "$CONDA_ENV_NAME"
  --output-dir "$OUTPUT_DIR"
  --checkpoints-dir "$CHECKPOINTS_DIR"
  --skip-latency
)

if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
  "${CMD[@]}"
  echo "[OK] dry-run done"
  exit 0
fi

nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

echo "[OK] A1/A2/A3 pruning experiments started in background"
echo "[OK] log: $LOG_FILE"
echo "[TIP] monitor: tail -f $LOG_FILE"
echo "[TIP] summary files after finish:"
echo "      $OUTPUT_DIR/prune_a123_summary.json"
echo "      $OUTPUT_DIR/prune_a123_summary.md"
