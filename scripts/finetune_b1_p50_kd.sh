#!/usr/bin/env bash
set -euo pipefail

python src/finetune_kd.py \
  --config configs/railsem19/segformer_b1_rs19_512x512_100ep_rtx4090.py \
  --pruned-model output/segformer_b1_uni_gm_50_gi/model_pruned.pth \
  --teacher-checkpoint runs/rs19/segformer_b1_512x512_100ep_rtx4090/best_mIoU.pth \
  --device cuda:0 \
  --work-dir runs/rs19/b1_p50_ft100_kd_logit \
  --finetune-epochs 100 \
  --lr 3e-5 \
  --weight-decay 0.01 \
  --distill logit \
  --kd-temperature 4.0 \
  --kd-loss-weight 0.05
