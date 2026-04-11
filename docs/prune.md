# 剪枝、微调、蒸馏对比实验命令总表（RailSem19 / SegFormer-B0）

> 说明：本文件汇总“基线 + 各剪枝策略 + 微调 + 蒸馏联合优化”全部命令。建议使用 `&&` 串联，确保上一步失败时自动停止。

## 0) 环境准备

```bash
source /home/lcy/miniconda3/etc/profile.d/conda.sh
conda activate railseg
export PYTHONPATH=/home/lcy/Projects/bishe/src:${PYTHONPATH}
cd /home/lcy/Projects/bishe
mkdir -p runs/rs19/experiments checkpoints exports reports
```

---

## 1) 基线（B）

```bash
python src/baseline_metrics.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--output exports/baseline_metrics.json \
	--device cuda:0 \
	--shape 512 512 \
	--iters 200 \
	--repeat 3 \
	--min-miou 55 \
	--strict-baseline-check
```

---

## 2) Local Iterative Pruning

### 2.1 P-L-10（0.10）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.10 --iterative-steps 5 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_local_10 \
	--save-model checkpoints/tp_local_10_model.pth \
	--save-model-object \
	--save-json exports/tp_local_10_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_10_model.pth \
	--output exports/metrics_tp_local_10.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_local_10.json \
	--title "Local Iterative Pruning 10%" \
	--out-md reports/compare_local_10.md
```

### 2.2 P-L-20（0.20）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.20 --iterative-steps 5 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_local_20 \
	--save-model checkpoints/tp_local_20_model.pth \
	--save-model-object \
	--save-json exports/tp_local_20_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_20_model.pth \
	--output exports/metrics_tp_local_20.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_local_20.json \
	--title "Local Iterative Pruning 20%" \
	--out-md reports/compare_local_20.md
```

### 2.3 P-L-30（0.30）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.30 --iterative-steps 5 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_local_30 \
	--save-model checkpoints/tp_local_30_model.pth \
	--save-model-object \
	--save-json exports/tp_local_30_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_30_model.pth \
	--output exports/metrics_tp_local_30.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_local_30.json \
	--title "Local Iterative Pruning 30%" \
	--out-md reports/compare_local_30.md
```

### 2.4 P-L-40（0.40）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.40 --iterative-steps 5 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_local_40 \
	--save-model checkpoints/tp_local_40_model.pth \
	--save-model-object \
	--save-json exports/tp_local_40_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_40_model.pth \
	--output exports/metrics_tp_local_40.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_local_40.json \
	--title "Local Iterative Pruning 40%" \
	--out-md reports/compare_local_40.md
```

---

## 3) Global Iterative Pruning

### 3.1 P-G-10（0.10）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.10 --iterative-steps 5 --global-pruning \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_global_10 \
	--save-model checkpoints/tp_global_10_model.pth \
	--save-model-object \
	--save-json exports/tp_global_10_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_global_10_model.pth \
	--output exports/metrics_tp_global_10.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_global_10.json \
	--title "Global Iterative Pruning 10%" \
	--out-md reports/compare_global_10.md
```

### 3.2 P-G-20（0.20）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.20 --iterative-steps 5 --global-pruning \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_global_20 \
	--save-model checkpoints/tp_global_20_model.pth \
	--save-model-object \
	--save-json exports/tp_global_20_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_global_20_model.pth \
	--output exports/metrics_tp_global_20.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_global_20.json \
	--title "Global Iterative Pruning 20%" \
	--out-md reports/compare_global_20.md
```

### 3.3 P-G-30（0.30）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.30 --iterative-steps 5 --global-pruning \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_global_30 \
	--save-model checkpoints/tp_global_30_model.pth \
	--save-model-object \
	--save-json exports/tp_global_30_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_global_30_model.pth \
	--output exports/metrics_tp_global_30.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_global_30.json \
	--title "Global Iterative Pruning 30%" \
	--out-md reports/compare_global_30.md
```

---

## 4) Global + Isomorphic Pruning

### 4.1 P-GI-20（0.20）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.20 --iterative-steps 5 --global-pruning --isomorphic --round-to 8 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_global_iso_20 \
	--save-model checkpoints/tp_global_iso_20_model.pth \
	--save-model-object \
	--save-json exports/tp_global_iso_20_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_global_iso_20_model.pth \
	--output exports/metrics_tp_global_iso_20.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_global_iso_20.json \
	--title "Global+Isomorphic Pruning 20%" \
	--out-md reports/compare_global_iso_20.md
```

### 4.2 P-GI-30（0.30）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.30 --iterative-steps 5 --global-pruning --isomorphic --round-to 8 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_global_iso_30 \
	--save-model checkpoints/tp_global_iso_30_model.pth \
	--save-model-object \
	--save-json exports/tp_global_iso_30_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_global_iso_30_model.pth \
	--output exports/metrics_tp_global_iso_30.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_global_iso_30.json \
	--title "Global+Isomorphic Pruning 30%" \
	--out-md reports/compare_global_iso_30.md
```

---

## 5) Safe Pruning

### 5.1 P-S-20（Safe Accuracy）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.20 --iterative-steps 5 --segformer-safe --safe-target accuracy \
	--safe-pruning-ratio 0.20 --safe-max-pruning-ratio 0.35 --safe-round-to 40 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_safe_acc_20 \
	--save-model checkpoints/tp_safe_acc_20_model.pth \
	--save-model-object \
	--save-json exports/tp_safe_acc_20_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_safe_acc_20_model.pth \
	--output exports/metrics_tp_safe_acc_20.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_safe_acc_20.json \
	--title "Safe Accuracy Pruning 20%" \
	--out-md reports/compare_safe_acc_20.md
```

### 5.2 P-S-30（Safe Accuracy）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.30 --iterative-steps 5 --segformer-safe --safe-target accuracy \
	--safe-pruning-ratio 0.30 --safe-max-pruning-ratio 0.35 --safe-round-to 40 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_safe_acc_30 \
	--save-model checkpoints/tp_safe_acc_30_model.pth \
	--save-model-object \
	--save-json exports/tp_safe_acc_30_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_safe_acc_30_model.pth \
	--output exports/metrics_tp_safe_acc_30.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_safe_acc_30.json \
	--title "Safe Accuracy Pruning 30%" \
	--out-md reports/compare_safe_acc_30.md
```

### 5.3 P-SL-20（Safe Latency）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.20 --iterative-steps 5 --segformer-safe --safe-target latency \
	--safe-pruning-ratio 0.20 --safe-max-pruning-ratio 0.30 --safe-round-to 8 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_safe_lat_20 \
	--save-model checkpoints/tp_safe_lat_20_model.pth \
	--save-model-object \
	--save-json exports/tp_safe_lat_20_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_safe_lat_20_model.pth \
	--output exports/metrics_tp_safe_lat_20.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_safe_lat_20.json \
	--title "Safe Latency Pruning 20%" \
	--out-md reports/compare_safe_lat_20.md
```

### 5.4 P-SL-30（Safe Latency）

```bash
# 剪枝
python src/torch_prune_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--pruning-ratio 0.30 --iterative-steps 5 --segformer-safe --safe-target latency \
	--safe-pruning-ratio 0.30 --safe-max-pruning-ratio 0.30 --safe-round-to 8 \
	--fail-fast-attn-violation \
	--shape 512 512 --device cuda:0 \
	--work-dir runs/rs19/exp_safe_lat_30 \
	--save-model checkpoints/tp_safe_lat_30_model.pth \
	--save-model-object \
	--save-json exports/tp_safe_lat_30_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_safe_lat_30_model.pth \
	--output exports/metrics_tp_safe_lat_30.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_tp_safe_lat_30.json \
	--title "Safe Latency Pruning 30%" \
	--out-md reports/compare_safe_lat_30.md
```

---

## 6) 剪枝后微调（基于 P-L-30）

### 6.1 FT-500

```bash
# 微调
python src/finetune_pruned_model.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_30_model.pth \
	--device cuda:0 --work-dir runs/rs19/ft_500 \
	--iters 500 --lr 1e-5 --weight-decay 0.01 --grad-clip 1.0 \
	--eval-after \
	--save-model checkpoints/tp_local_30_ft_500.pth \
	--save-json exports/tp_local_30_ft_500.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_30_ft_500.pth \
	--output exports/metrics_ft_500.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_ft_500.json \
	--title "Finetune 500 iters on Local 30%" \
	--out-md reports/compare_ft_500.md
```

### 6.2 FT-1000

```bash
# 微调
python src/finetune_pruned_model.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_30_model.pth \
	--device cuda:0 --work-dir runs/rs19/ft_1000 \
	--iters 1000 --lr 1e-5 --weight-decay 0.01 --grad-clip 1.0 \
	--eval-after \
	--save-model checkpoints/tp_local_30_ft_1000.pth \
	--save-json exports/tp_local_30_ft_1000.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_30_ft_1000.pth \
	--output exports/metrics_ft_1000.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_ft_1000.json \
	--title "Finetune 1000 iters on Local 30%" \
	--out-md reports/compare_ft_1000.md
```

### 6.3 FT-2000

```bash
# 微调
python src/finetune_pruned_model.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_30_model.pth \
	--device cuda:0 --work-dir runs/rs19/ft_2000 \
	--iters 2000 --lr 1e-5 --weight-decay 0.01 --grad-clip 1.0 \
	--eval-after \
	--save-model checkpoints/tp_local_30_ft_2000.pth \
	--save-json exports/tp_local_30_ft_2000.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_local_30_ft_2000.pth \
	--output exports/metrics_ft_2000.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_ft_2000.json \
	--title "Finetune 2000 iters on Local 30%" \
	--out-md reports/compare_ft_2000.md
```

### 6.4 FT-safe-20000（基于 P-S-20）

```bash
# 微调
python src/finetune_pruned_model.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_safe_acc_20_model.pth \
	--device cuda:0 \
	--work-dir runs/rs19/ft_safe_acc_20 \
	--iters 20000 \
	--lr 1e-5 \
	--weight-decay 0.01 \
	--grad-clip 1.0 \
	--eval-after \
	--save-model checkpoints/tp_safe_acc_20_ft_20000.pth \
	--save-json exports/tp_safe_acc_20_ft_20000.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_safe_acc_20_ft_20000.pth \
	--output exports/metrics_safe_acc_20_ft_20000.json \
	--device cuda:0 \
	--shape 512 512 \
	--work-dir runs/rs19/tmp_eval_ft

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_safe_acc_20_ft_20000.json \
	--title "Finetune 20000 iters on Safe Accuracy Pruning 20%" \
	--out-md reports/compare_safe_acc_20_ft_20000.md
```

---

## 7) 蒸馏 + 剪枝（MGD+Prune）

| 实验 ID | 策略 | 目标剪枝率 | 训练强度 | 备注 |
|---|---|---:|---|---|
| **D+P-12** | MGD+Prune | 12% | 800 × 3 rounds | 蒸馏+剪枝 |
| **D+P-20** | MGD+Prune | 20% | 800 × 4 rounds | 目标剪枝率 20% |
| **D+P+FT** | D+P-12 + 微调 | 12% | +1000 iter | 额外微调 |

### 7.1 D+P-12（12%，800×3）

```bash
# 蒸馏+剪枝
python src/distill_pruned_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--device cuda:0 --shape 512 512 \
	--target-pruning-rate 0.12 \
	--rounds 3 \
	--epsilon-miou-drop 2.5 \
	--max-pruning-ratio 0.20 \
	--round-to 40 \
	--round-iters 800 \
	--lr 8e-6 \
	--weight-decay 0.01 \
	--grad-clip 1.0 \
	--log-interval 100 \
	--mgd-lambda 0.05 \
	--mgd-mask-ratio 0.5 \
	--ignore-keywords decode_head auxiliary_head backbone.layers.2 backbone.layers.3 \
	--work-dir runs/rs19/distill_prune_12_r3 \
	--save-model checkpoints/tp_distill_prune_12_r3_best.pth \
	--save-json exports/tp_distill_prune_12_r3_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_distill_prune_12_r3_best.pth \
	--output exports/metrics_dp_12.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval_dp12

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_dp_12.json \
	--title "D+P-12 (MGD+Prune, 12%, 800x3)" \
	--out-md reports/compare_dp_12.md
```

### 7.2 D+P-20（20%，800×4）

```bash
# 蒸馏+剪枝
python src/distill_pruned_segformer.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--checkpoint runs/best_mIoU_iter_v1.pth \
	--device cuda:0 --shape 512 512 \
	--target-pruning-rate 0.20 \
	--rounds 4 \
	--epsilon-miou-drop 2.5 \
	--max-pruning-ratio 0.20 \
	--round-to 40 \
	--round-iters 800 \
	--lr 8e-6 \
	--weight-decay 0.01 \
	--grad-clip 1.0 \
	--log-interval 100 \
	--mgd-lambda 0.05 \
	--mgd-mask-ratio 0.5 \
	--ignore-keywords decode_head auxiliary_head backbone.layers.2 backbone.layers.3 \
	--work-dir runs/rs19/distill_prune_20_r4 \
	--save-model checkpoints/tp_distill_prune_20_r4_best.pth \
	--save-json exports/tp_distill_prune_20_r4_summary.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_distill_prune_20_r4_best.pth \
	--output exports/metrics_dp_20.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval_dp20

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_dp_20.json \
	--title "D+P-20 (MGD+Prune, 20%, 800x4)" \
	--out-md reports/compare_dp_20.md
```

### 7.3 D+P+FT（基于 D+P-12，再微调 1000 iter）

```bash
# 在 D+P-12 输出模型基础上微调
python src/finetune_pruned_model.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_distill_prune_12_r3_best.pth \
	--device cuda:0 --work-dir runs/rs19/ft_dp12_1000 \
	--iters 1000 --lr 1e-5 --weight-decay 0.01 --grad-clip 1.0 \
	--eval-after \
	--save-model checkpoints/tp_distill_prune_12_r3_ft_1000.pth \
	--save-json exports/tp_distill_prune_12_r3_ft_1000.json

# 评估
python src/eval_pruned_model_full.py \
	--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
	--model-path checkpoints/tp_distill_prune_12_r3_ft_1000.pth \
	--output exports/metrics_dp_12_ft_1000.json \
	--device cuda:0 --shape 512 512 \
	--work-dir runs/rs19/tmp_eval_dp12_ft

# 对比
python src/compare_baseline_pruned.py \
	--baseline-json exports/baseline_metrics.json \
	--pruned-json exports/metrics_dp_12_ft_1000.json \
	--title "D+P+FT (D+P-12 + 1000 iters)" \
	--out-md reports/compare_dp_12_ft_1000.md
```

