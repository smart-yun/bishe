# SegFormer RailSem19 实验运行记录

## 说明

* **默认配置文件：**`configs/experiments.yaml`
* **命令入口：**`python src/run_exp.py --exp <exp_name> --task <task> [--variant <variant>]`
* `variant` 说明：
  * `baseline`：基线 checkpoint
  * `pruned`：剪枝后、微调前模型
  * `ft`：微调后模型

---

## 一、Baseline 测量

### B0 baseline

```
 python src/run_exp.py --exp b0_base --task eval
 python src/run_exp.py --exp b0_base --task flops
 python src/run_exp.py --exp b0_base --task latency
```

### B1 baseline

```
 python src/run_exp.py --exp b1_base --task eval
 python src/run_exp.py --exp b1_base --task flops
 python src/run_exp.py --exp b1_base --task latency
```

---

## 二、B0 剪枝实验

### b0\_p10

```
 python src/run_exp.py --exp b0_p10 --task prune
 python src/run_exp.py --exp b0_p10 --task eval --variant pruned
 python src/run_exp.py --exp b0_p10 --task flops --variant pruned
 python src/run_exp.py --exp b0_p10 --task latency --variant pruned
 
 python src/run_exp.py --exp b0_p10 --task finetune
 
 python src/run_exp.py --exp b0_p10 --task eval --variant ft
 python src/run_exp.py --exp b0_p10 --task flops --variant ft
 python src/run_exp.py --exp b0_p10 --task latency --variant ft
```

### b0\_p30

```
 python src/run_exp.py --exp b0_p30 --task prune
 python src/run_exp.py --exp b0_p30 --task eval --variant pruned
 python src/run_exp.py --exp b0_p30 --task flops --variant pruned
 python src/run_exp.py --exp b0_p30 --task latency --variant pruned
 
 python src/run_exp.py --exp b0_p30 --task finetune
 
 python src/run_exp.py --exp b0_p30 --task eval --variant ft
 python src/run_exp.py --exp b0_p30 --task flops --variant ft
 python src/run_exp.py --exp b0_p30 --task latency --variant ft
```

### b0\_p50  剪枝加上微调25epoch基本能恢复到剪枝前的性能了

```
 python src/run_exp.py --exp b0_p50 --task prune
 python src/run_exp.py --exp b0_p50 --task eval --variant pruned
 python src/run_exp.py --exp b0_p50 --task flops --variant pruned
 python src/run_exp.py --exp b0_p50 --task latency --variant pruned
 
 python src/run_exp.py --exp b0_p50 --task finetune
 
 python src/run_exp.py --exp b0_p50 --task eval --variant ft
 python src/run_exp.py --exp b0_p50 --task flops --variant ft
 python src/run_exp.py --exp b0_p50 --task latency --variant ft
```

### b0\_p70

```
  python src/run_exp.py --exp b0_p70 --task prune
  python src/run_exp.py --exp b0_p70 --task eval --variant pruned
  python src/run_exp.py --exp b0_p70 --task flops --variant pruned
  python src/run_exp.py --exp b0_p70 --task latency --variant pruned 

  python src/run_exp.py --exp b0_p70 --task finetune
  python src/run_exp.py --exp b0_p70 --task eval --variant ft
  python src/run_exp.py --exp b0_p70 --task flops --variant ft
  python src/run_exp.py --exp b0_p70 --task latency --variant ft
```
### b0\_p75

```
  python src/run_exp.py --exp b0_p75 --task prune
  python src/run_exp.py --exp b0_p75 --task eval --variant pruned
  python src/run_exp.py --exp b0_p75 --task flops --variant pruned
  python src/run_exp.py --exp b0_p75 --task latency --variant pruned 

python src/run_exp.py --exp b0_p75 --task finetune
python src/run_exp.py --exp b0_p75 --task eval --variant ft
python src/run_exp.py --exp b0_p75 --task flops --variant ft
python src/run_exp.py --exp b0_p75 --task latency --variant ft
```

### B0-P75 + KD蒸馏
```
python src/finetune_kd.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  --pruned-model output/segformer_b0_mlp_gm_75/model_pruned.pth \
  --teacher-config configs/railsem19/segformer_b1_rs19_512x512_100ep_rtx4090.py \
  --teacher-checkpoint /root/bishe/runs/B1_best_mIoU.pth \
  --device cuda:0 \
  --work-dir runs/rs19/b0_p75_ft50_kd_logit \
  --finetune-epochs 50 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --distill logit \
  --kd-temperature 4.0 \
  --kd-loss-weight 0.05
```

### B0-P75 + Dice
```
python src/finetune_dice.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  --pruned-model output/segformer_b0_mlp_gm_75/model_pruned.pth \
  --device cuda:0 \
  --work-dir runs/rs19/b0_p75_ft50_ce_dice \
  --finetune-epochs 50 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --ce-loss-weight 1.0 \
  --dice-loss-weight 0.5

### 在Dice loss的微调结果基础上继续进行KD蒸馏：

python src/finetune_kd.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  --pruned-model runs/rs19/b0_p75_ft50_ce_dice/best_full_model.pth \
  --teacher-config configs/railsem19/segformer_b1_rs19_512x512_100ep_rtx4090.py \
  --teacher-checkpoint /root/bishe/runs/B1_best_mIoU.pth \
  --device cuda:0 \
  --work-dir runs/rs19/b0_p75_ft50_ce_dice_ft50_kd_logit \
  --finetune-epochs 50 \
  --lr 2e-5 \
  --weight-decay 0.01 \
  --distill logit \
  --kd-temperature 4.0 \
  --kd-loss-weight 0.05
```

### B0-75 参数测量脚本 
评估蒸馏
```
python src/eval.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  --model runs/rs19/b0_p75_ft50_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --work-dir runs/rs19/eval_b0_p75_ft50_kd_logit \
  --output-json output/segformer_b0_mlp_gm_75/metrics_ft50_kd_logit.json

python src/flops.py \
  --model runs/rs19/b0_p75_ft50_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --output-json output/segformer_b0_mlp_gm_75/flopss_ft50_kd_logit.json

python src/latency.py \
  --model runs/rs19/b0_p75_ft50_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --repeat 300 \
  --output-json output/segformer_b0_mlp_gm_75/latency_ft50_kd_logit.json

```

评估蒸馏+Dice
```
python src/eval.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  --model runs/rs19/b0_p75_ft50_ce_dice_ft50_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --work-dir runs/rs19/eval_b0_p75_ft50_ce_dice_ft50_kd_logit \
  --output-json output/segformer_b0_mlp_gm_75/metrics_ft50_ce_dice_ft50_kd_logit.json

python src/flops.py \
  --model runs/rs19/b0_p75_ft50_ce_dice_ft50_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --output-json output/segformer_b0_mlp_gm_75/flops_ft50_ce_dice_ft50_kd_logit.json

python src/latency.py \
  --model runs/rs19/b0_p75_ft50_ce_dice_ft50_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --repeat 300 \
  --output-json output/segformer_b0_mlp_gm_75/latency_ft50_ce_dice_ft50_kd_logit.json
```

### B0 global pruning
### b0\_p30\_global

```
   python src/run_exp.py --exp b0_p30_global --task prune
   python src/run_exp.py --exp b0_p30_global --task eval --variant pruned
   python src/run_exp.py --exp b0_p30_global --task flops --variant pruned
   python src/run_exp.py --exp b0_p30_global --task latency --variant pruned
   python src/run_exp.py --exp b0_p30_global --task finetune
   python src/run_exp.py --exp b0_p30_global --task eval --variant ft
   python src/run_exp.py --exp b0_p30_global --task flops --variant ft
   python src/run_exp.py --exp b0_p30_global --task latency --variant ft
```
### b0\_p50\_global

```
   python src/run_exp.py --exp b0_p50_global --task prune
   python src/run_exp.py --exp b0_p50_global --task eval --variant pruned
   python src/run_exp.py --exp b0_p50_global --task flops --variant pruned
   python src/run_exp.py --exp b0_p50_global --task latency --variant pruned
   python src/run_exp.py --exp b0_p50_global --task finetune
   python src/run_exp.py --exp b0_p50_global --task eval --variant ft
   python src/run_exp.py --exp b0_p50_global --task flops --variant ft
   python src/run_exp.py --exp b0_p50_global --task latency --variant ft
```

### B0 global+iso pruning
### b0\_p30\_global\_iso
```
  python src/run_exp.py --exp b0_p30_global_iso --task prune
  python src/run_exp.py --exp b0_p30_global_iso --task eval --variant pruned
  python src/run_exp.py --exp b0_p30_global_iso --task flops --variant pruned
  python src/run_exp.py --exp b0_p30_global_iso --task latency --variant pruned
  python src/run_exp.py --exp b0_p30_global_iso --task finetune
  python src/run_exp.py --exp b0_p30_global_iso --task eval --variant ft
  python src/run_exp.py --exp b0_p30_global_iso --task flops --variant ft
  python src/run_exp.py --exp b0_p30_global_iso --task latency --variant ft
```
### b0\_p50\_global\_iso
```
  python src/run_exp.py --exp b0_p50_global_iso --task prune
  python src/run_exp.py --exp b0_p50_global_iso --task eval --variant pruned
  python src/run_exp.py --exp b0_p50_global_iso --task flops --variant pruned
  python src/run_exp.py --exp b0_p50_global_iso --task latency --variant pruned
  python src/run_exp.py --exp b0_p50_global_iso --task finetune
  python src/run_exp.py --exp b0_p50_global_iso --task eval --variant ft
  python src/run_exp.py --exp b0_p50_global_iso --task flops --variant ft
  python src/run_exp.py --exp b0_p50_global_iso --task latency --variant ft
  
```

### b0\_p70\_global\_iso
```
  python src/run_exp.py --exp b0_p70_global_iso --task prune
  python src/run_exp.py --exp b0_p70_global_iso --task eval --variant pruned
  python src/run_exp.py --exp b0_p70_global_iso --task flops --variant pruned
  python src/run_exp.py --exp b0_p70_global_iso --task latency --variant pruned
  python src/run_exp.py --exp b0_p70_global_iso --task finetune
  python src/run_exp.py --exp b0_p70_global_iso --task eval --variant ft
  python src/run_exp.py --exp b0_p70_global_iso --task flops --variant ft
  python src/run_exp.py --exp b0_p70_global_iso --task latency --variant ft
```

### b0\_p75\_global\_iso
```
  python src/run_exp.py --exp b0_p75_global_iso --task prune
  python src/run_exp.py --exp b0_p75_global_iso --task eval --variant pruned
  python src/run_exp.py --exp b0_p75_global_iso --task flops --variant pruned
  python src/run_exp.py --exp b0_p75_global_iso --task latency --variant pruned
  python src/run_exp.py --exp b0_p75_global_iso --task finetune
  python src/run_exp.py --exp b0_p75_global_iso --task eval --variant ft
  python src/run_exp.py --exp b0_p75_global_iso --task flops --variant ft
  python src/run_exp.py --exp b0_p75_global_iso --task latency --variant ft
```

### b0_uni_p30_global_iso
```
  python src/run_exp.py --exp b0_uni_p30_global_iso --task prune
  python src/run_exp.py --exp b0_uni_p30_global_iso --task eval --variant pruned
  python src/run_exp.py --exp b0_uni_p30_global_iso --task flops --variant pruned
  python src/run_exp.py --exp b0_uni_p30_global_iso --task latency --variant pruned
  python src/run_exp.py --exp b0_uni_p30_global_iso --task finetune
  python src/run_exp.py --exp b0_uni_p30_global_iso --task eval --variant ft
  python src/run_exp.py --exp b0_uni_p30_global_iso --task flops --variant ft
  python src/run_exp.py --exp b0_uni_p30_global_iso --task latency --variant ft
```
### b0_uni_p50_global_iso
```
  python src/run_exp.py --exp b0_uni_p50_global_iso --task prune
  python src/run_exp.py --exp b0_uni_p50_global_iso --task eval --variant pruned
  python src/run_exp.py --exp b0_uni_p50_global_iso --task flops --variant pruned
  python src/run_exp.py --exp b0_uni_p50_global_iso --task latency --variant pruned
  python src/run_exp.py --exp b0_uni_p50_global_iso --task finetune
  python src/run_exp.py --exp b0_uni_p50_global_iso --task eval --variant ft
  python src/run_exp.py --exp b0_uni_p50_global_iso --task flops --variant ft
  python src/run_exp.py --exp b0_uni_p50_global_iso --task latency --variant ft
```



### B0-dice 
```
  python src/finetune_dice.py \
    --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
    --pruned-model runs/rs19/segformer_b0_mlp_gm_50_ft/best_full_model.pth \
    --device cuda:0 \
    --work-dir runs/rs19/b0_pruned_ft_ce_dice \
    --finetune-epochs 50 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --ce-loss-weight 1.0 \
    --dice-loss-weight 0.5
```
#参数测量脚本
```
python src/eval.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  --model runs/rs19/b1_p50_ft100_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --work-dir runs/tmp_eval_b1_mlp_gm_50_pruned_kd \
  --output-json output/segformer_b1_mlp_gm_50_kd_ft100/metrics_pruned.json

python src/flops.py \
  --model runs/rs19/b1_p50_ft100_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --output-json output/segformer_b1_mlp_gm_50_kd_ft100/flops_pruned_bs1.json

python src/latency.py \
  --model runs/rs19/b1_p50_ft100_kd_logit/best_full_model.pth \
  --device cuda:0 \
  --shape 512 512 \
  --batch-size 1 \
  --repeat 300 \
  --output-json output/segformer_b1_mlp_gm_50_kd_ft100/latency_pruned_bs1.json
```

---

## 三、B1 Local 剪枝实验

### b1\_p10  只剪枝不微调

```
 python src/run_exp.py --exp b1_p10 --task prune
 python src/run_exp.py --exp b1_p10 --task eval --variant pruned
 python src/run_exp.py --exp b1_p10 --task flops --variant pruned
 python src/run_exp.py --exp b1_p10 --task latency --variant pruned
 
 python src/run_exp.py --exp b1_p10 --task finetune
 
 python src/run_exp.py --exp b1_p10 --task eval --variant ft
 python src/run_exp.py --exp b1_p10 --task flops --variant ft
 python src/run_exp.py --exp b1_p10 --task latency --variant ft
```

### b1\_p30 只剪枝不微调

```
 python src/run_exp.py --exp b1_p30 --task prune
 python src/run_exp.py --exp b1_p30 --task eval --variant pruned
 python src/run_exp.py --exp b1_p30 --task flops --variant pruned
 python src/run_exp.py --exp b1_p30 --task latency --variant pruned
 
 python src/run_exp.py --exp b1_p30 --task finetune
 
 python src/run_exp.py --exp b1_p30 --task eval --variant ft
 python src/run_exp.py --exp b1_p30 --task flops --variant ft
 python src/run_exp.py --exp b1_p30 --task latency --variant ft
```

### b1\_p50  只剪枝不微调

```
 python src/run_exp.py --exp b1_p50 --task prune
 python src/run_exp.py --exp b1_p50 --task eval --variant pruned
 python src/run_exp.py --exp b1_p50 --task flops --variant pruned
 python src/run_exp.py --exp b1_p50 --task latency --variant pruned
 
 python src/run_exp.py --exp b1_p50 --task finetune
 
 python src/run_exp.py --exp b1_p50 --task eval --variant ft
 python src/run_exp.py --exp b1_p50 --task flops --variant ft
 python src/run_exp.py --exp b1_p50 --task latency --variant ft
```

### b1\_p70

```
python src/run_exp.py --exp b1_p70 --task prune
python src/run_exp.py --exp b1_p70 --task eval --variant pruned
python src/run_exp.py --exp b1_p70 --task flops --variant pruned
python src/run_exp.py --exp b1_p70 --task latency --variant pruned

python src/run_exp.py --exp b1_p70 --task finetune

python src/run_exp.py --exp b1_p70 --task eval --variant ft
python src/run_exp.py --exp b1_p70 --task flops --variant ft
python src/run_exp.py --exp b1_p70 --task latency --variant ft
```

### b1\_p90 剪枝加微调60epoch

```
python src/run_exp.py --exp b1_p90 --task prune
python src/run_exp.py --exp b1_p90 --task eval --variant pruned
python src/run_exp.py --exp b1_p90 --task flops --variant pruned
python src/run_exp.py --exp b1_p90 --task latency --variant pruned

python src/run_exp.py --exp b1_p90 --task finetune

python src/run_exp.py --exp b1_p90 --task eval --variant ft
python src/run_exp.py --exp b1_p90 --task flops --variant ft
python src/run_exp.py --exp b1_p90 --task latency --variant ft
```

---

## 四、B1 Global 剪枝实验

### b1\_p30\_global  只剪枝不微调

```
python src/run_exp.py --exp b1_p30_global --task prune
python src/run_exp.py --exp b1_p30_global --task eval --variant pruned
python src/run_exp.py --exp b1_p30_global --task flops --variant pruned
python src/run_exp.py --exp b1_p30_global --task latency --variant pruned

python src/run_exp.py --exp b1_p30_global --task finetune

python src/run_exp.py --exp b1_p30_global --task eval --variant ft
python src/run_exp.py --exp b1_p30_global --task flops --variant ft
python src/run_exp.py --exp b1_p30_global --task latency --variant ft
```

### b1\_p30\_global\_iso

```
python src/run_exp.py --exp b1_p30_global_iso --task prune
python src/run_exp.py --exp b1_p30_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_p30_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_p30_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_p30_global_iso --task finetune

python src/run_exp.py --exp b1_p30_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_p30_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_p30_global_iso --task latency --variant ft
```

### b1\_p50\_global

```
python src/run_exp.py --exp b1_p50_global --task prune
python src/run_exp.py --exp b1_p50_global --task eval --variant pruned
python src/run_exp.py --exp b1_p50_global --task flops --variant pruned
python src/run_exp.py --exp b1_p50_global --task latency --variant pruned

python src/run_exp.py --exp b1_p50_global --task finetune

python src/run_exp.py --exp b1_p50_global --task eval --variant ft
python src/run_exp.py --exp b1_p50_global --task flops --variant ft
python src/run_exp.py --exp b1_p50_global --task latency --variant ft
```

### b1\_p50\_global\_iso

```
python src/run_exp.py --exp b1_p50_global_iso --task prune
python src/run_exp.py --exp b1_p50_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_p50_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_p50_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_p50_global_iso --task finetune

python src/run_exp.py --exp b1_p50_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_p50_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_p50_global_iso --task latency --variant ft
```

### b1\_p90\_global\_iso
```
python src/run_exp.py --exp b1_p90_global_iso --task prune
python src/run_exp.py --exp b1_p90_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_p90_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_p90_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_p90_global_iso --task finetune

python src/run_exp.py --exp b1_p90_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_p90_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_p90_global_iso --task latency --variant ft
```

---

方案A: uniform_linear + [2,3,4]
### b1_uni_p30_global_iso   epoch = 20
```
python src/run_exp.py --exp b1_uni_p30_global_iso --task prune
python src/run_exp.py --exp b1_uni_p30_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_uni_p30_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_uni_p30_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_uni_p30_global_iso --task finetune

python src/run_exp.py --exp b1_uni_p30_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_uni_p30_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_uni_p30_global_iso --task latency --variant ft
```

### b1_uni_p50_global_iso  epoch = 100
```
python src/run_exp.py --exp b1_uni_p50_global_iso --task prune
python src/run_exp.py --exp b1_uni_p50_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_uni_p50_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_uni_p50_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_uni_p50_global_iso --task finetune

python src/run_exp.py --exp b1_uni_p50_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_uni_p50_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_uni_p50_global_iso --task latency --variant ft
```

### b1_uni_p70_global_iso  epoch = 60
```
python src/run_exp.py --exp b1_uni_p70_global_iso --task prune
python src/run_exp.py --exp b1_uni_p70_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_uni_p70_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_uni_p70_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_uni_p70_global_iso --task finetune

python src/run_exp.py --exp b1_uni_p70_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_uni_p70_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_uni_p70_global_iso --task latency --variant ft
```

方案B: mlp_bottleneck + [1,2,3,4]
### b1_mlp_all_p50_global_iso
```
python src/run_exp.py --exp b1_mlp_all_p50_global_iso --task prune
python src/run_exp.py --exp b1_mlp_all_p50_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_mlp_all_p50_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_mlp_all_p50_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_mlp_all_p50_global_iso --task finetune

python src/run_exp.py --exp b1_mlp_all_p50_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_mlp_all_p50_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_mlp_all_p50_global_iso --task latency --variant ft

```

### b1_mlp_all_p70_global_iso
```
python src/run_exp.py --exp b1_mlp_all_p70_global_iso --task prune
python src/run_exp.py --exp b1_mlp_all_p70_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_mlp_all_p70_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_mlp_all_p70_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_mlp_all_p70_global_iso --task finetune

python src/run_exp.py --exp b1_mlp_all_p70_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_mlp_all_p70_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_mlp_all_p70_global_iso --task latency --variant ft
```

方案C: uniform_linear + [1,2,3,4]
### b1_uni_all_p30_global_iso
```
python src/run_exp.py --exp b1_uni_all_p30_global_iso --task prune
python src/run_exp.py --exp b1_uni_all_p30_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_uni_all_p30_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_uni_all_p30_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_uni_all_p30_global_iso --task finetune

python src/run_exp.py --exp b1_uni_all_p30_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_uni_all_p30_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_uni_all_p30_global_iso --task latency --variant ft

```

### b1_uni_all_p50_global_iso
```
python src/run_exp.py --exp b1_uni_all_p50_global_iso --task prune
python src/run_exp.py --exp b1_uni_all_p50_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_uni_all_p50_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_uni_all_p50_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_uni_all_p50_global_iso --task finetune

python src/run_exp.py --exp b1_uni_all_p50_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_uni_all_p50_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_uni_all_p50_global_iso --task latency --variant ft
```

### b1_uni_all_p70_global_iso
```
python src/run_exp.py --exp b1_uni_all_p70_global_iso --task prune
python src/run_exp.py --exp b1_uni_all_p70_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_uni_all_p70_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_uni_all_p70_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_uni_all_p70_global_iso --task finetune

python src/run_exp.py --exp b1_uni_all_p70_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_uni_all_p70_global_iso --task flops
python src/run_exp.py --exp b1_uni_all_p70_global_iso --task latency --variant ft
```


## 五、B1 Taylor Importance 实验

### b1\_p30\_taylor

```
python src/run_exp.py --exp b1_p30_taylor --task prune
python src/run_exp.py --exp b1_p30_taylor --task eval --variant pruned
python src/run_exp.py --exp b1_p30_taylor --task flops --variant pruned
python src/run_exp.py --exp b1_p30_taylor --task latency --variant pruned

python src/run_exp.py --exp b1_p30_taylor --task finetune

python src/run_exp.py --exp b1_p30_taylor --task eval --variant ft
python src/run_exp.py --exp b1_p30_taylor --task flops --variant ft
python src/run_exp.py --exp b1_p30_taylor --task latency --variant ft
```

### b1\_p30\_taylor\_global

```
python src/run_exp.py --exp b1_p30_taylor_global --task prune
python src/run_exp.py --exp b1_p30_taylor_global --task eval --variant pruned
python src/run_exp.py --exp b1_p30_taylor_global --task flops --variant pruned
python src/run_exp.py --exp b1_p30_taylor_global --task latency --variant pruned

python src/run_exp.py --exp b1_p30_taylor_global --task finetune

python src/run_exp.py --exp b1_p30_taylor_global --task eval --variant ft
python src/run_exp.py --exp b1_p30_taylor_global --task flops --variant ft
python src/run_exp.py --exp b1_p30_taylor_global --task latency --variant ft
```

### b1\_p30\_taylor\_global\_iso

```
python src/run_exp.py --exp b1_p30_taylor_global_iso --task prune
python src/run_exp.py --exp b1_p30_taylor_global_iso --task eval --variant pruned
python src/run_exp.py --exp b1_p30_taylor_global_iso --task flops --variant pruned
python src/run_exp.py --exp b1_p30_taylor_global_iso --task latency --variant pruned

python src/run_exp.py --exp b1_p30_taylor_global_iso --task finetune

python src/run_exp.py --exp b1_p30_taylor_global_iso --task eval --variant ft
python src/run_exp.py --exp b1_p30_taylor_global_iso --task flops --variant ft
python src/run_exp.py --exp b1_p30_taylor_global_iso --task latency --variant ft
```

---

## 六、快速检查命令

```
python src/run_exp.py --exp b1_p30 --task prune --dry-run
python src/run_exp.py --exp b1_p30_global_iso --task prune --dry-run
python src/run_exp.py --exp b1_p30_taylor --task prune --dry-run
python src/run_exp.py --exp b1_p30_taylor_global_iso --task prune --dry-run

python src/run_exp.py --exp b1_p30 --task eval --variant pruned --dry-run
python src/run_exp.py --exp b1_p30 --task flops --variant pruned --dry-run
python src/run_exp.py --exp b1_p30 --task latency --variant pruned --dry-run
```

---



### 蒸馏的命令

### logit(加上微调100epoch，学习率3e-5，权重衰减0.01，蒸馏温度4.0，蒸馏损失权重1.0)
```
python src/finetune_kd.py \
  --config configs/railsem19/segformer_b1_rs19_512x512_100ep_rtx4090.py \
  --pruned-model output/segformer_b1_uni_gm_50_gi/model_pruned.pth \
  --teacher-checkpoint runs/rs19/segformer_b1_512x512_100ep_rtx4090/best_mIoU.pth \
  --device cuda:0 \
  --work-dir runs/rs19/b1_p50_ft5_kd_logit \
  --finetune-epochs 5 \
  --lr 3e-5 \
  --weight-decay 0.01 \
  --distill logit \
  --kd-temperature 4.0 \
  --kd-loss-weight 0.3
```

### b0\_p70\_global\_iso  蒸馏
```
python src/finetune_kd.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_150ep_rtx4090.py \
  --pruned-model output/segformer_b0_mlp_gm_70_gi/model_pruned.pth \
  --teacher-config configs/railsem19/segformer_b1_rs19_512x512_100ep_rtx4090.py \
  --teacher-checkpoint runs/rs19/segformer_b1_512x512_100ep_rtx4090/best_mIoU.pth \
  --device cuda:0 \
  --work-dir runs/rs19/b0_mlp_gm_70_gi_ft100_kd_logit_T2_w01 \
  --finetune-epochs 100 \
  --lr 3e-5 \
  --weight-decay 0.01 \
  --distill logit \
  --kd-temperature 2.0 \
  --kd-loss-weight 0.1

```

### logit+cwd
```
python src/finetune_kd.py \
  --config configs/railsem19/segformer_b1_rs19_512x512_100ep_rtx4090.py \
  --pruned-model output/segformer_b1_uni_gm_50_gi/model_pruned.pth \
  --teacher-checkpoint runs/rs19/segformer_b1_512x512_100ep_rtx4090/best_mIoU.pth \
  --device cuda:0 \
  --work-dir runs/rs19/b1_p50_ft100_kd_logit_cwd \
  --finetune-epochs 100 \
  --lr 3e-5 \
  --weight-decay 0.01 \
  --distill logit+cwd \
  --kd-temperature 4.0 \
  --kd-loss-weight 0.05 \
  --cwd-tau 1.0 \
  --cwd-loss-weight 1.0 \
  --cwd-feature-index -1

```



### 模型ONNX导出
```
MODEL_PATH="runs/rs19/b0_pruned_ft_ce_dice/best_full_model.pth"

PYTHONPATH=src python deploy/scripts/export_pruned_to_onnx.py \
  --model "$MODEL_PATH" \
  --output deploy/onnx/b0_mlp_gm50_ce_dice_512.onnx \
  --height 512 \
  --width 512 \
  --opset 13 \
  --device cuda:0 \
  2>&1 | tee deploy/logs/export_b0_mlp_gm50_ce_dice_512.log
```

###  简化ONNX模型
```
python -m onnxsim \
  deploy/onnx/b0_mlp_gm50_ce_dice_512.onnx \
  deploy/onnx/b0_mlp_gm50_ce_dice_512_sim.onnx \
  2>&1 | tee deploy/logs/simplify_b0_mlp_gm50_ce_dice_512.log

Simplifying...
Finish! Here is the difference:
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃            ┃ Original Model ┃ Simplified Model ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Add        │ 108            │ 100              │
│ Concat     │ 31             │ 1                │
│ Constant   │ 508            │ 224              │
│ Conv       │ 40             │ 40               │
│ Div        │ 46             │ 46               │
│ Erf        │ 8              │ 8                │
│ Gather     │ 16             │ 16               │
│ Gemm       │ 8              │ 8                │
│ MatMul     │ 32             │ 32               │
│ Mod        │ 8              │ 0                │
│ Mul        │ 46             │ 46               │
│ Pow        │ 30             │ 30               │
│ ReduceMean │ 60             │ 60               │
│ Relu       │ 5              │ 5                │
│ Reshape    │ 100            │ 78               │
│ Resize     │ 4              │ 4                │
│ Shape      │ 30             │ 0                │
│ Slice      │ 38             │ 0                │
│ Softmax    │ 8              │ 8                │
│ Sqrt       │ 30             │ 30               │
│ Squeeze    │ 8              │ 8                │
│ Sub        │ 30             │ 30               │
│ Transpose  │ 98             │ 98               │
│ Unsqueeze  │ 8              │ 8                │
│ Model Size │ 11.4MiB        │ 11.4MiB          │
└────────────┴────────────────┴──────────────────┘

python -m onnxsim \
  deploy/onnx/b0_mlp_gm50_ce_dice_512_upsample.onnx \
  deploy/onnx/b0_mlp_gm50_ce_dice_512_upsample_sim.onnx \
  2>&1 | tee deploy/logs/simplify_b0_mlp_gm50_ce_dice_512_upsample.log
Simplifying...
Finish! Here is the difference:
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃            ┃ Original Model ┃ Simplified Model ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Add        │ 108            │ 100              │
│ Concat     │ 32             │ 1                │
│ Constant   │ 512            │ 225              │
│ Conv       │ 40             │ 40               │
│ Div        │ 46             │ 46               │
│ Erf        │ 8              │ 8                │
│ Gather     │ 16             │ 16               │
│ Gemm       │ 8              │ 8                │
│ MatMul     │ 32             │ 32               │
│ Mod        │ 8              │ 0                │
│ Mul        │ 46             │ 46               │
│ Pow        │ 30             │ 30               │
│ ReduceMean │ 60             │ 60               │
│ Relu       │ 5              │ 5                │
│ Reshape    │ 100            │ 78               │
│ Resize     │ 5              │ 5                │
│ Shape      │ 31             │ 0                │
│ Slice      │ 39             │ 0                │
│ Softmax    │ 8              │ 8                │
│ Sqrt       │ 30             │ 30               │
│ Squeeze    │ 8              │ 8                │
│ Sub        │ 30             │ 30               │
│ Transpose  │ 98             │ 98               │
│ Unsqueeze  │ 8              │ 8                │
│ Model Size │ 11.4MiB        │ 11.4MiB          │
└────────────┴────────────────┴──────────────────┘
```