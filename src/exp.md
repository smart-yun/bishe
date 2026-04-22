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
