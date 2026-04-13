# RailSem19 + SegFormer Pruning Experiment Record

## 1. Experiment Overview

| ID | Stage | Model | Setting | Status | Best mIoU | Params(M) | MACs/FLOPs | Latency(ms) | FPS | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| A1 | Baseline | SegFormer-B0 | baseline | Not Started |  |  |  |  |  |  |
| A2 | Baseline | SegFormer-B1 | baseline | Not Started |  |  |  |  |  |  |
| B1 | Prune | SegFormer-B1 | mlp_bottleneck + group_magnitude + ratio=0.10 | Done |  | 13.06 |  |  |  | params reduced successfully |
| B2 | Prune | SegFormer-B1 | mlp_bottleneck + group_magnitude + ratio=0.15 | Not Started |  |  |  |  |  |  |
| B3 | Prune | SegFormer-B1 | mlp_bottleneck + group_magnitude + ratio=0.20 | Not Started |  |  |  |  |  |  |
| C1 | FT | SegFormer-B1 | prune0.10 + finetune | Not Started |  |  |  |  |  |  |
| C2 | FT | SegFormer-B1 | prune0.15 + finetune | Not Started |  |  |  |  |  |  |
| C3 | FT | SegFormer-B1 | prune0.20 + finetune | Not Started |  |  |  |  |  |  |
| D1 | KD | SegFormer-B1 | prune-best + finetune + KD | Not Started |  |  |  |  |  |  |
| E1 | Sanity | SegFormer-B0 | mlp_bottleneck + group_magnitude + ratio=0.05 | Not Started |  |  |  |  |  |  |
| E2 | Sanity | SegFormer-B0 | mlp_bottleneck + group_magnitude + ratio=0.10 | Not Started |  |  |  |  |  |  |
| E3 | Sanity | SegFormer-B0 | mlp_bottleneck + group_magnitude + ratio=0.15 | Not Started |  |  |  |  |  |  |

---

## 2. Fixed Experimental Settings

### 2.1 Dataset
| Item | Value |
|---|---|
| Dataset | RailSem19 |
| Task | Semantic Segmentation |
| Input Size | 512 ¡Á 512 |
| Train Split |  |
| Val Split |  |
| Test Split |  |

### 2.2 Key Classes
| Class |
|---|
| rail-track |
| trackbed |
| rail-raised |
| rail-embedded |
| on-rails |

### 2.3 Common Metrics
| Metric |
|---|
| mIoU |
| mAcc |
| aAcc |
| Per-class IoU |
| Params |
| MACs / FLOPs |
| Latency |
| FPS |
| GPU Memory |

---

## 3. Detailed Records

---

# Experiment: A1_B0_BASE

## 3.1 Basic Info

| Item | Value |
|---|---|
| Experiment ID | A1_B0_BASE |
| Model | SegFormer-B0 |
| Purpose | Lightweight baseline |
| Status |  |
| Date |  |
| Operator |  |

## 3.2 Config

| Item | Value |
|---|---|
| Config Path |  |
| Checkpoint Path |  |
| Work Dir |  |
| Batch Size |  |
| Epochs / Iters |  |
| Input Size | 512x512 |
| Pretrained or Scratch |  |

## 3.3 Commands

### Train
```bash