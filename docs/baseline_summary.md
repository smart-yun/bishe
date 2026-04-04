# SegFormer-B0 基线模型总结 (Baseline Summary)

**版本信息:**
- **模型:** SegFormer-B0
- **配置文件:** `configs/railsem19/segformer_b0_rs19_512x512_40000it.py`
- **数据集:** RailSem19
- **输入尺寸:** 512x512
- **最佳权重:** `runs/rs19/segformer_b0_512x512_40000it/best_mIoU_iter_40000.pth`

---

## 1. 核心性能指标

| 指标 (Metric) | 数值 (Value) | 单位 (Unit) | 备注 (Notes) |
| :--- | :--- | :--- | :--- |
| **mIoU** | `57.17` | % | 在 RailSem19 验证集上评估 |
| **参数量 (Params)** | `3.72` | M | 512x512 输入下统计（mmengine complexity） |
| **计算量 (FLOPs)** | `7.96` | G | 512x512 输入下统计（mmengine complexity） |
| **推理延迟 (Latency)** | `35.91` | ms | 本机 RTX 3050，batch=1，warmup=5，iters=200 |

---

## 2. 优点 (Pros)

- **精度较高:** 作为轻量级模型，在 RailSem19 数据集上达到了一个很不错的 mIoU 基准。
- **结构现代:** 采用 Transformer 结构，没有复杂的解码器头，便于后续部署。
- **社区成熟:** 基于 MMSegmentation，生态完善，有大量的文档和工具支持。

---

## 3. 待优化点 (Cons / Opportunities)

- **参数量仍有压缩空间:** 对于端侧部署（如 Jetson Orin），`3.72`M 的参数量仍可继续压缩，是结构化剪枝的主要目标。
- **推理速度未达极致:** `35.91`ms（约 `27.85 FPS`）距离实时目标（< 33ms）还有差距，需要通过剪枝和量化进行加速。
- **对小目标和边缘分割存在不足:** 从可视化结果看，模型对一些细长的轨道、信号灯以及类别边缘的处理不够精细，这是后续需要关注的精度保持点。

---

## 4. 初步结论

该基线模型成功跑通了数据、训练、评估的全流程，并提供了一个可靠的性能锚点。**下一阶段的核心任务是：在尽可能保持 mIoU 的前提下，显著降低参数量和推理延迟。**

---

## 5. 80k 训练曲线补充（2026-04-04）

> 说明：以下为 80k server 训练任务的曲线留痕，属于在同任务设定下对训练轮次延展后的结果记录。

- 曲线图：`results/curves/miou_curve_80k.png`
- 摘要：`results/curves/miou_curve_80k_summary.json`
- 训练日志：`runs/rs19/segformer_b0_512x512_80000it_server/20260404_021343/vis_data/20260404_021343.json`

### 5.1 关键统计

| 指标 | 数值 |
| :--- | :--- |
| `best_miou` | **59.75** |
| `best_step` | 79000 |
| `last_miou` | 59.54 |
| `last_step` | 80000 |
| `miou_drop_after_best` | 0.21 |
| `critical_found` | false |

### 5.2 解读

- mIoU 曲线整体随迭代稳步上升，后段趋于平台。
- 最优点在 79k，最终点仅轻微回落，未触发过拟合关键点检测。
- 若用于后续压缩/剪枝对比，建议以 `best_mIoU_iter_79000.pth` 作为 80k 阶段参考权重。

### 5.3 复杂度与推理速度（best@79k）

> 数据来源：`exports/metrics_80k_best_79000.json`

| 指标 (Metric) | 数值 (Value) | 单位 (Unit) | 备注 (Notes) |
| :--- | :--- | :--- | :--- |
| **参数量 (Params)** | `3.720051` | M | 与基线同量级（SegFormer-B0） |
| **计算量 (FLOPs)** | `7.956135936` | G | 输入尺寸 512x512 |
| **推理延迟 (Latency mean)** | `9.4368` | ms | `cuda:0`，warmup=5，timed=195 |
| **推理帧率 (FPS mean)** | `105.968` | FPS | 与上述时延统计同设置 |
