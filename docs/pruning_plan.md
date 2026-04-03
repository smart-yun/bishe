# SegFormer 结构化剪枝计划 (Pruning Plan)

## 1. 研究目标

- **主要目标：** 在尽量保持分割精度的前提下，降低 SegFormer-B0 的参数量和 FLOPs。
- **次要目标：** 为后续的部署、量化和端侧推理做准备。
- **期望指标：**
  - 模型压缩率 > 40%
  - Jetson Orin NX 上推理速度 > 30 FPS

---

## 2. 剪枝对象分析

SegFormer 的主干网络是层级式 Transformer Encoder。每个 Encoder 由若干个 Transformer Block 堆叠而成。本计划优先分析并剪枝 **Transformer Block** 内部的模块。

每个 Transformer Block 主要包含两部分：

1. **多头自注意力（Multi-Head Self-Attention, MSA）**
2. **前馈网络（Feed-Forward Network, FFN）**，在 SegFormer 中通常称为 `MixFFN`

### 可剪枝的关键层

- **MSA 部分：**
  - `attn.qkv`：通常是 `nn.Linear`，可用于剪枝注意力头相关通道。
  - `attn.proj`：通常是 `nn.Linear`，其输入/输出通道会受到 `qkv` 剪枝影响。

- **FFN（MixFFN）部分：**
  - `ffn.layers[0]`：第一个 `1x1 Conv`，用于扩展通道数，适合做通道剪枝。
  - `ffn.layers[1]`：深度可分离卷积中的 depthwise `3x3 Conv`，需要跟随前一层通道变化。
  - `ffn.layers[4]`：最后一个 `1x1 Conv`，用于投影回原通道数。

### 结构示意

```mermaid
graph TD
    subgraph Transformer Block
        A[Input] --> B{LayerNorm}
        B --> C[MSA]
        C --> D[Add]
        A --> D
        D --> E{LayerNorm}
        E --> F[MixFFN]
        F --> G[Add]
        D --> G
        G --> H[Output]
    end

    subgraph MSA
        C_IN[Input] --> C1[qkv (Linear)]
        C1 --> C2[Attention Heads]
        C2 --> C3[proj (Linear)]
        C3 --> C_OUT[Output]
    end

    subgraph MixFFN
        F_IN[Input] --> F1[Conv 1x1]
        F1 --> F2[Depthwise Conv 3x3]
        F2 --> F3[GELU]
        F3 --> F4[Conv 1x1]
        F4 --> F_OUT[Output]
    end
```

> 结论：**FFN/MixFFN 是当前最适合优先尝试结构化剪枝的部分**，因为其通道依赖关系清晰，且对 `torch-pruning` 更友好。

---

## 3. 剪枝策略

- **工具：** `torch-pruning`
- **依赖分析：** 使用 `torch_pruning.DependencyGraph` 构建模型依赖图。
- **重要性评估：** 优先使用 L1-norm 或 `torch_pruning.importance.MagnitudeImportance`。
- **剪枝原则：**
  1. 先从 FFN 的通道剪枝入手。
  2. 再尝试 MSA 中的注意力头或相关线性层。
  3. 控制剪枝比例，避免一次性剪得太狠。
  4. 剪枝后进行微调（fine-tuning）恢复精度。

---

## 4. 实验阶段规划

### 阶段一：轻量级剪枝（10% - 20%）

- **目标：** 验证结构化剪枝流程是否稳定。
- **建议比例：** 20%
- **验证方式：**
  1. 在 SegFormer-B0 上进行 20% 剪枝。
  2. 观察 mIoU 是否大幅下降。
  3. 进行 5000 - 10000 iterations 的短微调。
  4. 统计 FLOPs 和 mIoU 的变化。

### 阶段二：中等强度剪枝（30% - 50%）

- **目标：** 找到速度与精度之间的平衡点。
- **备选方案：**
  - **方案 A：** 只剪 FFN
  - **方案 B：** 只剪 MSA
  - **方案 C：** FFN + MSA 组合剪枝
- **验证方式：**
  1. 分别尝试 A、B、C 三种方案。
  2. 比较精度、速度和参数量。
  3. 选择综合表现最优的方案继续推进。

---

## 5. 风险控制

- **风险 1：** 剪枝后模型结构不再兼容原始 checkpoint。
  - **对策：** 每次剪枝后立即保存新模型，并记录层结构变化。

- **风险 2：** 剪枝后精度下降明显。
  - **对策：** 采用更温和的剪枝比例，并增加微调轮数。

- **风险 3：** `torch-pruning` 对 SegFormer 的某些算子支持不完全。
  - **对策：** 利用 `DependencyGraph` 的依赖追踪，必要时手动处理特殊依赖。

---

## 6. 实验记录（Experiment Logs）

### 实验 1：`prune_test.py` 的试剪

- **日期：** 2026-04-02
- **目的：** 验证 `torch-pruning` 是否能正确剪枝 SegFormer-B0 的 `MixFFN`。
- **脚本：** `src/prune_test.py`

#### 配置

- **目标层：** `backbone.layers.0.1.0.ffn.layers.0`
- **剪枝比例：** `0.1`（10%）
- **剪枝准则：** L1-norm

#### 结果

1. 成功加载模型。
2. 成功构建依赖图。
3. 成功完成剪枝，并保存为 `checkpoints/pruned_test_model.pth`。

#### 剪枝前后对比

- **剪枝前：**
  - `ffn.layers[0]`: `Conv2d(32, 128, kernel_size=(1, 1))`
  - `ffn.layers[1]`: `Conv2d(128, 128, kernel_size=(3, 3), groups=128)`
  - `ffn.layers[4]`: `Conv2d(128, 32, kernel_size=(1, 1))`

- **剪枝后：**
  - `ffn.layers[0]`: `Conv2d(32, 116, kernel_size=(1, 1))`
  - `ffn.layers[1]`: `Conv2d(116, 116, kernel_size=(3, 3), groups=116)`
  - `ffn.layers[4]`: `Conv2d(116, 32, kernel_size=(1, 1))`

#### 结论

- `torch-pruning` 可以正确处理 `MixFFN` 的通道依赖。
- 10% 剪枝后，第一阶段 FFN 的中间通道从 128 变为 116，符合预期。

---

### 实验 2：全局 FFN 剪枝

- **日期：** 2026-04-02
- **目的：** 对整个 SegFormer-B0 的 `MixFFN` 进行全局剪枝。
- **脚本：** `src/global_prune.py`

#### 配置

- **剪枝对象：** 所有 `MixFFN` 中的第一个卷积层
- **剪枝比例：** `0.2`（20%）
- **剪枝准则：** L1-norm

#### 结果

1. 成功加载模型。
2. 成功找到 8 个目标层。
3. 全局剪枝完成。
4. 前向推理通过。
5. 模型已保存为 `checkpoints/globally_pruned_ffn_20p.pth`。

#### 剪枝前后（部分）对比

- **Stage 1 FFN：**
  - 剪枝前：`Conv2d(32, 128, ...)`
  - 剪枝后：`Conv2d(32, 103, ...)`

- **Stage 2 FFN：**
  - 剪枝前：`Conv2d(64, 256, ...)`
  - 剪枝后：`Conv2d(64, 205, ...)`

- **Stage 3 FFN：**
  - 剪枝前：`Conv2d(160, 640, ...)`
  - 剪枝后：`Conv2d(160, 512, ...)`

- **Stage 4 FFN：**
  - 剪枝前：`Conv2d(256, 1024, ...)`
  - 剪枝后：`Conv2d(256, 820, ...)`

#### 结论

- 全局 20% 剪枝后，FFN 的通道数明显下降。
- 这说明当前剪枝策略是可行的，后续可以继续尝试更高比例或加入微调。

---

## 7. 后续计划

- 尝试只剪 `MixFFN` 与只剪 `MSA` 的效果对比。
- 对剪枝后的模型进行短轮次微调。
- 记录 mIoU、FLOPs、参数量和推理速度。
- 为后续 TensorRT / Jetson Orin NX 部署做准备。

---

## 8. 执行落地路线（细化到每个步骤）

> 目标：把“后续计划”变成可执行流水线，做到每一步都有输入、动作、产出、验收标准。

### 8.1 总体流程图（执行顺序）

1. 基线补全（mIoU / Params / FLOPs / Latency）
2. A 线：FFN-only（20% -> 30%）
3. B 线：MSA-only（10% -> 20%）
4. C 线：FFN+MSA（20% 组合）
5. 候选模型短微调（5k / 10k / 20k）
6. 统一评测与阶段门决策（Go / Conditional / No-Go）
7. 部署链路预演（Jetson Orin NX）

### 8.2 分步执行清单（Step-by-Step）

| Step | 输入 | 动作 | 输出 | 验收标准 | 失败回滚 |
|---|---|---|---|---|---|
| S0 基线冻结 | 基线配置+权重 | 固定 baseline 版本与评测口径 | baseline 记录项 | 指标可复现（误差可接受） | 回到基线重新评测 |
| S1 剪枝实验矩阵初始化 | A/B/C 方案 | 建立实验 ID 与命名规则 | 矩阵表（见 Project_Log） | 每次实验可唯一追踪 | 重建命名并对齐旧记录 |
| S2 A-20 生成 | baseline 权重 | 仅 FFN 20% 剪枝 | A-20 剪枝权重 | 前向通过、结构合法 | 降比例到 10% |
| S3 A-20 微调 | A-20 权重 | 短微调 10k iter | A-20 best ckpt | mIoU 有恢复趋势 | 增迭代到 20k 或降 lr |
| S4 A-30 生成+微调 | baseline/A-20 | FFN 30% 剪枝+微调 | A-30 best ckpt | 相对 A-20 速度有收益 | 回退至 A-20 作为候选 |
| S5 B-10 生成+微调 | baseline | MSA 10% 剪枝+微调 | B-10 best ckpt | mIoU 不出现灾难性下降 | 降到 5% 或暂缓 B 线 |
| S6 B-20 生成+微调 | baseline/B-10 | MSA 20% 剪枝+微调 | B-20 best ckpt | 精度/速度有可比优势 | 保留 B-10，放弃 B-20 |
| S7 C-20 组合剪枝 | baseline | FFN+MSA 组合剪枝 | C-20 剪枝权重 | 前向稳定无结构错误 | 拆分为两阶段剪枝 |
| S8 C-20 微调 | C-20 权重 | 短微调 15k~20k iter | C-20 best ckpt | mIoU 恢复到可用区间 | 提升迭代/调整学习率 |
| S9 统一评测 | A/B/C 候选 | 汇总 mIoU/FLOPs/Params/FPS | 对比总表 | 至少选出 1 个晋级模型 | 回到对应分支补实验 |
| S10 部署预演 | 晋级模型 | ONNX/TensorRT/端侧测速 | 部署报告 | Orin NX 接近或达到 30 FPS | 保留桌面端最优并继续优化 |

### 8.3 每个实验的“固定动作模板”

每个 ExpID（例如 A-20）都按以下顺序执行：

1. 记录实验元数据（日期、配置、权重来源、剪枝比例、随机种子）。
2. 执行剪枝并保存模型。
3. 立刻做前向检查（必须通过）。
4. 记录关键层通道变化（至少 2 个 stage）。
5. 执行短微调（5k/10k/20k 三档之一）。
6. 评估 mIoU（验证集）。
7. 统计 Params/FLOPs。
8. 测试延迟/FPS（写明 device、batch、warmup）。
9. 填写结论（Go / Conditional / No-Go）。
10. 更新实验矩阵总表。

### 8.4 阶段验收门槛（建议）

| 阶段 | 必过项 | 建议阈值 |
|---|---|---|
| 阶段 1（10%~20%） | 前向稳定 + mIoU 可恢复 + 速度提升 | mIoU 相对 baseline 下降不超过 2.0~3.0 pt |
| 阶段 2（30%~50%） | A/B/C 至少一条线可用 | 压缩率显著提升且精度可接受 |
| 部署前 | 端侧测速稳定 | Jetson Orin NX > 30 FPS（目标） |

### 8.5 指标计算口径（统一写法）

- 精度变化：$\Delta mIoU = mIoU_{pruned} - mIoU_{baseline}$
- 压缩率：$Compression = 1 - \frac{Params_{pruned}}{Params_{baseline}}$
- FLOPs 降幅：$FLOPs_{drop} = 1 - \frac{FLOPs_{pruned}}{FLOPs_{baseline}}$
- 速度提升：$Speedup = \frac{Latency_{baseline}}{Latency_{pruned}}$

### 8.6 周节奏建议（便于持续推进）

- 周一：完成 1~2 个剪枝模型生成（不做大规模微调）。
- 周二~周三：做短微调与验证。
- 周四：做 FLOPs/Params/Latency/FPS 汇总。
- 周五：阶段门决策 + 更新 `docs/Project_Log.md` 与本计划文档。

### 8.7 里程碑交付物

1. 实验矩阵总表（持续更新）
2. 最优候选模型权重（至少 1 个）
3. 对比可视化（难例/边缘类）
4. 部署预演报告（端侧速度）
5. 最终结论（推荐方案 + 后续优化方向）
