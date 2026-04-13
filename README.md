# SegFormer-B1 + Torch-Pruning (RailSem19)

This package refactors pruning into the same workflow style used in Torch-Pruning's transformer examples:

- `prune.py`: structural pruning only
- `finetune.py`: real dataset finetuning for the pruned model
- `latency.py`: latency / FPS measurement
- `utils_mmseg.py`: shared helpers
- `scripts/*.sh`: runnable command templates

## Suggested experiment order

1. **Baseline**
   - Train `SegFormer-B1` on RailSem19 first.
2. **Bottleneck pruning first**
   - Start with `--mode mlp_bottleneck`
   - Start with stages `2 3 4`
   - Start with ratios `0.10 / 0.15 / 0.20`
3. **Finetune**
   - Finetune each pruned model for `20~30` epoch-equivalent
4. **Latency**
   - Measure baseline and pruned models with batch size 1 on the target device
5. **Only then try Taylor**
   - Taylor requires real segmentation batches, not dummy random labels

## Why this design

For Transformer pruning, Torch-Pruning's official examples separate pruning, finetuning, and latency measurement.
For your SegFormer case, this is even more important because:

- segmentation needs real data recovery after pruning
- attention/head divisibility can break if pruning is too aggressive
- latency gains must be validated with the deployed forward path

## Recommended first experiments

### A. Main pruning line
- `mode=mlp_bottleneck`
- `importance=group_magnitude`
- `prune_stages=2 3 4`
- `ratios=0.10, 0.15, 0.20`

### B. After stable
- `importance=taylor`
- `taylor_batches=10`
- same bottleneck setting

### C. Do NOT start with
- aggressive global uniform pruning
- pruning decode head
- dummy finetune


官方 README 里对 ViT 给的经验是：
GroupTaylorImportance + BasePruner 是很好的选择，同时区分 uniform 和 bottleneck 两类剪枝。对你这个语义分割任务，我把第一版主线改成了更稳的 MLP bottleneck pruning，而不是一上来全局乱剪 attention 外部维度。原因很简单：你之前 B0 已经暴露出 attention 维度/head 数约束问题，而官方 examples 里的 bottleneck 思路本来就是“优先剪内部维度，不先改外部骨架”


第一组是主实验线
SegFormer-B1 baseline
B1 + mlp_bottleneck + group_magnitude + ratio 0.10
B1 + mlp_bottleneck + group_magnitude + ratio 0.15
B1 + mlp_bottleneck + group_magnitude + ratio 0.20
这里我默认先剪 stages 2/3/4，不从最前面的高分辨率 stage 开刀


第二组是恢复线：
对上面每个 pruned model 做 20~30 epoch 等价 finetune
学习率建议从 baseline 低一个量级，比如 3e-5

第三组是 Taylor 线：

等 group_magnitude 跑稳以后，再把 importance=taylor
taylor_batches=10
其它设置先不变(group_magnitude 整体重要性分数，通常是L2范数，剪枝的是一个group而不是一层)


第四组是 latency 线：

测 baseline B1
测 pruned B1
batch size 先固定 1
用 tp.utils.benchmark.measure_latency
如果版本不兼容，就退回 manual CUDA timer


主要剪枝mlp层，attention层先不动，避免维度/head数约束问题，其实就是升维度，降维度的层
运行的指令
bash scripts/prune_b1_mlp15.sh
bash scripts/finetune_b1_mlp15.sh
bash scripts/latency_b1_mlp15.sh


实验的调整：
#local、global、isomorphic
#group_magnitude、taylor
#加上蒸馏

1、attention / head维度约束。slef_attention 本质上会把每个token的通道维度c拆成h个头：每个head的维度是 d = C/h。如果你剪掉的通道数不能被头数整除，就会出问题。必须满足C%h=0。
2、round_to表示吧兼职之后的通道数和维度数向某个整数对齐，利于硬件加速
3、segformer里面 的mlp子层：x -> norm2 -> fc1 -> activation / dwconv -> fc2 -> residual add
fc1是升维，fc2是降维。剪mlp层的升维部分，等于剪掉了中间的隐藏维度，这个维度通常是外部骨架维度的4倍，剪掉它可以大幅减少计算量，但不会直接改动外部骨架的通道数和head数，所以更稳健一些。
4、store_true是argparse的一个参数选项，表示如果命令行里出现了这个参数，就把对应的变量设置为True，否则默认是False。比如你在命令行里加上--use_taylor，那么args.use_taylor就会是True；如果不加这个参数，args.use_taylor就是False。这种设计方便我们在代码里根据是否提供某个参数来控制程序的行为。
5、global_pruning会在所有层之间做统一的重要性排序
6、isomorphic=True：全局竞争时，再尽量保持结构更平衡一些，避免过度集中在某几层
7、taylor:用真实 train_loader 取 batch,在 真实 RailSem19 分割损失 下产生梯度,把这些梯度交给 Taylor importance 去估算 group 的重要性。同时看参数大小和它对当前任务 loss 的敏感性
8、B1 有 4 个 stage，每个 stage 有 2 个 Transformer block；每个 block 由 attention 子层和 MLP 子层组成，每个子层前面各有一个 LayerNorm。
9、剪枝local:默认不是“某一层”，而是stage2/3/4 里所有被 build_target_linear_names() 选中的 mlp.fc1 这些 root 层，各自做 local pruning。总共有6层。


毕设价值
B0 baseline vs B1 剪枝后模型 如果最后结果是 B0：60 mIoU，速度快 B1-pruned：61.5 mIoU，速度接近 B0，那 B1 剪枝就很有价值





