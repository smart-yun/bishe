:1、mode 与 global关系：

`mode` 决定的是：**哪些层能成为剪枝候选根层**；

而 `--global-pruning` 管的是：
对这些已经选出来的候选层，**重要性排序是分层/分组各剪各的，还是放到一起全局统一排序再剪**。

2、local 剪枝比例反而比 global 更大：
这 18 个 root 的规模差异很大。后面 stage 的 FFN 通道宽得多，前面 stage 小得多。
在这种情况下，global 很容易把预算分配到“importance 更低但参数没那么大的层”上，而不是平均落到所有大 FFN 上。结果就是：
同样写 0.3，global 的真实总参数下降反而更少。剪枝30%
local:11.8M
global:12.4M
global+iso:12.2M

重点运行剪枝0.9+global+iso 13.6M->8.18M  mIoU 等待运行60epoch

3、b1_uni_p70_global_iso 剪枝效果比较好，但是精度上不去  60个epoch微调之后与baseline还是差距10个百分点
现在等待 b1_uni_p50_global_iso 的微调效果 运行100个epoch

4、等实验结果出来试一下蒸馏恢复精度

今晚、明晚改一下论文



