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
现在等待 b1_uni_p50_global_iso 的微调效果 运行100个epoch   目前 效果看起来不错

4、等实验结果出来试一下蒸馏恢复精度

今晚、明晚改一下论文 包括蒸馏部分、微调的结果

5、在 50% 剪枝 student 上直接采用 logit distillation，且蒸馏损失权重设为 1.0 时，训练过程中 KD loss 的量级显著高于分割监督损失，导致优化目标被蒸馏项主导，验证集 mIoU 长期维持在 1 左右，未表现出有效恢复趋势，说明当前蒸馏损失设计或权重设置不合理。
修改成0.05 效果稳定多了 现在看看0.1是否稳定

5、微调效果还是很好的



目前完成的实验：

segformer_b0_mlp_gm_10_ft
segformer_b0_mlp_gm_50_ft

<!-- segformer_b1_mlp_gm_10
segformer_b1_mlp_gm_30
segformer_b1_mlp_gm_50
segformer_b1_mlp_gm_70
segformer_b1_mlp_gm_90 -->

<!-- segformer_b1_mlp_gm_30_gi
segformer_b1_mlp_gm_90_gi -->

segformer_b1_mlp_gm_10_ft
segformer_b1_mlp_gm_30_ft
segformer_b1_mlp_gm_90_ft

<!-- segformer_b1_uni_gm_30_gi
segformer_b1_uni_gm_50_gi
segformer_b1_uni_gm_70_gi -->

segformer_b1_uni_gm_30_gi_ft
segformer_b1_uni_gm_90_gi_ft

b1_uni_p50_global_iso 微调100epoch
b1_uni_p70_global_iso 微调60epoch

b1_p50_ft50_kd_logit  微调加蒸馏50epoch

6、aAcc和mAcc:aAcc是所有像素分类正确的比例，mAcc是每个类别分类正确的平均比例。aAcc受大类影响较大，mAcc更能反映小类的表现。

7、记录数据、作图、结论
晚上改论文

8、很多语义分割模型内部 decoder 会先输出低分辨率 logits，然后在评估或可视化阶段再上采样到原图大小
