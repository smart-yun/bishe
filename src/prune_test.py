# -*- coding: utf-8 -*-
import torch
import torch_pruning as tp
import os
from mmengine.config import Config  #方便地加载在 MMSegmentation 框架中定义的模型
from mmseg.models import build_segmentor
from mmengine.model import revert_sync_batchnorm

def get_model_from_config(config_path, checkpoint_path):
    #""" 从配置文件和权重文件加载模型 """
    cfg = Config.fromfile(config_path) #config_path中定义了模型的完整模型配置,backbone, neck, head等
    
    # 从配置文件创建一个模型实例
    model = build_segmentor(cfg.model)
    
    # 加载权重
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        
    # 切换到评估模式
    model.eval()
    # 转换 SyncBN 为 BN，SyncBN 用于多 GPU 训练，而剪枝和测试通常在单 GPU 或 CPU 上进行。
    model = revert_sync_batchnorm(model)
    return model, cfg

def main():
    # --- 1. 配置 ---
    config_path = 'configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py'
    # 请将 'path/to/your/best_baseline_checkpoint.pth' 替换为你的基线模型权重路径
    checkpoint_path = 'runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth' # 示例路径，请务必修改
    
    print(f"from  config  '{config_path}' load model...")
    model, cfg = get_model_from_config(config_path, checkpoint_path)
    
    # --- 2. 打印原始模型结构 ---#可以通过 print(model) 查看模型所有层的名称
    print("\n--- original model ---")
    print(model)
    
    # --- 3. 结构化剪枝 ---
    DG = tp.DependencyGraph()#DependencyGraph (依赖图) 是一个用来分析模型中所有层、张量之间相互依赖关系的对象。
    #通过一次“模拟”的前向传播来追踪数据在网络中的流动路径，从而构建出这个图。
    #example_inputs 就是这次模拟传播的输入数据，其尺寸需要和模型实际输入一致。
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 512, 512))
    
    # 选择一个目标层进行试剪，例如第一个encoder block的FFN层
    # 你可以通过 print(model) 找到你想要剪枝的层的确切名称
    #SegFormer 主干网络中第一个 Block 里的一个卷积层
    target_layer_name = 'backbone.layers.0.1.0.ffn.layers.0' # Conv2d层，便于通道剪枝

    
    
    try:
        target_layer = model.get_submodule(target_layer_name)
        print(f"\nfind target layer: {target_layer_name}")
    except AttributeError:
        print(f"Error: can not find '{target_layer_name}' layer. Please check model and refresh the layer name.")
        print("usable top model:")
        for name, module in model.named_children():
            print(f"- {name}")
        return

    # 定义剪枝策略：剪掉 L1-norm 最小的 10% 的输出通道（兼容 torch_pruning 1.6.x）
    # 基本思想是,一个卷积核(滤波器)的权重绝对值之和(即L1范数)越小,代
    # 表它对输出特征图的贡献越小，因此可以被认为是“不重要”的，可以被优先剪掉。
    pruning_ratio = 0.1#即移除 10% 的最不重要通道。
    weight = target_layer.weight.detach()#获取目标层卷积层的权重张量
    n_channels = weight.size(0)#获取通道数
    n_prune = max(1, int(n_channels * pruning_ratio)) #获取剪枝数
    l1 = weight.abs().flatten(1).sum(1) #求l1范数
    # .flatten(1) : 将每个滤波器的所有权重“压平”成一个一维向量。例如，形状从 
    # (输出通道数, 输入通道数, 3, 3) 变为 (输出通道数, 输入通道数 * 3 * 3) 。
    # 这样一来，每一行就代表一个完整滤波器的所有权重值。
    #
    # sum(1)表示沿着第二维(dim=1)进行求和
    idxs = torch.argsort(l1)[:n_prune].tolist()

    # Conv2d out-channel 剪枝
    group = DG.get_pruning_group(
        target_layer,              # 1. 连锁反应的起点
        tp.prune_conv_out_channels, # 2. 在起点执行的操作类型，它会沿着依赖图 向前和向后 追踪所有与这些被移除通道有依赖关系的层。
        idxs=idxs                  # 3. 具体要操作的通道索引
    )

    # --- 4. 执行剪枝并验证 ---
    print(f"\ndo '{target_layer_name}' about {pruning_ratio*100}% cut...")
    if hasattr(group, 'is_pruned') and group.is_pruned:
        print("Warning: this part has been cut.")
    else:
        group.prune() #执行剪枝

    # --- 5. 打印剪枝后模型结构 ---
    print("\n--- the model infra after cut ---")
    print(model)
    
    # --- 6. 验证前向传播 ---
    try:
        print("\ntest the forward after cut...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 512, 512)
            output = model(test_input)
        print(f"Forward successed! output shape: {output[0].shape}")
    except Exception as e:
        print(f"Forward failed!: {e}")

    # --- 7. 保存试剪后的模型 ---
    pruned_checkpoint_path = 'checkpoints/pruned_test_model_80000it_79000best.pth'
    os.makedirs(os.path.dirname(pruned_checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), pruned_checkpoint_path)
    print(f"\nthe model been cut is saved to: {pruned_checkpoint_path}")

if __name__ == '__main__':
    main()
