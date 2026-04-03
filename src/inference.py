# -*- coding: utf-8 -*-
import argparse
import os
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
import glob
import os.path as osp
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="MMSegmentation Inference Script")
    parser.add_argument(
        '--img', 
        default='data/railsem19/validation/images',
        help='Image file or folder path'
    )
    parser.add_argument(
        '--config', 
        default='configs/railsem19/segformer_b0_rs19_512x512_40000it.py',
        help='Config file'
    )
    parser.add_argument(
        '--checkpoint', 
        default='runs/rs19/segformer_b0_512x512_40000it/best_mIoU_iter_40000.pth',
        help='Checkpoint file'
    )
    parser.add_argument(
        '--out-dir', 
        default='results/vis_check',
        help='Directory to save inference results'
    )
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        help='Device used for inference'
    )
    parser.add_argument(
        '--palette',
        default='railsem19',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    
    args = parser.parse_args()

    # 检查权重文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 权重文件未找到 at '{args.checkpoint}'")
        print("请确保路径正确，或者先完成模型训练。")
        return

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 从配置文件和权重文件初始化分割器
    print("正在加载模型...")
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, using palette "railsem19" by default')
        model.CLASSES = get_palette(args.palette)

    # 获取图片列表
    if os.path.isdir(args.img):
        img_list = glob.glob(os.path.join(args.img, '*.jpg')) + \
                   glob.glob(os.path.join(args.img, '*.png'))
    else:
        img_list = [args.img]

    if not img_list:
        print(f"在 '{args.img}' 路径下未找到任何图片。")
        return
        
    print(f"找到 {len(img_list)} 张图片，开始推理...")
    # 对每张图片进行推理
    for img_path in tqdm(img_list, desc="处理图片"):
        result = inference_segmentor(model, img_path)
        
        # 保存可视化结果
        out_path = osp.join(args.out_dir, osp.basename(img_path))
        model.show_result(
            img_path,
            result,
            palette=get_palette(args.palette),
            show=False,
            out_file=out_path,
            opacity=args.opacity)
    
    print(f"推理完成！结果已保存到: {args.out_dir}")

if __name__ == '__main__':
    main()
