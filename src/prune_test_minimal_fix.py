# -*- coding: utf-8 -*-
"""
最小改动版：在结构化剪枝后，直接基于“剪枝后的内存模型”输出真实评估结果。
输出 JSON 包含：mIoU / Params / FLOPs（可选 latency）。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch_pruning as tp

from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner

from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules


def to_percent(value: float) -> float:
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def to_unit(value: float, unit: str) -> float:
    if unit == 'M':
        return value / 1e6
    if unit == 'G':
        return value / 1e9
    return value


def get_model_from_config(config_path: str, checkpoint_path: str):
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        incompatible = model.load_state_dict(state_dict, strict=False)
        if hasattr(incompatible, 'missing_keys') and incompatible.missing_keys:
            print(f"[load] missing_keys: {len(incompatible.missing_keys)}")
        if hasattr(incompatible, 'unexpected_keys') and incompatible.unexpected_keys:
            print(f"[load] unexpected_keys: {len(incompatible.unexpected_keys)}")

    model = revert_sync_batchnorm(model)
    model.eval()
    return model, cfg


def eval_miou_with_pruned_model(cfg: Config, model: torch.nn.Module, work_dir: str) -> Dict[str, float]:
    """关键点：复用 cfg 的 dataloader/evaluator，但模型使用“剪枝后内存模型”。"""
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval.launcher = 'none'
    cfg_eval.work_dir = work_dir
    cfg_eval.load_from = None
    cfg_eval.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer')

    runner = Runner.from_cfg(cfg_eval)
    # 用剪枝后模型覆盖 runner 内部模型，避免“重建原结构 + 加载不匹配权重”
    runner.model = model

    metrics = runner.test()
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    if 'mIoU' in out:
        out['mIoU'] = to_percent(out['mIoU'])
    return out


def eval_flops_params_from_model(model: torch.nn.Module, input_hw: Tuple[int, int], device: str) -> Dict[str, float]:
    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    if use_cuda:
        model = model.to(device)

    model = revert_sync_batchnorm(model)
    model.eval()

    h, w = input_hw
    data_info = {'ori_shape': (h, w), 'pad_shape': (h, w)}
    data_batch = {
        'inputs': [torch.rand((3, h, w), device=device if use_cuda else 'cpu')],
        'data_samples': [SegDataSample(metainfo=data_info)]
    }
    data = model.data_preprocessor(data_batch)

    outputs = get_model_complexity_info(
        model,
        input_shape=None,
        inputs=data['inputs'],
        show_table=False,
        show_arch=False,
    )

    flops = float(outputs['flops'])
    params = float(outputs['params'])
    return {
        'flops': flops,
        'params': params,
        'flops_g': to_unit(flops, 'G'),
        'params_m': to_unit(params, 'M'),
    }


def eval_latency_from_model(
    cfg: Config,
    model: torch.nn.Module,
    device: str,
    warmup: int,
    total_iters: int,
    repeat_times: int,
) -> Dict[str, Any]:
    if total_iters <= warmup:
        raise ValueError(f'total_iters({total_iters}) must be > warmup({warmup})')

    cfg_local = copy.deepcopy(cfg)
    cfg_local.test_dataloader.batch_size = 1
    data_loader = Runner.build_dataloader(cfg_local.test_dataloader)

    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    if use_cuda:
        model = model.to(device)
    model.eval()

    all_fps: List[float] = []
    all_latency_ms: List[float] = []

    for _ in range(repeat_times):
        data_iter = cycle(data_loader)
        measured: List[float] = []

        for i in range(total_iters):
            data = next(data_iter)
            data = model.data_preprocessor(data, True)
            inputs = data['inputs']
            data_samples = data['data_samples']

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(inputs, data_samples, mode='predict')
            if use_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            if i >= warmup:
                measured.append(dt)

        mean_s = statistics.mean(measured)
        all_latency_ms.append(mean_s * 1000.0)
        all_fps.append(1.0 / mean_s)

    return {
        'latency_ms_mean': statistics.mean(all_latency_ms),
        'latency_ms_std': statistics.pstdev(all_latency_ms) if len(all_latency_ms) > 1 else 0.0,
        'fps_mean': statistics.mean(all_fps),
        'fps_std': statistics.pstdev(all_fps) if len(all_fps) > 1 else 0.0,
        'repeat_times': repeat_times,
        'warmup': warmup,
        'total_iters': total_iters,
        'timed_iters': total_iters - warmup,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Minimal prune + true pruned eval script')
    parser.add_argument('--config', default='configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py')
    parser.add_argument('--checkpoint', default='runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth')
    parser.add_argument('--target-layer', default='backbone.layers.0.1.0.ffn.layers.0')
    parser.add_argument('--pruning-ratio', type=float, default=0.1)
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--work-dir', default='runs/rs19/prune_eval_script')
    parser.add_argument('--pruned-checkpoint', default='checkpoints/pruned_test_model_80000it_79000best.pth')
    parser.add_argument('--output-json', default='exports/pruned_eval.json')

    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--skip-miou', action='store_true')
    parser.add_argument('--skip-flops', action='store_true')
    parser.add_argument('--skip-latency', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_all_modules(init_default_scope=True)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] CUDA not available, fallback to CPU.')
        device = 'cpu'

    os.makedirs(args.work_dir, exist_ok=True)

    print(f"[load] config={args.config}")
    model, cfg = get_model_from_config(args.config, args.checkpoint)

    # 1) 构建依赖图并剪枝
    print('[prune] build dependency graph...')
    dg = tp.DependencyGraph()
    dg.build_dependency(model, example_inputs=torch.randn(1, 3, args.shape[0], args.shape[1]))

    target_layer = model.get_submodule(args.target_layer)
    w = target_layer.weight.detach()
    n_channels = w.size(0)
    n_prune = max(1, int(n_channels * args.pruning_ratio))
    l1 = w.abs().flatten(1).sum(1)
    idxs = torch.argsort(l1)[:n_prune].tolist()

    group = dg.get_pruning_group(target_layer, tp.prune_conv_out_channels, idxs=idxs)
    if hasattr(group, 'is_pruned') and group.is_pruned:
        print('[prune] warning: group already pruned.')
    else:
        group.prune()

    # 2) 快速前向验证
    print('[check] forward smoke test...')
    with torch.no_grad():
        x = torch.randn(1, 3, args.shape[0], args.shape[1])
        _ = model(x)

    # 3) 保存剪枝模型权重（state_dict）
    os.makedirs(os.path.dirname(args.pruned_checkpoint), exist_ok=True)
    torch.save(model.state_dict(), args.pruned_checkpoint)
    print(f"[save] pruned checkpoint: {args.pruned_checkpoint}")

    # 4) 直接对剪枝后模型评估（真实链路）
    result: Dict[str, Any] = {
        'config': args.config,
        'baseline_checkpoint': args.checkpoint,
        'pruned_checkpoint': args.pruned_checkpoint,
        'target_layer': args.target_layer,
        'pruning_ratio': args.pruning_ratio,
        'device': device,
        'shape': args.shape,
    }

    if not args.skip_miou:
        print('[eval] mIoU on pruned in-memory model...')
        result['miou_metrics'] = eval_miou_with_pruned_model(cfg, model, args.work_dir)

    if not args.skip_flops:
        print('[eval] Params/FLOPs on pruned in-memory model...')
        result['complexity'] = eval_flops_params_from_model(model, (args.shape[0], args.shape[1]), device)

    if not args.skip_latency:
        print('[eval] Latency/FPS on pruned in-memory model...')
        result['latency'] = eval_latency_from_model(
            cfg=cfg,
            model=model,
            device=device,
            warmup=args.warmup,
            total_iters=args.iters,
            repeat_times=args.repeat,
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[done] JSON saved to: {output_json}")


if __name__ == '__main__':
    main()
