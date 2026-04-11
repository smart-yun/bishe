# -*- coding: utf-8 -*-
"""Evaluate a structurally-pruned full model object and export 4 metrics.

Outputs:
- miou_metrics.mIoU
- complexity.{params, flops, params_m, flops_g}
- latency.{latency_ms_mean, fps_mean, ...}
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules


def prepare_pythonpath(project_root: Path) -> None:
    src_dir = str(project_root / 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def build_eval_cfg(config_path: str, work_dir: str) -> Config:
    cfg = Config.fromfile(config_path)
    cfg.launcher = 'none'
    cfg.work_dir = work_dir
    cfg.load_from = None
    cfg.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer')
    return cfg


def to_unit(value: float, unit: str) -> float:
    if unit == 'M':
        return value / 1e6
    if unit == 'G':
        return value / 1e9
    return value


def _nearest_valid_num_heads(embed_dim: int, old_heads: int) -> int:
    """Pick a valid num_heads close to old_heads, preferring smaller change."""
    if embed_dim <= 0:
        return 1
    if old_heads > 0 and embed_dim % old_heads == 0:
        return old_heads

    # Prefer divisors <= old_heads first (keeps behavior closer in practice).
    for h in range(max(old_heads - 1, 1), 0, -1):
        if embed_dim % h == 0:
            return h

    # Fallback: search upward.
    for h in range(max(old_heads + 1, 2), embed_dim + 1):
        if embed_dim % h == 0:
            return h
    return 1


def sanitize_multihead_attention(model: torch.nn.Module) -> None:
    """Fix invalid MultiheadAttention settings after structural pruning."""
    fixed = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            embed_dim = int(module.embed_dim)
            num_heads = int(module.num_heads)
            if embed_dim % num_heads == 0:
                continue

            new_heads = _nearest_valid_num_heads(embed_dim, num_heads)
            if new_heads != num_heads:
                module.num_heads = new_heads
                module.head_dim = embed_dim // new_heads
                fixed += 1
                print(
                    f'[WARN] Fixed invalid MultiheadAttention at {name}: '
                    f'embed_dim={embed_dim}, num_heads={num_heads} -> {new_heads}'
                )

    if fixed > 0:
        print(f'[INFO] Fixed {fixed} invalid MultiheadAttention module(s).')


def load_pruned_model(model_path: str, device: str):
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location='cpu')

    sanitize_multihead_attention(model)

    model = revert_sync_batchnorm(model)
    if device.startswith('cuda') and torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model.to('cpu')
    model.eval()
    return model


def eval_miou(cfg: Config, model: torch.nn.Module) -> Dict[str, float]:
    runner = Runner.from_cfg(copy.deepcopy(cfg))
    runner.model = model
    metrics = runner.test()

    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except Exception:
            continue

    # NOTE: mmseg IoUMetric already reports percentage values (0~100).
    # Do not multiply again, otherwise 0.74% becomes 74.00% incorrectly.
    return out


def eval_flops_params(model: torch.nn.Module, input_hw: Tuple[int, int], device: str) -> Dict[str, float]:
    init_default_scope('mmseg')

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


def eval_latency(
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
    parser = argparse.ArgumentParser(description='Evaluate full pruned model object (4 metrics)')
    parser.add_argument('--config', required=True, help='Path to mmseg config .py')
    parser.add_argument('--model-path', '--checkpoint', dest='model_path', required=True,
                        help='Path to full pruned model .pth (torch.save(model))')
    parser.add_argument('--work-dir', default='runs/rs19/pruned_eval_script', help='Work dir for eval artifacts')
    parser.add_argument('--device', default='cuda:0', help='Device, e.g. cuda:0 or cpu')
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512], help='Input shape H W for FLOPs')

    parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations for latency benchmark')
    parser.add_argument('--iters', type=int, default=200, help='Total iterations (including warmup)')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat times for latency benchmark')

    parser.add_argument('--output-json', '--output', dest='output_json',
                        default='exports/pruned_full_metrics.json', help='Output JSON path')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    prepare_pythonpath(project_root)

    register_all_modules(init_default_scope=True)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] CUDA is not available, fallback to CPU.')
        device = 'cpu'

    os.makedirs(args.work_dir, exist_ok=True)

    cfg = build_eval_cfg(config_path=args.config, work_dir=args.work_dir)
    model = load_pruned_model(args.model_path, device)

    result: Dict[str, Any] = {
        'config': args.config,
        'model_path': args.model_path,
        'device': device,
        'shape': args.shape,
    }

    print('\n[1/3] Running mIoU evaluation...')
    miou_metrics = eval_miou(cfg, model)
    result['miou_metrics'] = miou_metrics
    print(f"  -> mIoU: {miou_metrics.get('mIoU', float('nan')):.2f}%")

    print('\n[2/3] Computing FLOPs / Params...')
    complexity = eval_flops_params(model, (args.shape[0], args.shape[1]), device)
    result['complexity'] = complexity
    print(f"  -> Params: {complexity['params_m']:.2f} M")
    print(f"  -> FLOPs:  {complexity['flops_g']:.2f} G")

    print('\n[3/3] Running Latency / FPS benchmark...')
    latency = eval_latency(
        cfg=cfg,
        model=model,
        device=device,
        warmup=args.warmup,
        total_iters=args.iters,
        repeat_times=args.repeat,
    )
    result['latency'] = latency
    print(f"  -> Latency: {latency['latency_ms_mean']:.2f} ms")
    print(f"  -> FPS:     {latency['fps_mean']:.2f}")

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = project_root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n================ Pruned Summary ================')
    if 'miou_metrics' in result and 'mIoU' in result['miou_metrics']:
        print(f"mIoU:     {result['miou_metrics']['mIoU']:.2f} %")
    print(f"Params:   {result['complexity']['params_m']:.2f} M")
    print(f"FLOPs:    {result['complexity']['flops_g']:.2f} G")
    print(f"Latency:  {result['latency']['latency_ms_mean']:.2f} ms")
    print(f"FPS:      {result['latency']['fps_mean']:.2f}")
    print(f'JSON saved to: {output_json}')
    print('================================================\n')


if __name__ == '__main__':
    main()