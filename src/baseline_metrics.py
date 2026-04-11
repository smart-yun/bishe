# -*- coding: utf-8 -*-
"""
One-file baseline evaluation for MMSegmentation models.

This script computes:
1) mIoU (full validation set)
2) Params / FLOPs
3) Latency / FPS

Example:
python src/baseline_metrics.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_40000it.py \
  --checkpoint runs/rs19/segformer_b0_512x512_40000it/best_mIoU_iter_40000.pth \
  --device cuda:0


#baseline
  python src/baseline_metrics.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
  --checkpoint runs/best_mIoU_iter_v1.pth \
  --output exports/80000it_baseline_metrics.json\
  --device cuda:0

#prune_test
    python src/baseline_metrics.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py \
  --checkpoint checkpoints/pruned_test_model_80000it_79000best.pth \
  --output exports/80000it_prune_test_metrics.json\
  --device cuda:0
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
from torch import nn

from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules


def prepare_pythonpath(project_root: Path) -> None:
    """Ensure custom modules (e.g. datasets.rs19_mmseg_dataset) are importable."""
    src_dir = str(project_root / 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def build_eval_cfg(config_path: str, checkpoint_path: str, work_dir: str) -> Config:
    """Load config and patch runtime-friendly settings."""
    cfg = Config.fromfile(config_path)
    cfg.launcher = 'none'
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint_path

    # Avoid tensorboard dependency in environments that only need evaluation.
    cfg.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer')

    return cfg


def to_unit(value: float, unit: str) -> float:
    """Convert raw counts to M or G."""
    if unit == 'M':
        return value / 1e6
    if unit == 'G':
        return value / 1e9
    return value


def _safe_torch_load(checkpoint_path: str) -> Any:
    """Load checkpoint object while being compatible with PyTorch versions."""
    try:
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Older PyTorch versions may not support `weights_only`.
        return torch.load(checkpoint_path, map_location='cpu')


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common distributed prefixes from state_dict keys."""
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith('module.') for k in keys):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
    """Extract a state_dict from different checkpoint object formats."""
    if isinstance(checkpoint_obj, dict):
        for key in ('state_dict', 'model', 'model_state_dict'):
            maybe = checkpoint_obj.get(key)
            if isinstance(maybe, dict):
                return _normalize_state_dict_keys(maybe)

        # Raw state_dict format: {param_name: tensor, ...}
        if checkpoint_obj and all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
            return _normalize_state_dict_keys(checkpoint_obj)

    if isinstance(checkpoint_obj, nn.Module):
        return _normalize_state_dict_keys(checkpoint_obj.state_dict())

    if hasattr(checkpoint_obj, 'state_dict') and callable(checkpoint_obj.state_dict):
        return _normalize_state_dict_keys(checkpoint_obj.state_dict())

    raise TypeError(
        f'Unsupported checkpoint object type: {type(checkpoint_obj)}. '
        'Expected dict / state_dict / nn.Module.'
    )


def robust_load_model_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    """Load checkpoint robustly for both normal and full-model serialized formats."""
    try:
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        return
    except TypeError:
        # Fallback for cases like torch.save(model), where mmengine expects dict-like obj.
        pass

    checkpoint_obj = _safe_torch_load(checkpoint_path)
    state_dict = _extract_state_dict(checkpoint_obj)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f'[WARN] Missing keys when loading checkpoint: {len(missing_keys)}')
    if unexpected_keys:
        print(f'[WARN] Unexpected keys when loading checkpoint: {len(unexpected_keys)}')


def eval_miou(cfg: Config) -> Dict[str, float]:
    """Run full validation and return metric dict."""
    cfg_local = copy.deepcopy(cfg)
    checkpoint_path = cfg_local.get('load_from', None)
    cfg_local.load_from = None

    runner = Runner.from_cfg(cfg_local)
    if checkpoint_path:
        robust_load_model_checkpoint(runner.model, checkpoint_path)

    metrics = runner.test()

    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except Exception:
            continue

    # NOTE: mmseg IoUMetric already reports mIoU in percentage (0~100).
    # Keep the raw value to avoid accidental x100 amplification.
    return out


def eval_flops_params(cfg: Config, input_hw: Tuple[int, int], device: str) -> Dict[str, float]:
    """Compute FLOPs and params using mmengine complexity API."""
    cfg_local = copy.deepcopy(cfg)
    init_default_scope(cfg_local.get('default_scope', 'mmseg'))

    model = MODELS.build(cfg_local.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None

    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    if use_cuda:
        model = model.to(device)

    model = revert_sync_batchnorm(model)
    model.eval()

    h, w = input_hw
    data_info = {'ori_shape': (h, w), 'pad_shape': (h, w)}
    data_batch = {
        'inputs': [torch.rand((3, h, w))],
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


def build_benchmark_model_and_loader(cfg: Config, checkpoint_path: str, device: str):
    """Create model + dataloader for latency benchmark."""
    cfg_local = copy.deepcopy(cfg)
    cfg_local.model.pretrained = None
    cfg_local.model.train_cfg = None
    cfg_local.test_dataloader.batch_size = 1

    data_loader = Runner.build_dataloader(cfg_local.test_dataloader)

    model = MODELS.build(cfg_local.model)
    robust_load_model_checkpoint(model, checkpoint_path)

    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    if use_cuda:
        model = model.to(device)

    model = revert_sync_batchnorm(model)
    model.eval()

    return model, data_loader


def eval_latency(
    cfg: Config,
    checkpoint_path: str,
    device: str,
    warmup: int,
    total_iters: int,
    repeat_times: int,
) -> Dict[str, Any]:
    """Benchmark latency/FPS using dataloader-driven inference."""
    if total_iters <= warmup:
        raise ValueError(f'total_iters({total_iters}) must be > warmup({warmup})')

    torch.backends.cudnn.benchmark = False

    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    all_fps: List[float] = []
    all_latency_ms: List[float] = []

    for _ in range(repeat_times):
        model, data_loader = build_benchmark_model_and_loader(cfg, checkpoint_path, device)
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
    parser = argparse.ArgumentParser(description='Baseline metrics script for MMSeg')
    parser.add_argument('--config', required=True, help='Path to mmseg config .py')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pth')
    parser.add_argument('--work-dir', default='runs/rs19/baseline_eval_script', help='Work dir for eval artifacts')
    parser.add_argument('--device', default='cuda:0', help='Device, e.g. cuda:0 or cpu')
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512], help='Input shape H W for FLOPs')

    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations for latency benchmark')
    parser.add_argument('--iters', type=int, default=200, help='Total iterations (including warmup)')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat times for latency benchmark')

    parser.add_argument('--skip-miou', action='store_true', help='Skip mIoU evaluation')
    parser.add_argument('--skip-flops', action='store_true', help='Skip FLOPs/Params')
    parser.add_argument('--skip-latency', action='store_true', help='Skip Latency/FPS benchmark')
    parser.add_argument('--min-miou', type=float, default=None, help='Minimum acceptable mIoU (%) for baseline sanity check')
    parser.add_argument('--strict-baseline-check', action='store_true', help='Exit with non-zero code if baseline mIoU < --min-miou')

    parser.add_argument('--output-json', '--output', dest='output_json', default='exports/baseline_metrics.json', help='Output JSON path')
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

    cfg = build_eval_cfg(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        work_dir=args.work_dir,
    )

    result: Dict[str, Any] = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': device,
        'shape': args.shape,
    }

    print('\n[1/3] Running mIoU evaluation...')
    if not args.skip_miou:
        miou_metrics = eval_miou(cfg)
        result['miou_metrics'] = miou_metrics
        if 'mIoU' in miou_metrics:
            print(f"  -> mIoU: {miou_metrics['mIoU']:.2f}%")
            if args.min_miou is not None and miou_metrics['mIoU'] < float(args.min_miou):
                msg = (
                    f"[BASELINE-CHECK] mIoU {miou_metrics['mIoU']:.2f}% is below minimum "
                    f"{float(args.min_miou):.2f}%"
                )
                if args.strict_baseline_check:
                    raise SystemExit(msg)
                print(f'[WARN] {msg}')
        else:
            print(f'  -> metrics: {miou_metrics}')
    else:
        print('  -> skipped')

    print('\n[2/3] Computing FLOPs / Params...')
    if not args.skip_flops:
        complexity = eval_flops_params(cfg, (args.shape[0], args.shape[1]), device)
        result['complexity'] = complexity
        print(f"  -> Params: {complexity['params_m']:.2f} M")
        print(f"  -> FLOPs:  {complexity['flops_g']:.2f} G")
    else:
        print('  -> skipped')

    print('\n[3/3] Running Latency / FPS benchmark...')
    if not args.skip_latency:
        latency = eval_latency(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            device=device,
            warmup=args.warmup,
            total_iters=args.iters,
            repeat_times=args.repeat,
        )
        result['latency'] = latency
        print(f"  -> Latency: {latency['latency_ms_mean']:.2f} ms")
        print(f"  -> FPS:     {latency['fps_mean']:.2f}")
    else:
        print('  -> skipped')

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = project_root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n================ Baseline Summary ================')
    if 'miou_metrics' in result and 'mIoU' in result['miou_metrics']:
        print(f"mIoU:     {result['miou_metrics']['mIoU']:.2f} %")
    if 'complexity' in result:
        print(f"Params:   {result['complexity']['params_m']:.2f} M")
        print(f"FLOPs:    {result['complexity']['flops_g']:.2f} G")
    if 'latency' in result:
        print(f"Latency:  {result['latency']['latency_ms_mean']:.2f} ms")
        print(f"FPS:      {result['latency']['fps_mean']:.2f}")
    print(f'JSON saved to: {output_json}')
    print('==================================================\n')


if __name__ == '__main__':
    main()
