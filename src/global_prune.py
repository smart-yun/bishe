# -*- coding: utf-8 -*-
"""
Global pruning + optional short fine-tuning + evaluation for SegFormer (MixFFN layer0 channels).

Features:
1) True global ranking pruning across all candidate FFN layer0 channels.
2) Evaluate pruned in-memory model directly (mIoU / Params / FLOPs / Latency).
3) Optional short fine-tuning on pruned model.
4) Export JSON for report-ready comparison (post-prune vs post-finetune).

usage: global_prune.py [-h] --config CONFIG --checkpoint CHECKPOINT
                       [--pruning-ratio PRUNING_RATIO]
                       [--target-stages [TARGET_STAGES ...]]
                       [--max-target-layers MAX_TARGET_LAYERS]
                       [--shape SHAPE SHAPE] [--device DEVICE]
                       [--work-dir WORK_DIR]
                       [--pruned-checkpoint PRUNED_CHECKPOINT]
                       [--output-json OUTPUT_JSON] [--warmup WARMUP]
                       [--iters ITERS] [--repeat REPEAT] [--skip-miou]
                       [--skip-flops] [--skip-latency] [--enable-finetune]
                       [--finetune-iters FINETUNE_ITERS]
                       [--finetune-lr FINETUNE_LR]
                       [--finetune-weight-decay FINETUNE_WEIGHT_DECAY]
                       [--finetune-eval-interval FINETUNE_EVAL_INTERVAL]
                       [--finetune-log-interval FINETUNE_LOG_INTERVAL]
                       [--finetune-save-best FINETUNE_SAVE_BEST]
                       [--finetune-save-last FINETUNE_SAVE_LAST]

Global channel pruning + optional fine-tune + evaluation for SegFormer FFN

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to mmseg config .py
  --checkpoint CHECKPOINT
                        Path to baseline checkpoint .pth
  --pruning-ratio PRUNING_RATIO
  --target-stages [TARGET_STAGES ...]
  --max-target-layers MAX_TARGET_LAYERS
  --shape SHAPE SHAPE
  --device DEVICE
  --work-dir WORK_DIR
  --pruned-checkpoint PRUNED_CHECKPOINT
  --output-json OUTPUT_JSON
  --warmup WARMUP
  --iters ITERS
  --repeat REPEAT
  --skip-miou
  --skip-flops
  --skip-latency
  --enable-finetune
  --finetune-iters FINETUNE_ITERS
  --finetune-lr FINETUNE_LR
  --finetune-weight-decay FINETUNE_WEIGHT_DECAY
  --finetune-eval-interval FINETUNE_EVAL_INTERVAL
  --finetune-log-interval FINETUNE_LOG_INTERVAL
  --finetune-save-best FINETUNE_SAVE_BEST
  --finetune-save-last FINETUNE_SAVE_LAST
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch_pruning as tp

from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner

from mmseg.registry import METRICS
from mmseg.models import build_segmentor
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules

try:
    from mmseg.models.backbones.mit import MixFFN  # mmseg >= 1.x
except ImportError:
    from mmseg.models.backbones.mix_transformer import MixFFN  # mmseg < 1.x


def prepare_pythonpath(project_root: Path) -> None:
    src_dir = str(project_root / 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


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


def sum_loss_values(loss_dict: Dict[str, Any]) -> torch.Tensor:
    total = None
    for v in loss_dict.values():
        if torch.is_tensor(v):
            cur = v
        elif isinstance(v, (list, tuple)):
            tensors = [x for x in v if torch.is_tensor(x)]
            if not tensors:
                continue
            cur = sum(tensors)
        else:
            continue
        total = cur if total is None else (total + cur)

    if total is None:
        raise RuntimeError(f'No tensor losses found in loss dict keys={list(loss_dict.keys())}')
    return total


def normalize_batched_inputs(inputs: Any) -> torch.Tensor:
    """Normalize mmseg preprocessed inputs into a batched tensor [N,C,H,W]."""
    if torch.is_tensor(inputs):
        return inputs
    if isinstance(inputs, (list, tuple)):
        tensors = [x for x in inputs if torch.is_tensor(x)]
        if not tensors:
            raise TypeError(f'inputs list/tuple contains no tensors, got type={type(inputs)}')
        return torch.stack(tensors, dim=0)
    raise TypeError(f'Unsupported inputs type for model forward: {type(inputs)}')


def get_model_from_config(config_path: str, checkpoint_path: str):
    cfg = Config.fromfile(config_path)
    cfg.launcher = 'none'

    model = build_segmentor(cfg.model)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if hasattr(incompatible, 'missing_keys') and incompatible.missing_keys:
        print(f"[load] missing_keys: {len(incompatible.missing_keys)}")
    if hasattr(incompatible, 'unexpected_keys') and incompatible.unexpected_keys:
        print(f"[load] unexpected_keys: {len(incompatible.unexpected_keys)}")

    model = revert_sync_batchnorm(model)
    model.eval()
    return model, cfg


def _parse_stage_from_target_name(target_name: str) -> int | None:
    # e.g. backbone.layers.2.1.0.ffn.layers.0  -> stage=3
    m = re.search(r'backbone\.layers\.(\d+)\.', target_name)
    if m is None:
        return None
    return int(m.group(1)) + 1


def collect_ffn_first_convs(
    model: torch.nn.Module,
    target_stages: Sequence[int] | None = None,
    max_target_layers: int = 0,
) -> List[Tuple[str, torch.nn.Module]]:
    allowed = set(target_stages) if target_stages else None

    targets: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, MixFFN):
            continue

        target_name = f'{name}.layers.0'
        stage_id = _parse_stage_from_target_name(target_name)
        if allowed is not None and stage_id not in allowed:
            continue

        targets.append((target_name, module.layers[0]))

    if max_target_layers > 0:
        targets = targets[:max_target_layers]
    return targets


def global_prune_ffn(
    model: torch.nn.Module,
    shape: Tuple[int, int],
    pruning_ratio: float,
    target_stages: List[int] | None = None,
    max_target_layers: int = 0,
) -> Dict[str, Any]:
    if not (0.0 < pruning_ratio < 1.0):
        raise ValueError(f'pruning_ratio must be in (0,1), got {pruning_ratio}')

    targets = collect_ffn_first_convs(model, target_stages=target_stages, max_target_layers=max_target_layers)
    if not targets:
        raise RuntimeError('No MixFFN target conv layers found for global pruning.')

    dg = tp.DependencyGraph()
    h, w = shape
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device('cpu')
    example = torch.randn(1, 3, h, w, device=model_device)
    dg.build_dependency(model, example_inputs=example)

    all_scores: List[Tuple[float, int, int]] = []
    total_channels = 0

    for layer_idx, (_, layer) in enumerate(targets):
        weight = layer.weight.detach()
        l1 = weight.abs().flatten(1).sum(1)
        total_channels += int(l1.numel())
        for ch_idx, score in enumerate(l1.tolist()):
            all_scores.append((float(score), layer_idx, ch_idx))

    budget = max(1, int(total_channels * pruning_ratio))
    all_scores.sort(key=lambda x: x[0])
    picked = all_scores[:budget]

    layer_to_idxs: Dict[int, List[int]] = defaultdict(list)
    for _, layer_idx, ch_idx in picked:
        layer_to_idxs[layer_idx].append(ch_idx)

    applied_layers = 0
    applied_channels = 0

    for layer_idx, idxs in layer_to_idxs.items():
        _, layer = targets[layer_idx]
        unique_idxs = sorted(set(idxs))
        if len(unique_idxs) >= layer.out_channels:
            unique_idxs = unique_idxs[:layer.out_channels - 1]
        if not unique_idxs:
            continue

        group = dg.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=unique_idxs)
        if hasattr(group, 'is_pruned') and group.is_pruned:
            continue

        group.prune()
        applied_layers += 1
        applied_channels += len(unique_idxs)

    return {
        'target_layer_count': len(targets),
        'total_candidate_channels': total_channels,
        'global_budget_channels': budget,
        'applied_layer_count': applied_layers,
        'applied_channel_count': applied_channels,
    }


def eval_miou_in_memory(model: torch.nn.Module, cfg: Config, device: str) -> Dict[str, float]:
    cfg_local = copy.deepcopy(cfg)
    cfg_local.launcher = 'none'
    if 'val_dataloader' in cfg_local:
        cfg_local.val_dataloader.num_workers = 0
        cfg_local.val_dataloader.persistent_workers = False
        cfg_local.val_dataloader.pin_memory = False

    val_loader = Runner.build_dataloader(cfg_local.val_dataloader)
    metric_cfg = cfg_local.val_evaluator
    if isinstance(metric_cfg, list):
        # Keep this utility focused on the common single-metric setting.
        if len(metric_cfg) != 1:
            raise ValueError('eval_miou_in_memory currently supports one val_evaluator metric config.')
        metric_cfg = metric_cfg[0]

    metric = METRICS.build(metric_cfg)

    # Provide dataset meta needed by IoUMetric (classes/palette).
    dataset_meta = None
    if hasattr(val_loader, 'dataset'):
        ds = val_loader.dataset
        if hasattr(ds, 'metainfo'):
            dataset_meta = ds.metainfo
        elif hasattr(ds, 'METAINFO'):
            dataset_meta = ds.METAINFO
    if dataset_meta is not None:
        metric.dataset_meta = dataset_meta

    # Clear residual states in case this function is called multiple times.
    if hasattr(metric, 'results') and isinstance(metric.results, list):
        metric.results.clear()

    model.eval()
    for data in val_loader:
        data = model.data_preprocessor(data, False)
        inputs = normalize_batched_inputs(data['inputs'])
        data_samples = data['data_samples']

        if device.startswith('cuda') and torch.cuda.is_available():
            inputs = inputs.to(device)
            data_samples = [x.to(device) for x in data_samples]

        with torch.no_grad():
            outputs = model(inputs, data_samples, mode='predict')

        # IoUMetric (mmseg 1.2.x) expects dict-like samples with pred/gt fields.
        # Merge gt from batch samples when prediction samples do not carry it,
        # then convert to dict for compatibility.
        merged_outputs = []
        for pred, gt in zip(outputs, data_samples):
            if not hasattr(pred, 'gt_sem_seg') and hasattr(gt, 'gt_sem_seg'):
                pred.gt_sem_seg = gt.gt_sem_seg
            if not hasattr(pred, 'img_path') and hasattr(gt, 'img_path'):
                pred.img_path = gt.img_path
            merged_outputs.append(pred.to_dict() if hasattr(pred, 'to_dict') else pred)

        metric.process(data_batch={}, data_samples=merged_outputs)

    metrics = metric.evaluate(len(val_loader.dataset))
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except Exception:
            continue

    if 'mIoU' in out:
        out['mIoU'] = to_percent(out['mIoU'])
    if 'aAcc' in out:
        out['aAcc'] = to_percent(out['aAcc'])
    if 'mAcc' in out:
        out['mAcc'] = to_percent(out['mAcc'])
    return out


def eval_flops_params_in_memory(model: torch.nn.Module, shape: Tuple[int, int]) -> Dict[str, float]:
    model.eval()
    h, w = shape
    data_info = {'ori_shape': (h, w), 'pad_shape': (h, w)}
    data_batch = {
        'inputs': [torch.rand((3, h, w))],
        'data_samples': [SegDataSample(metainfo=data_info)],
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


def eval_latency_in_memory(
    model: torch.nn.Module,
    cfg: Config,
    device: str,
    warmup: int,
    total_iters: int,
    repeat_times: int,
) -> Dict[str, Any]:
    if total_iters <= warmup:
        raise ValueError(f'total_iters({total_iters}) must be > warmup({warmup})')

    cfg_local = copy.deepcopy(cfg)
    cfg_local.test_dataloader.batch_size = 1
    cfg_local.test_dataloader.num_workers = 0
    cfg_local.test_dataloader.persistent_workers = False
    cfg_local.test_dataloader.pin_memory = False
    data_loader = Runner.build_dataloader(cfg_local.test_dataloader)

    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    all_fps: List[float] = []
    all_latency_ms: List[float] = []

    model.eval()
    for _ in range(repeat_times):
        data_iter = cycle(data_loader)
        measured: List[float] = []

        for i in range(total_iters):
            data = next(data_iter)
            data = model.data_preprocessor(data, True)
            inputs = normalize_batched_inputs(data['inputs'])
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


def finetune_pruned_model(
    model: torch.nn.Module,
    cfg: Config,
    device: str,
    iters: int,
    lr: float,
    weight_decay: float,
    eval_interval: int,
    log_interval: int,
    save_best: str,
    save_last: str,
) -> Dict[str, Any]:
    cfg_local = copy.deepcopy(cfg)
    cfg_local.train_dataloader.batch_size = cfg_local.train_dataloader.get('batch_size', 2)
    cfg_local.train_dataloader.num_workers = 0
    cfg_local.train_dataloader.persistent_workers = False
    cfg_local.train_dataloader.pin_memory = False
    if 'val_dataloader' in cfg_local:
        cfg_local.val_dataloader.num_workers = 0
        cfg_local.val_dataloader.persistent_workers = False
        cfg_local.val_dataloader.pin_memory = False

    train_loader = Runner.build_dataloader(cfg_local.train_dataloader)
    train_iter = cycle(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    model.train()
    best_miou = -1e9
    best_iter = -1
    losses: List[float] = []

    for step in range(1, iters + 1):
        data = next(train_iter)
        data = model.data_preprocessor(data, True)
        inputs = normalize_batched_inputs(data['inputs'])
        data_samples = data['data_samples']

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(inputs, data_samples, mode='loss')
        loss = sum_loss_values(loss_dict)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        if step % max(1, log_interval) == 0:
            print(f'[finetune] iter={step}/{iters}, loss={loss_value:.6f}')

        if step % max(1, eval_interval) == 0 or step == iters:
            miou_metrics = eval_miou_in_memory(model, cfg_local, device)
            cur_miou = float(miou_metrics.get('mIoU', -1e9))
            print(f"[finetune] eval iter={step}, mIoU={cur_miou:.2f}")
            if cur_miou > best_miou:
                best_miou = cur_miou
                best_iter = step
                Path(save_best).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_best)

    Path(save_last).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_last)

    return {
        'enabled': True,
        'iters': iters,
        'lr': lr,
        'weight_decay': weight_decay,
        'eval_interval': eval_interval,
        'log_interval': log_interval,
        'loss_mean': float(statistics.mean(losses)) if losses else None,
        'loss_last': float(losses[-1]) if losses else None,
        'best_miou_during_ft': float(best_miou) if best_iter > 0 else None,
        'best_iter': int(best_iter) if best_iter > 0 else None,
        'best_ckpt': save_best,
        'last_ckpt': save_last,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Global channel pruning + optional fine-tune + evaluation for SegFormer FFN')

    parser.add_argument('--config', required=True, help='Path to mmseg config .py')
    parser.add_argument('--checkpoint', required=True, help='Path to baseline checkpoint .pth')

    parser.add_argument('--pruning-ratio', type=float, default=0.1)
    parser.add_argument('--target-stages', type=int, nargs='*', default=[1, 2, 3])
    parser.add_argument('--max-target-layers', type=int, default=4)

    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--work-dir', default='runs/rs19/prune_eval')

    parser.add_argument('--pruned-checkpoint', default='checkpoints/globally_pruned_ffn.pth')
    parser.add_argument('--output-json', default='exports/pruned_eval.json')

    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--repeat', type=int, default=1)

    parser.add_argument('--skip-miou', action='store_true')
    parser.add_argument('--skip-flops', action='store_true')
    parser.add_argument('--skip-latency', action='store_true')

    parser.add_argument('--enable-finetune', action='store_true')
    parser.add_argument('--finetune-iters', type=int, default=1000)
    parser.add_argument('--finetune-lr', type=float, default=1e-5)
    parser.add_argument('--finetune-weight-decay', type=float, default=1e-2)
    parser.add_argument('--finetune-eval-interval', type=int, default=200)
    parser.add_argument('--finetune-log-interval', type=int, default=50)
    parser.add_argument('--finetune-save-best', default='checkpoints/global_pruned_finetune_best.pth')
    parser.add_argument('--finetune-save-last', default='checkpoints/global_pruned_finetune_last.pth')

    return parser.parse_args()


def maybe_to_device(model: torch.nn.Module, device: str) -> str:
    if device.startswith('cuda') and torch.cuda.is_available():
        model.to(device)
        return device
    print('[WARN] CUDA not available or disabled, fallback to CPU.')
    return 'cpu'


def eval_bundle(model: torch.nn.Module, cfg: Config, args: argparse.Namespace, device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not args.skip_miou:
        out['miou_metrics'] = eval_miou_in_memory(model, cfg, device)
    if not args.skip_flops:
        out['complexity'] = eval_flops_params_in_memory(model, (args.shape[0], args.shape[1]))
    if not args.skip_latency:
        out['latency'] = eval_latency_in_memory(
            model,
            cfg,
            device=device,
            warmup=args.warmup,
            total_iters=args.iters,
            repeat_times=args.repeat,
        )
    return out


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    prepare_pythonpath(project_root)

    register_all_modules(init_default_scope=True)

    print(f"Loading model from config '{args.config}'...")
    model, cfg = get_model_from_config(args.config, args.checkpoint)

    device = maybe_to_device(model, args.device)

    prune_meta = global_prune_ffn(
        model,
        shape=(args.shape[0], args.shape[1]),
        pruning_ratio=args.pruning_ratio,
        target_stages=args.target_stages,
        max_target_layers=args.max_target_layers,
    )

    pruned_ckpt = Path(args.pruned_checkpoint)
    if not pruned_ckpt.is_absolute():
        pruned_ckpt = project_root / pruned_ckpt
    pruned_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pruned_ckpt)
    print(f'[OK] pruned checkpoint saved: {pruned_ckpt}')

    result: Dict[str, Any] = {
        'config': args.config,
        'baseline_checkpoint': args.checkpoint,
        'pruned_checkpoint': str(pruned_ckpt),
        'pruning_ratio': args.pruning_ratio,
        'target_stages': list(args.target_stages),
        'max_target_layers': args.max_target_layers,
        'prune_scope': 'segformer_mixffn_layers0_conv',
        'prune_meta': prune_meta,
        'device': device,
        'shape': list(args.shape),
        'finetune_enabled': bool(args.enable_finetune),
    }

    print('[1/2] Evaluating post-prune model...')
    result['post_prune_eval'] = eval_bundle(model, cfg, args, device)

    if args.enable_finetune:
        print('[2/2] Running short fine-tuning...')
        ft_info = finetune_pruned_model(
            model,
            cfg,
            device=device,
            iters=args.finetune_iters,
            lr=args.finetune_lr,
            weight_decay=args.finetune_weight_decay,
            eval_interval=args.finetune_eval_interval,
            log_interval=args.finetune_log_interval,
            save_best=args.finetune_save_best,
            save_last=args.finetune_save_last,
        )
        result['finetune'] = ft_info
        result['post_finetune_eval'] = eval_bundle(model, cfg, args, device)

    out_json = Path(args.output_json)
    if not out_json.is_absolute():
        out_json = project_root / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] JSON saved to: {out_json}')


if __name__ == '__main__':
    main()
