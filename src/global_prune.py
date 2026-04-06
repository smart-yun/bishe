# -*- coding: utf-8 -*-
"""
Global pruning + optional short fine-tuning + evaluation for SegFormer (MixFFN conv1 channels).

Features:
1) True global ranking pruning (rank channels across all candidate layers).
2) Evaluate pruned in-memory model directly (mIoU / Params / FLOPs / Latency).
3) Optional short fine-tuning on pruned model with a switch.
4) Export JSON for report-ready comparison (immediate vs post-finetune).

usage
python -m py_compile src/global_prune.py && conda run -n railseg python src/global_prune.py --help | head -n 60
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

example:

#smoke_test
conda run -n railseg python src/global_prune.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py \
  --checkpoint runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth \
  --pruning-ratio 0.10 \
  --target-stages 1 2 3 \
  --max-target-layers 4 \
  --skip-miou --skip-latency \
  --output-json exports/global_pruned_eval_r10_smoke.json \
  --device cuda:0

#no-tuning
conda run -n railseg python src/global_prune.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py \
  --checkpoint runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth \
  --pruning-ratio 0.10 \
  --target-stages 1 2 3 \
  --max-target-layers 4 \
  --skip-latency \
  --output-json exports/global_pruned_eval_r10_miou.json \
  --device cuda:0


#finetune
conda run -n railseg python src/global_prune.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py \
  --checkpoint runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth \
  --pruning-ratio 0.10 \
  --target-stages 1 2 3 \
  --max-target-layers 4 \
  --enable-finetune \
  --finetune-iters 1000 \
  --finetune-lr 1e-5 \
  --finetune-weight-decay 1e-2 \
  --finetune-eval-interval 200 \
  --finetune-log-interval 50 \
  --finetune-save-best checkpoints/global_pruned_r10_finetune_best.pth \
  --finetune-save-last checkpoints/global_pruned_r10_finetune_last.pth \
  --skip-latency \
  --output-json exports/global_pruned_eval_r10_finetune.json \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time
from collections import defaultdict
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

try:
    from mmseg.models.backbones.mit import MixFFN  # mmseg >= 1.x
except ImportError:
    from mmseg.models.backbones.mix_transformer import MixFFN  # mmseg < 1.x


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


def get_model_from_config(config_path: str, checkpoint_path: str):
    cfg = Config.fromfile(config_path)
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


def eval_miou_with_pruned_model(cfg: Config, model: torch.nn.Module, work_dir: str) -> Dict[str, float]:
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval.launcher = 'none'
    cfg_eval.work_dir = work_dir
    cfg_eval.load_from = None
    cfg_eval.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer')

    runner = Runner.from_cfg(cfg_eval)
    # Use pruned in-memory model directly.
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


def _parse_stage_from_target_name(target_name: str) -> int | None:
    # e.g. backbone.layers.2.1.0.ffn.layers.0
    parts = target_name.split('.')
    if len(parts) > 2 and parts[0] == 'backbone' and parts[1] == 'layers':
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def collect_ffn_first_convs(
    model: torch.nn.Module,
    target_stages: List[int] | None = None,
    max_target_layers: int = 0,
) -> List[Tuple[str, torch.nn.Module]]:
    targets: List[Tuple[str, torch.nn.Module]] = []
    allowed = set(target_stages) if target_stages else None

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
    dg.build_dependency(model, example_inputs=torch.randn(1, 3, h, w))

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


def short_finetune_pruned_model(
    cfg: Config,
    model: torch.nn.Module,
    device: str,
    total_iters: int,
    lr: float,
    weight_decay: float,
    eval_interval: int,
    work_dir: str,
    save_best_path: str,
    save_last_path: str,
    log_interval: int,
) -> Dict[str, Any]:
    if total_iters <= 0:
        raise ValueError('finetune-iters must be > 0 when fine-tune is enabled')

    cfg_train = copy.deepcopy(cfg)
    train_loader = Runner.build_dataloader(cfg_train.train_dataloader)

    use_cuda = device.startswith('cuda') and torch.cuda.is_available()
    if use_cuda:
        model = model.to(device)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_miou = -1.0
    best_iter = 0
    iter_losses: List[float] = []
    data_iter = cycle(train_loader)

    os.makedirs(Path(save_best_path).parent.as_posix(), exist_ok=True)
    os.makedirs(Path(save_last_path).parent.as_posix(), exist_ok=True)

    for it in range(1, total_iters + 1):
        batch = next(data_iter)
        batch = model.data_preprocessor(batch, True)

        loss_dict = model(batch['inputs'], batch['data_samples'], mode='loss')
        loss = sum_loss_values(loss_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        iter_losses.append(loss_val)

        if log_interval > 0 and (it % log_interval == 0 or it == 1 or it == total_iters):
            print(f'[ft] iter {it}/{total_iters} loss={loss_val:.6f}')

        if eval_interval > 0 and it % eval_interval == 0:
            model.eval()
            metrics = eval_miou_with_pruned_model(cfg, model, work_dir)
            cur_miou = float(metrics.get('mIoU', -1.0))
            print(f'[ft] eval@{it}: mIoU={cur_miou:.4f}')
            if cur_miou > best_miou:
                best_miou = cur_miou
                best_iter = it
                torch.save(model.state_dict(), save_best_path)
                print(f'[ft] new best saved: {save_best_path}')
            model.train()

    torch.save(model.state_dict(), save_last_path)
    print(f'[ft] last checkpoint saved: {save_last_path}')

    # If eval never ran, best == last.
    if best_miou < 0:
        torch.save(model.state_dict(), save_best_path)
        best_miou = float('nan')
        best_iter = total_iters

    model.eval()
    return {
        'enabled': True,
        'iters': total_iters,
        'lr': lr,
        'weight_decay': weight_decay,
        'eval_interval': eval_interval,
        'log_interval': log_interval,
        'loss_mean': float(statistics.mean(iter_losses)) if iter_losses else None,
        'loss_last': float(iter_losses[-1]) if iter_losses else None,
        'best_miou_during_ft': best_miou,
        'best_iter': best_iter,
        'best_ckpt': save_best_path,
        'last_ckpt': save_last_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Global channel pruning + optional fine-tune + evaluation for SegFormer FFN')
    parser.add_argument('--config', required=True, help='Path to mmseg config .py')
    parser.add_argument('--checkpoint', required=True, help='Path to baseline checkpoint .pth')
    parser.add_argument('--pruning-ratio', type=float, default=0.1, help='Global channel pruning ratio in (0,1)')
    parser.add_argument('--target-stages', type=int, nargs='*', default=None,
                        help='Only prune selected backbone stages, e.g. --target-stages 1 2 3')
    parser.add_argument('--max-target-layers', type=int, default=0,
                        help='Max number of candidate layers to prune, 0 means no limit')
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512], help='Input shape H W')
    parser.add_argument('--device', default='cuda:0', help='Device, e.g. cuda:0 or cpu')

    parser.add_argument('--work-dir', default='runs/rs19/global_prune_eval_script', help='Work dir for eval artifacts')
    parser.add_argument('--pruned-checkpoint', default='checkpoints/globally_pruned_ffn.pth', help='Path to save pruned state_dict')
    parser.add_argument('--output-json', default='exports/global_pruned_eval.json', help='Output JSON path')

    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--skip-miou', action='store_true')
    parser.add_argument('--skip-flops', action='store_true')
    parser.add_argument('--skip-latency', action='store_true')

    # Fine-tune switch (off by default)
    parser.add_argument('--enable-finetune', action='store_true', help='Enable short fine-tuning after pruning')
    parser.add_argument('--finetune-iters', type=int, default=1000, help='Total fine-tune iterations')
    parser.add_argument('--finetune-lr', type=float, default=1e-5, help='Fine-tune learning rate')
    parser.add_argument('--finetune-weight-decay', type=float, default=1e-2, help='Fine-tune weight decay')
    parser.add_argument('--finetune-eval-interval', type=int, default=0,
                        help='Run val mIoU every N iters during fine-tune, 0 means no mid-eval')
    parser.add_argument('--finetune-log-interval', type=int, default=50, help='Print train loss every N iters')
    parser.add_argument('--finetune-save-best', default='checkpoints/global_pruned_finetune_best.pth',
                        help='Path to save best fine-tuned checkpoint')
    parser.add_argument('--finetune-save-last', default='checkpoints/global_pruned_finetune_last.pth',
                        help='Path to save last fine-tuned checkpoint')
    return parser.parse_args()


def run_eval_bundle(cfg: Config, model: torch.nn.Module, args: argparse.Namespace, device: str, prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not args.skip_miou:
        print(f'[eval:{prefix}] mIoU on pruned in-memory model...')
        out['miou_metrics'] = eval_miou_with_pruned_model(cfg, model, args.work_dir)

    if not args.skip_flops:
        print(f'[eval:{prefix}] Params/FLOPs on pruned in-memory model...')
        out['complexity'] = eval_flops_params_from_model(model, (args.shape[0], args.shape[1]), device)

    if not args.skip_latency:
        print(f'[eval:{prefix}] Latency/FPS on pruned in-memory model...')
        out['latency'] = eval_latency_from_model(
            cfg=cfg,
            model=model,
            device=device,
            warmup=args.warmup,
            total_iters=args.iters,
            repeat_times=args.repeat,
        )
    return out


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

    print(f"[prune] true global FFN pruning ratio={args.pruning_ratio:.4f}")
    prune_meta = global_prune_ffn(
        model,
        (args.shape[0], args.shape[1]),
        args.pruning_ratio,
        target_stages=args.target_stages,
        max_target_layers=args.max_target_layers,
    )
    print(f"[prune] applied {prune_meta['applied_channel_count']} channels over {prune_meta['applied_layer_count']} layers")

    print('[check] forward smoke test...')
    with torch.no_grad():
        x = torch.randn(1, 3, args.shape[0], args.shape[1])
        _ = model(x)

    os.makedirs(os.path.dirname(args.pruned_checkpoint), exist_ok=True)
    torch.save(model.state_dict(), args.pruned_checkpoint)
    print(f"[save] pruned checkpoint: {args.pruned_checkpoint}")

    result: Dict[str, Any] = {
        'config': args.config,
        'baseline_checkpoint': args.checkpoint,
        'pruned_checkpoint': args.pruned_checkpoint,
        'pruning_ratio': args.pruning_ratio,
        'target_stages': args.target_stages,
        'max_target_layers': args.max_target_layers,
        'prune_scope': 'segformer_mixffn_layers0_conv',
        'prune_meta': prune_meta,
        'device': device,
        'shape': args.shape,
        'finetune_enabled': bool(args.enable_finetune),
    }

    # 1) Immediate post-prune evaluation
    result['post_prune_eval'] = run_eval_bundle(cfg, model, args, device, prefix='post_prune')

    # 2) Optional short fine-tune
    if args.enable_finetune:
        print('[ft] start short fine-tuning...')
        ft_info = short_finetune_pruned_model(
            cfg=cfg,
            model=model,
            device=device,
            total_iters=args.finetune_iters,
            lr=args.finetune_lr,
            weight_decay=args.finetune_weight_decay,
            eval_interval=args.finetune_eval_interval,
            work_dir=args.work_dir,
            save_best_path=args.finetune_save_best,
            save_last_path=args.finetune_save_last,
            log_interval=args.finetune_log_interval,
        )
        result['finetune'] = ft_info

        # Load best fine-tuned weights and evaluate again
        best_ckpt = ft_info.get('best_ckpt')
        if best_ckpt and Path(best_ckpt).exists():
            state_dict = torch.load(best_ckpt, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model.eval()

        result['post_finetune_eval'] = run_eval_bundle(cfg, model, args, device, prefix='post_finetune')

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[done] JSON saved to: {output_json}")


if __name__ == '__main__':
    main()
