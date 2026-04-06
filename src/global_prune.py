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

def main():
    # --- 1. ???? ---
    config_path = 'configs/railsem19/segformer_b0_rs19_512x512_40000it.py'
    checkpoint_path = 'runs/rs19/segformer_b0_512x512_40000it/best_mIoU_iter_40000.pth'
    pruning_ratio = 0.2
    
    print(f"Loading model from config '{config_path}'...")
    model, cfg = get_model_from_config(config_path, checkpoint_path)
    
    # --- 2. ????????? ---
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 512, 512))
    
    targets_to_prune = []
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
            print(f"Warning: Target {i+1} has been pruned, skipping.")
        else:
            group.prune()

    print("Global pruning finished.")

    print("\n--- Pruned Model Structure ---")
    print(model)
    
    try:
        print("\nTesting forward pass...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 512, 512)
            output = model(test_input)
        print(f"Forward pass successful! Output shape: {output[0].shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

    pruned_checkpoint_path = f'checkpoints/globally_pruned_ffn_{int(pruning_ratio*100)}p.pth'
    os.makedirs(os.path.dirname(pruned_checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), pruned_checkpoint_path)
    print(f"\nPruned model saved to: {pruned_checkpoint_path}")

if __name__ == '__main__':
    main()
