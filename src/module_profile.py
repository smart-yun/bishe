# -*- coding: utf-8 -*-
"""
Module-level resource profiler for MMSeg models (e.g. SegFormer).

Outputs:
- Params share per core module
- Approx FLOPs share per core module
- Recommended pruning priority

Example:
python src/module_profile.py \
  --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py \
  --checkpoint runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth \
  --device cuda:0 \
  --shape 512 512 \
  --output-json exports/module_profile_80k.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def _bootstrap_runtime() -> None:
    """Ensure mmcv/mmseg are available by switching to known conda envs if needed."""
    try:
        import mmcv  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != 'mmcv':
            raise

    candidates: List[str] = []
    env_python = os.environ.get('BISHE_PYTHON')
    if env_python:
        candidates.append(env_python)

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidates.extend([
            os.path.join(conda_prefix, 'envs', 'railseg', 'bin', 'python'),
            os.path.join(conda_prefix, 'envs', 'railseg2', 'bin', 'python'),
        ])

    candidates.extend([
        '/home/lcy/miniconda3/envs/railseg/bin/python',
        '/home/lcy/miniconda3/envs/railseg2/bin/python',
    ])

    current_python = os.path.realpath(sys.executable)
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.path.realpath(candidate) != current_python:
            os.execv(candidate, [candidate, *sys.argv])

    raise ModuleNotFoundError(
        "No module named 'mmcv'. Run with railseg env or set BISHE_PYTHON to a valid interpreter."
    )


_bootstrap_runtime()

from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Profile module-level params/FLOPs for pruning')
    parser.add_argument('--config', required=True, help='Path to mmseg config .py')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pth')
    parser.add_argument('--device', default='cuda:0', help='Device, e.g. cuda:0 or cpu')
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512], help='Input H W')
    parser.add_argument('--topk', type=int, default=20, help='Print top-k modules by priority')
    parser.add_argument('--output-json', default='exports/module_profile.json', help='Output JSON path')
    return parser.parse_args()


def prepare_pythonpath(project_root: Path) -> None:
    src_dir = str(project_root / 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def map_to_core_module(module_name: str) -> str:
    if module_name.startswith('backbone.layers.'):
        parts = module_name.split('.')
        stage = int(parts[2]) + 1 if len(parts) > 2 and parts[2].isdigit() else 0

        if len(parts) > 3 and parts[3] == '0':
            return f'backbone.stage{stage}.patch_embed'
        if '.attn' in module_name:
            return f'backbone.stage{stage}.attn'
        if '.ffn' in module_name:
            return f'backbone.stage{stage}.ffn'
        if len(parts) > 3 and parts[3] == '2':
            return f'backbone.stage{stage}.norm'
        return f'backbone.stage{stage}.other'

    if module_name.startswith('decode_head.convs'):
        return 'decode_head.proj'
    if module_name.startswith('decode_head.fusion_conv'):
        return 'decode_head.fusion'
    if module_name.startswith('decode_head.cls_seg'):
        return 'decode_head.cls'
    if module_name.startswith('decode_head'):
        return 'decode_head.other'
    if module_name.startswith('backbone'):
        return 'backbone.other'
    return 'model.other'


def build_model(config_path: str, checkpoint_path: str, device: str):
    cfg = Config.fromfile(config_path)
    cfg.launcher = 'none'

    init_default_scope(cfg.get('default_scope', 'mmseg'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')

    model = revert_sync_batchnorm(model)
    model.eval()

    if device.startswith('cuda') and torch.cuda.is_available():
        model = model.to(device)
    else:
        device = 'cpu'

    return model, device


def collect_param_stats(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'params': 0, 'flops': 0.0, 'examples': set()})

    for name, p in model.named_parameters():
        module_name = name.rsplit('.', 1)[0] if '.' in name else ''
        core = map_to_core_module(module_name)
        stats[core]['params'] += int(p.numel())
        if module_name and len(stats[core]['examples']) < 3:
            stats[core]['examples'].add(module_name)

    return stats


def conv2d_flops(mod: nn.Conv2d, out: torch.Tensor) -> float:
    b, cout, hout, wout = out.shape
    cin = mod.in_channels
    groups = mod.groups
    kh, kw = mod.kernel_size
    return float(2.0 * b * hout * wout * cout * (cin / groups) * kh * kw)


def linear_flops(mod: nn.Linear, x: torch.Tensor) -> float:
    vectors = x.numel() / mod.in_features
    return float(2.0 * vectors * mod.in_features * mod.out_features)


def mha_extra_flops(mod: nn.MultiheadAttention, q: torch.Tensor, k: torch.Tensor) -> float:
    if q.ndim != 3 or k.ndim != 3:
        return 0.0
    if mod.batch_first:
        b, lq, e = q.shape
        lk = k.shape[1]
    else:
        lq, b, e = q.shape
        lk = k.shape[0]
    h = max(1, mod.num_heads)
    d = e // h if h > 0 else e

    # QK^T and Attn*V
    attn_matmuls = 2.0 * b * h * lq * lk * d + 2.0 * b * h * lq * lk * d
    # Approx in-proj for q/k/v when fused in functional path
    in_proj = 2.0 * b * (lq * e * e + lk * e * e + lk * e * e)
    return float(attn_matmuls + in_proj)


def register_flops_hooks(model: nn.Module, stats: Dict[str, Dict[str, Any]]):
    handles = []

    for module_name, module in model.named_modules():
        core = map_to_core_module(module_name)

        if isinstance(module, nn.Conv2d):
            def conv_hook(mod, inputs, outputs, core_key=core, mname=module_name):
                out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                if isinstance(out, torch.Tensor) and out.ndim == 4:
                    stats[core_key]['flops'] += conv2d_flops(mod, out)
                    if mname and len(stats[core_key]['examples']) < 3:
                        stats[core_key]['examples'].add(mname)
            handles.append(module.register_forward_hook(conv_hook))

        elif isinstance(module, nn.Linear):
            def linear_hook(mod, inputs, outputs, core_key=core, mname=module_name):
                if not inputs:
                    return
                x = inputs[0]
                if isinstance(x, torch.Tensor):
                    stats[core_key]['flops'] += linear_flops(mod, x)
                    if mname and len(stats[core_key]['examples']) < 3:
                        stats[core_key]['examples'].add(mname)
            handles.append(module.register_forward_hook(linear_hook))

        elif isinstance(module, nn.MultiheadAttention):
            def mha_hook(mod, inputs, outputs, core_key=core, mname=module_name):
                if len(inputs) < 2:
                    return
                q, k = inputs[0], inputs[1]
                if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
                    stats[core_key]['flops'] += mha_extra_flops(mod, q, k)
                    if mname and len(stats[core_key]['examples']) < 3:
                        stats[core_key]['examples'].add(mname)
            handles.append(module.register_forward_hook(mha_hook))

    return handles


def run_one_forward(model: nn.Module, device: str, shape: Tuple[int, int]) -> None:
    h, w = shape
    batch = {
        'inputs': [torch.rand((3, h, w))],
        'data_samples': [
            SegDataSample(
                metainfo={
                    'ori_shape': (h, w),
                    'img_shape': (h, w),
                    'pad_shape': (h, w),
                    'scale_factor': (1.0, 1.0),
                    'flip': False,
                    'flip_direction': None,
                })
        ],
    }
    data = model.data_preprocessor(batch)

    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        # Use tensor mode to avoid decode-head predict() metadata requirements
        # (e.g. img_shape) during synthetic one-pass FLOPs profiling.
        model(data['inputs'], data_samples=None, mode='tensor')
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def pruneability_multiplier(name: str) -> float:
    if '.ffn' in name:
        return 1.25
    if '.attn' in name:
        return 1.10
    if '.patch_embed' in name:
        return 0.95
    if name.startswith('decode_head'):
        return 0.90
    if name.startswith('backbone'):
        return 0.80
    return 0.60


def pruning_tip(name: str) -> str:
    if '.ffn' in name:
        return 'HIGH priority: prune 1x1 channels first (L1 / importance score).'
    if '.attn' in name:
        return 'MED-HIGH priority: prune qkv/proj with head-dim checks.'
    if '.patch_embed' in name:
        return 'MEDIUM priority: use small pruning ratios.'
    if name.startswith('decode_head'):
        return 'MEDIUM priority: useful but usually below backbone impact.'
    return 'LOW priority: prune after hotspot modules.'


def build_report(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    total_params = sum(v['params'] for v in stats.values())
    total_flops = sum(v['flops'] for v in stats.values())

    rows: List[Dict[str, Any]] = []
    for core, v in stats.items():
        params = int(v['params'])
        flops = float(v['flops'])
        p_share = (params / total_params * 100.0) if total_params > 0 else 0.0
        f_share = (flops / total_flops * 100.0) if total_flops > 0 else 0.0
        score = (0.55 * (f_share / 100.0) + 0.45 * (p_share / 100.0)) * pruneability_multiplier(core)

        rows.append({
            'core_module': core,
            'params': params,
            'params_m': params / 1e6,
            'params_share_pct': p_share,
            'flops': flops,
            'flops_g': flops / 1e9,
            'flops_share_pct': f_share,
            'priority_score': score,
            'example_layers': sorted(v['examples']),
            'recommendation': pruning_tip(core),
        })

    rows.sort(key=lambda x: x['priority_score'], reverse=True)

    n = len(rows)
    if n > 0:
        high_cut = max(1, int(round(0.30 * n)))
        med_cut = max(high_cut + 1, int(round(0.70 * n)))
        for i, row in enumerate(rows):
            if i < high_cut:
                row['priority_level'] = 'HIGH'
            elif i < med_cut:
                row['priority_level'] = 'MEDIUM'
            else:
                row['priority_level'] = 'LOW'

    return {
        'totals': {
            'params': total_params,
            'params_m': total_params / 1e6,
            'flops': total_flops,
            'flops_g': total_flops / 1e9,
            'flops_note': 'Approximate FLOPs from Conv2d/Linear hooks + estimated MHA cost',
        },
        'modules': rows,
    }


def print_topk(report: Dict[str, Any], topk: int) -> None:
    totals = report['totals']
    rows = report['modules'][:topk]

    print('\n================ Module Profile Summary ================')
    print(f"Total Params: {totals['params_m']:.3f} M")
    print(f"Total FLOPs:  {totals['flops_g']:.3f} G (approx)")
    print('--------------------------------------------------------')
    print(f"{'Rank':<5} {'Core Module':<34} {'Param(M)':>9} {'P%':>7} {'FLOPs(G)':>10} {'F%':>7} {'Prio':>7}")
    print('-' * 92)

    for i, r in enumerate(rows, start=1):
        print(
            f"{i:<5} {r['core_module']:<34} {r['params_m']:>9.3f} {r['params_share_pct']:>6.2f}% "
            f"{r['flops_g']:>10.3f} {r['flops_share_pct']:>6.2f}% {r['priority_level']:>7}"
        )

    print('\nTop recommendations:')
    for r in rows[: min(5, len(rows))]:
        print(f"- [{r['priority_level']}] {r['core_module']}: {r['recommendation']}")
    print('========================================================\n')


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    prepare_pythonpath(project_root)
    register_all_modules(init_default_scope=True)

    model, device = build_model(args.config, args.checkpoint, args.device)

    stats = collect_param_stats(model)
    hooks = register_flops_hooks(model, stats)
    try:
        run_one_forward(model, device, (args.shape[0], args.shape[1]))
    finally:
        for h in hooks:
            h.remove()

    report = build_report(stats)
    print_topk(report, args.topk)

    output = Path(args.output_json)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': device,
        'shape': args.shape,
        **report,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] JSON saved to: {output}')


if __name__ == '__main__':
    main()
