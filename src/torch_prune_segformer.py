# -*- coding: utf-8 -*-
"""
SegFormer + Torch-Pruning (High-level / Iterative / Global / Isomorphic)

========================= Runtime Argument Templates =========================

# 0) Recommended: set project PYTHONPATH first
# export PYTHONPATH=/home/lcy/Projects/bishe/src:${PYTHONPATH}

# 1) Local + Iterative (safe starting point)
# python src/torch_prune_segformer.py \
#   --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py \
#   --checkpoint runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth \
#   --pruning-ratio 0.30 \
#   --iterative-steps 5 \
#   --shape 512 512 \
#   --device cuda:0 \
#   --work-dir runs/rs19/tp_local_iter \
#   --save-model checkpoints/tp_local_iter_model.pth \
#   --save-tp-state checkpoints/tp_local_iter_tpstate.pth \
#   --save-json exports/tp_local_iter_summary.json

# 2) Global + Iterative
# python src/torch_prune_segformer.py \
#   --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py \
#   --checkpoint runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth \
#   --pruning-ratio 0.30 \
#   --iterative-steps 5 \
#   --global-pruning \
#   --shape 512 512 \
#   --device cuda:0

# 3) Global + Isomorphic + Iterative (more stable global strategy)
# python src/torch_prune_segformer.py \
#   --config configs/railsem19/segformer_b0_rs19_512x512_80000it_server.py \
#   --checkpoint runs/rs19/segformer_b0_512x512_80000it_server/best_mIoU_iter_79000.pth \
#   --pruning-ratio 0.30 \
#   --iterative-steps 5 \
#   --global-pruning --isomorphic \
#   --round-to 8 \
#   --shape 512 512 \
#   --device cuda:0

Notes:
1) pruning_ratio is channel/dim ratio, not direct parameter ratio.
2) After structural pruning, prefer saving full model object or tp.state_dict.
3) This script ignores decode_head/auxiliary_head classifiers by default.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp

from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmseg.models import build_segmentor
from mmseg.utils import register_all_modules


def prepare_pythonpath(project_root: Path) -> None:
    src_dir = str(project_root / 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Torch-Pruning script for SegFormer (mmseg)')

    parser.add_argument('--config', required=True, help='Path to mmseg config (.py)')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint (.pth)')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 / cpu')
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512], help='Example input shape: H W')

    parser.add_argument('--pruning-ratio', type=float, default=0.30, help='Channel pruning ratio in (0,1)')
    parser.add_argument('--iterative-steps', type=int, default=5, help='Progressive pruning steps')
    parser.add_argument('--global-pruning', action='store_true', help='Enable global ranking')
    parser.add_argument('--isomorphic', action='store_true', help='Enable isomorphic pruning (if supported)')
    parser.add_argument('--max-pruning-ratio', type=float, default=1.0, help='Safety upper bound per layer')
    parser.add_argument('--round-to', type=int, default=8, help='Round channels to multiple of N; <=0 means disabled')

    parser.add_argument(
        '--importance',
        choices=['group_magnitude', 'magnitude', 'taylor'],
        default='group_magnitude',
        help='Importance metric',
    )

    parser.add_argument(
        '--ignore-keywords',
        nargs='*',
        default=['decode_head.conv_seg', 'auxiliary_head.conv_seg'],
        help='Module-name keywords to ignore (freeze output channels)',
    )

    parser.add_argument('--work-dir', default='runs/rs19/tp_prune', help='Working directory')
    parser.add_argument('--save-model', default='checkpoints/segformer_tp_pruned_model.pth', help='Save full model object')
    parser.add_argument('--save-tp-state', default='checkpoints/segformer_tp_pruned_tpstate.pth', help='Save tp.state_dict (if available)')
    parser.add_argument('--save-state-dict', default='checkpoints/segformer_tp_pruned_state_dict.pth', help='Save raw model.state_dict')
    parser.add_argument('--save-json', default='exports/segformer_tp_prune_summary.json', help='Summary JSON path')

    parser.add_argument('--count-macs', action='store_true', help='Try counting MACs via tp.utils if supported')
    return parser.parse_args()


def maybe_to_device(model: nn.Module, device: str) -> str:
    if device.startswith('cuda') and torch.cuda.is_available():
        model.to(device)
        return device
    print('[WARN] CUDA not available or disabled; fallback to CPU.')
    model.to('cpu')
    return 'cpu'


def load_model_from_cfg(config_path: str, checkpoint_path: str) -> Tuple[nn.Module, Config]:
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if getattr(incompatible, 'missing_keys', None):
        print(f"[load] missing_keys: {len(incompatible.missing_keys)}")
    if getattr(incompatible, 'unexpected_keys', None):
        print(f"[load] unexpected_keys: {len(incompatible.unexpected_keys)}")

    model = revert_sync_batchnorm(model)
    model.eval()
    return model, cfg


def seg_forward_fn(model: nn.Module, example_inputs: torch.Tensor) -> Any:
    """Forward function for mmseg segmentor in tensor mode."""
    return model(example_inputs, data_samples=None, mode='tensor')


def choose_importance(name: str):
    if name == 'group_magnitude':
        if hasattr(tp.importance, 'GroupMagnitudeImportance'):
            return tp.importance.GroupMagnitudeImportance(p=2)
        return tp.importance.MagnitudeImportance(p=2)

    if name == 'magnitude':
        return tp.importance.MagnitudeImportance(p=2)

    # taylor
    if hasattr(tp.importance, 'TaylorImportance'):
        return tp.importance.TaylorImportance()
    raise RuntimeError('TaylorImportance is not available in current torch_pruning version.')


def collect_ignored_layers(model: nn.Module, keywords: List[str]) -> List[nn.Module]:
    ignored: List[nn.Module] = []

    # 1) Name-based protection
    for name, module in model.named_modules():
        if any(k in name for k in keywords):
            ignored.append(module)

    # 2) Num-class head protection (conv/linear with out_channels/out_features == num_classes)
    num_classes = None
    if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'num_classes'):
        num_classes = int(model.decode_head.num_classes)

    if num_classes is not None:
        for _, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == num_classes:
                ignored.append(module)
            elif isinstance(module, nn.Linear) and module.out_features == num_classes:
                ignored.append(module)

    # deduplicate by object id
    uniq: Dict[int, nn.Module] = {id(m): m for m in ignored}
    out = list(uniq.values())
    print(f'[info] ignored_layers: {len(out)}')
    return out


def build_pruner(
    model: nn.Module,
    example_inputs: torch.Tensor,
    importance,
    args: argparse.Namespace,
    ignored_layers: List[nn.Module],
):
    pruner_cls = getattr(tp.pruner, 'MagnitudePruner', None)
    if pruner_cls is None:
        pruner_cls = getattr(tp.pruner, 'BasePruner', None)
    if pruner_cls is None:
        raise RuntimeError('No compatible high-level pruner found (MagnitudePruner/BasePruner).')

    sig = inspect.signature(pruner_cls.__init__)
    params = sig.parameters
    kwargs: Dict[str, Any] = {}

    if 'importance' in params:
        kwargs['importance'] = importance
    if 'iterative_steps' in params:
        kwargs['iterative_steps'] = args.iterative_steps
    if 'global_pruning' in params:
        kwargs['global_pruning'] = bool(args.global_pruning)
    if 'isomorphic' in params:
        kwargs['isomorphic'] = bool(args.isomorphic)
    if 'pruning_ratio' in params:
        kwargs['pruning_ratio'] = float(args.pruning_ratio)
    elif 'ch_sparsity' in params:
        kwargs['ch_sparsity'] = float(args.pruning_ratio)
    if 'max_pruning_ratio' in params:
        kwargs['max_pruning_ratio'] = float(args.max_pruning_ratio)
    if 'ignored_layers' in params:
        kwargs['ignored_layers'] = ignored_layers
    if 'round_to' in params and args.round_to > 0:
        kwargs['round_to'] = int(args.round_to)
    if 'root_module_types' in params:
        kwargs['root_module_types'] = [nn.Conv2d, nn.Linear]
    if 'forward_fn' in params:
        kwargs['forward_fn'] = seg_forward_fn

    print(f"[info] using pruner: {pruner_cls.__name__}")
    return pruner_cls(model, example_inputs, **kwargs)


def tensor_to_loss_scalar(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output.mean()
    if isinstance(output, (list, tuple)):
        ts = [x for x in output if torch.is_tensor(x)]
        if not ts:
            raise RuntimeError('No tensor found in model output for Taylor loss proxy.')
        return sum(x.mean() for x in ts)
    if isinstance(output, dict):
        ts = [x for x in output.values() if torch.is_tensor(x)]
        if not ts:
            raise RuntimeError('No tensor found in model output dict for Taylor loss proxy.')
        return sum(x.mean() for x in ts)
    raise RuntimeError(f'Unsupported output type for Taylor proxy: {type(output)}')


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def try_count_macs(
    model: nn.Module,
    example_inputs: torch.Tensor,
    enabled: bool,
) -> int | None:
    if not enabled:
        return None
    if not hasattr(tp, 'utils') or not hasattr(tp.utils, 'count_ops_and_params'):
        print('[WARN] tp.utils.count_ops_and_params not available; skip MACs counting.')
        return None

    fn = tp.utils.count_ops_and_params
    try:
        sig = inspect.signature(fn)
        kwargs: Dict[str, Any] = {}
        if 'forward_fn' in sig.parameters:
            kwargs['forward_fn'] = seg_forward_fn
        macs, _ = fn(model, example_inputs, **kwargs)
        return int(macs)
    except Exception as e:
        print(f'[WARN] count_ops_and_params failed: {e}')
        return None


def ensure_parent(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    args = parse_args()

    if not (0.0 < args.pruning_ratio < 1.0):
        raise ValueError(f'--pruning-ratio must be in (0,1), got {args.pruning_ratio}')
    if args.iterative_steps < 1:
        raise ValueError(f'--iterative-steps must be >=1, got {args.iterative_steps}')

    project_root = Path(__file__).resolve().parents[1]
    prepare_pythonpath(project_root)
    register_all_modules(init_default_scope=True)

    model, _ = load_model_from_cfg(args.config, args.checkpoint)
    device = maybe_to_device(model, args.device)
    h, w = int(args.shape[0]), int(args.shape[1])
    example_inputs = torch.randn(1, 3, h, w, device=device)

    ignored_layers = collect_ignored_layers(model, args.ignore_keywords)
    importance = choose_importance(args.importance)
    pruner = build_pruner(model, example_inputs, importance, args, ignored_layers)

    mode_desc = 'local'
    if args.global_pruning and args.isomorphic:
        mode_desc = 'global+isomorphic'
    elif args.global_pruning:
        mode_desc = 'global'

    history: List[Dict[str, Any]] = []
    base_params = param_count(model)
    base_macs = try_count_macs(model, example_inputs, enabled=args.count_macs)
    history.append({'step': 0, 'params': base_params, 'macs': base_macs})
    print(f'[step 0] params={base_params:,}' + (f', macs={base_macs:,}' if base_macs is not None else ''))

    for step in range(1, args.iterative_steps + 1):
        model.train()
        if args.importance == 'taylor':
            model.zero_grad(set_to_none=True)
            out = seg_forward_fn(model, example_inputs)
            proxy_loss = tensor_to_loss_scalar(out)
            proxy_loss.backward()

        model.eval()
        pruner.step()

        cur_params = param_count(model)
        cur_macs = try_count_macs(model, example_inputs, enabled=args.count_macs)
        history.append({'step': step, 'params': cur_params, 'macs': cur_macs})
        print(
            f'[step {step}] params={cur_params:,}, '
            f'pruned={(base_params - cur_params):,}'
            + (f', macs={cur_macs:,}' if cur_macs is not None else '')
        )

    # Save artifacts
    save_model_path = ensure_parent(args.save_model, project_root)
    save_tp_state_path = ensure_parent(args.save_tp_state, project_root)
    save_state_dict_path = ensure_parent(args.save_state_dict, project_root)
    save_json_path = ensure_parent(args.save_json, project_root)

    model.zero_grad(set_to_none=True)
    torch.save(model, save_model_path)
    torch.save(model.state_dict(), save_state_dict_path)

    tp_state_saved = False
    if hasattr(tp, 'state_dict'):
        try:
            pruned_state = tp.state_dict(model)
            torch.save(pruned_state, save_tp_state_path)
            tp_state_saved = True
        except Exception as e:
            print(f'[WARN] tp.state_dict save failed: {e}')

    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': device,
        'shape': [h, w],
        'mode': mode_desc,
        'importance': args.importance,
        'pruning_ratio': float(args.pruning_ratio),
        'iterative_steps': int(args.iterative_steps),
        'global_pruning': bool(args.global_pruning),
        'isomorphic': bool(args.isomorphic),
        'round_to': int(args.round_to),
        'max_pruning_ratio': float(args.max_pruning_ratio),
        'ignored_layer_count': int(len(ignored_layers)),
        'history': history,
        'save_model': str(save_model_path),
        'save_state_dict': str(save_state_dict_path),
        'save_tp_state': str(save_tp_state_path),
        'tp_state_saved': bool(tp_state_saved),
    }

    save_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[OK] pruning done.')
    print(f'[OK] model      : {save_model_path}')
    print(f'[OK] state_dict : {save_state_dict_path}')
    print(f'[OK] tp_state   : {save_tp_state_path} (saved={tp_state_saved})')
    print(f'[OK] summary    : {save_json_path}')


if __name__ == '__main__':
    main()
