# -*- coding: utf-8 -*-
"""
SegFormer + Torch-Pruning (High-level / Iterative / Global / Isomorphic)

========================= Runtime Argument Templates =========================

# 0) Recommended: set project PYTHONPATH first
# export PYTHONPATH=/home/lcy/Projects/bishe/src:${PYTHONPATH}

# 1) Local + Iterative (safe starting point)
# python src/torch_prune_segformer.py \
#   --config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py \
#   --checkpoint runs/best_mIoU_iter_v1.pth \
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

python src/distill_pruned_segformer.py 
--config configs/railsem19/segformer_b0_rs19_512x512_100ep_rtx4090_from_best.py 
--checkpoint runs/best_mIoU_iter_v1.pth 
--device cuda:0 --shape 512 512 
--target-pruning-rate 0.06 
--rounds 2 
--epsilon-miou-drop 2.5 
--max-pruning-ratio 0.20 
--round-to 40 
--round-iters 800 
--lr 8e-6 
--weight-decay 0.01 
--grad-clip 1.0 
--log-interval 100 
--mgd-lambda 0.05 
--mgd-mask-ratio 0.5 
--ignore-keywords decode_head auxiliary_head backbone.layers.2 backbone.layers.3 
--work-dir runs/rs19/distill_prune_stable_r2 
--save-model checkpoints/tp_distill_stable_r2_best.pth 
--save-json exports/tp_distill_stable_r2_summary.json

"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

try:
    import importlib.metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore

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
    parser.add_argument('--save-model', default='checkpoints/segformer_tp_pruned_model.pth', help='Save full model object path (only used with --save-model-object)')
    parser.add_argument('--save-model-object', action='store_true', help='Save full model object (not recommended for long-term portability)')
    parser.add_argument('--save-tp-state', default='checkpoints/segformer_tp_pruned_tpstate.pth', help='Save tp.state_dict (if available)')
    parser.add_argument('--save-state-dict', default='checkpoints/segformer_tp_pruned_state_dict.pth', help='Save raw model.state_dict')
    parser.add_argument('--save-json', default='exports/segformer_tp_prune_summary.json', help='Summary JSON path')

    parser.add_argument('--count-macs', action='store_true', help='Try counting MACs via tp.utils if supported')
    parser.add_argument(
        '--fail-fast-attn-violation',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Abort pruning and rollback current step when attention dim/head divisibility violations are detected',
    )

    # Conservative preset for SegFormer-like backbones
    parser.add_argument('--segformer-safe', action='store_true', help='Enable a conservative pruning profile for SegFormer')
    parser.add_argument(
        '--safe-target',
        choices=['accuracy', 'latency'],
        default='accuracy',
        help='Safe profile target: accuracy or latency',
    )
    parser.add_argument('--safe-pruning-ratio', type=float, default=0.20, help='Upper bound for pruning-ratio in safe mode')
    parser.add_argument('--safe-max-pruning-ratio', type=float, default=0.35, help='Upper bound for max-pruning-ratio in safe mode')
    parser.add_argument('--safe-round-to', type=int, default=40, help='Compatible round-to value in safe mode')

    parser.add_argument(
        '--loss-fn',
        choices=['ce'],
        default='ce',
        help='Loss function used by Taylor importance gradient and optional finetune',
    )
    parser.add_argument('--aux-loss-weight', type=float, default=0.4, help='Auxiliary logits loss weight when output has aux head')
    parser.add_argument('--dummy-ignore-index', type=int, default=255, help='Ignore index used in dummy target loss')

    parser.add_argument('--finetune', action='store_true', help='Run a lightweight post-prune finetune loop (dummy supervised recovery)')
    parser.add_argument('--finetune-iters', type=int, default=200, help='Post-prune finetune iterations')
    parser.add_argument('--finetune-lr', type=float, default=1e-5, help='Post-prune finetune learning rate')
    parser.add_argument('--finetune-weight-decay', type=float, default=0.01, help='Post-prune finetune weight decay')
    parser.add_argument('--finetune-batch-size', type=int, default=2, help='Post-prune finetune dummy batch size')
    parser.add_argument('--finetune-grad-clip', type=float, default=1.0, help='Post-prune finetune grad clipping (<=0 disables)')
    parser.add_argument('--finetune-log-interval', type=int, default=50, help='Post-prune finetune log interval')
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


def extract_logits_from_output(output: Any) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    if torch.is_tensor(output):
        return output, []

    if isinstance(output, (list, tuple)):
        tensors = [x for x in output if torch.is_tensor(x)]
        if not tensors:
            raise RuntimeError('Model output list/tuple has no tensor logits.')
        return tensors[0], tensors[1:]

    if isinstance(output, dict):
        if 'logits' in output and torch.is_tensor(output['logits']):
            aux = output.get('aux_logits', [])
            if torch.is_tensor(aux):
                aux = [aux]
            elif isinstance(aux, (list, tuple)):
                aux = [x for x in aux if torch.is_tensor(x)]
            else:
                aux = []
            return output['logits'], aux

        ts = [x for x in output.values() if torch.is_tensor(x)]
        if not ts:
            raise RuntimeError('Model output dict has no tensor logits.')
        return ts[0], ts[1:]

    raise RuntimeError(f'Unsupported output type: {type(output)}')


def build_dummy_seg_target(
    batch_size: int,
    out_h: int,
    out_w: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(0, max(2, int(num_classes)), (batch_size, out_h, out_w), device=device, dtype=torch.long)


def compute_seg_loss(
    logits: torch.Tensor,
    aux_logits: List[torch.Tensor],
    target: torch.Tensor,
    loss_fn: str,
    aux_loss_weight: float,
    ignore_index: int,
) -> torch.Tensor:
    if loss_fn != 'ce':
        raise RuntimeError(f'Unsupported --loss-fn={loss_fn}.')

    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
    loss = F.cross_entropy(logits, target, ignore_index=int(ignore_index))

    for aux in aux_logits:
        if aux.shape[-2:] != target.shape[-2:]:
            aux = F.interpolate(aux, size=target.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + float(aux_loss_weight) * F.cross_entropy(aux, target, ignore_index=int(ignore_index))
    return loss


def seg_forward_fn(
    model: nn.Module,
    example_inputs: torch.Tensor,
    mode: str = 'tensor',
    loss_fn: str = 'ce',
    target: torch.Tensor | None = None,
    aux_loss_weight: float = 0.4,
    ignore_index: int = 255,
) -> Any:
    """Forward function for mmseg segmentor in tensor/loss mode."""
    output = model(example_inputs, data_samples=None, mode='tensor')
    if mode == 'tensor':
        return output
    if mode != 'loss':
        raise RuntimeError(f'Unsupported forward mode: {mode}')

    logits, aux_logits = extract_logits_from_output(output)
    if target is None:
        target = build_dummy_seg_target(
            batch_size=logits.shape[0],
            out_h=logits.shape[2],
            out_w=logits.shape[3],
            num_classes=logits.shape[1],
            device=logits.device,
        )
    return compute_seg_loss(logits, aux_logits, target, loss_fn, aux_loss_weight, ignore_index)


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


def collect_ignored_layers(model: nn.Module, keywords: List[str]) -> Tuple[List[nn.Module], List[str]]:
    ignored: List[Tuple[str, nn.Module]] = []

    # 1) Name-based protection
    for name, module in model.named_modules():
        if any(k in name for k in keywords):
            ignored.append((name, module))

    # 2) Num-class head protection (conv/linear with out_channels/out_features == num_classes)
    num_classes = None
    if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'num_classes'):
        num_classes = int(model.decode_head.num_classes)

    if num_classes is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == num_classes:
                ignored.append((name, module))
            elif isinstance(module, nn.Linear) and module.out_features == num_classes:
                ignored.append((name, module))

    # deduplicate by object id
    uniq: Dict[int, Tuple[str, nn.Module]] = {}
    for name, module in ignored:
        if id(module) not in uniq:
            uniq[id(module)] = (name, module)
    names = [item[0] for item in uniq.values()]
    out = [item[1] for item in uniq.values()]
    print(f'[info] ignored_layers: {len(out)}')
    return out, names


def build_segformer_stage_ratio_dict(
    model: nn.Module,
    base_ratio: float,
    multipliers: List[float] | None = None,
) -> Dict[nn.Module, float]:
    """Conservative stage-wise ratio settings for SegFormer-like backbones."""
    ratio_dict: Dict[nn.Module, float] = {}
    if not hasattr(model, 'backbone') or not hasattr(model.backbone, 'layers'):
        return ratio_dict

    layers = list(model.backbone.layers)
    if multipliers is None:
        multipliers = [0.35, 0.55, 0.80, 1.00]
    for layer, mul in zip(layers, multipliers):
        ratio = max(0.01, min(base_ratio * mul, base_ratio))
        ratio_dict[layer] = float(ratio)
    return ratio_dict


def validate_segformer_attention_dims(model: nn.Module) -> List[str]:
    """Check stage embed dim divisibility by number of heads and return violations."""
    if not hasattr(model, 'backbone') or not hasattr(model.backbone, 'layers'):
        return []

    violations: List[str] = []
    for stage_idx, stage in enumerate(model.backbone.layers, start=1):
        try:
            blocks = stage[1]
        except Exception:
            blocks = None
        if blocks is None:
            continue

        for block_idx, blk in enumerate(blocks):
            num_heads = None
            if hasattr(blk, 'attn'):
                num_heads = getattr(blk.attn, 'num_heads', None)
                if num_heads is None and hasattr(blk.attn, 'attn'):
                    num_heads = getattr(blk.attn.attn, 'num_heads', None)

            dim = None
            if hasattr(blk, 'norm1'):
                ns = getattr(blk.norm1, 'normalized_shape', None)
                if isinstance(ns, (list, tuple)) and len(ns) > 0:
                    dim = int(ns[0])

            if num_heads is not None and dim is not None and dim % int(num_heads) != 0:
                violations.append(f'stage{stage_idx}.block{block_idx}: dim={dim}, heads={int(num_heads)}')

    if violations:
        joined = '; '.join(violations[:8])
        suffix = ' ...' if len(violations) > 8 else ''
        print(
            '[WARN] Attention dimension divisibility violations found: '
            f'{joined}{suffix}. '
            'Try increasing/adjusting --round-to (B0 commonly uses 8 or 40) or reducing pruning strength.'
        )
    return violations


def get_tp_version() -> str:
    v = getattr(tp, '__version__', None)
    if isinstance(v, str) and v.strip():
        return v
    try:
        return str(importlib_metadata.version('torch-pruning'))
    except Exception:
        return 'unknown'


def version_at_least(version: str, minimum: str) -> bool:
    def parse(v: str) -> List[int]:
        nums = re.findall(r'\d+', v)
        return [int(x) for x in nums[:3]] if nums else [0, 0, 0]

    a = parse(version)
    b = parse(minimum)
    while len(a) < len(b):
        a.append(0)
    while len(b) < len(a):
        b.append(0)
    return a >= b


def format_pruner_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == 'ignored_layers' and isinstance(v, list):
            out[k] = f'[{len(v)} modules]'
        elif k == 'pruning_ratio_dict' and isinstance(v, dict):
            out[k] = f'[{len(v)} module-ratios]'
        elif k == 'root_module_types' and isinstance(v, list):
            out[k] = [t.__name__ if hasattr(t, '__name__') else str(t) for t in v]
        else:
            out[k] = v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)
    return out


def build_pruner(
    model: nn.Module,
    example_inputs: torch.Tensor,
    importance,
    args: argparse.Namespace,
    ignored_layers: List[nn.Module],
    pruning_ratio_dict: Dict[nn.Module, float] | None = None,
    conv_only_root: bool = False,
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
    if 'pruning_ratio_dict' in params and pruning_ratio_dict:
        kwargs['pruning_ratio_dict'] = pruning_ratio_dict
    if 'max_pruning_ratio' in params:
        kwargs['max_pruning_ratio'] = float(args.max_pruning_ratio)
    if 'ignored_layers' in params:
        kwargs['ignored_layers'] = ignored_layers
    if 'round_to' in params and args.round_to > 0:
        kwargs['round_to'] = int(args.round_to)
    if 'root_module_types' in params:
        kwargs['root_module_types'] = [nn.Conv2d] if conv_only_root else [nn.Conv2d, nn.Linear]
    if 'forward_fn' in params:
        kwargs['forward_fn'] = seg_forward_fn

    unsupported: List[str] = []
    if 'iterative_steps' not in params:
        unsupported.append('iterative_steps')
    if 'global_pruning' not in params and args.global_pruning:
        unsupported.append('global_pruning')
    if 'isomorphic' not in params and args.isomorphic:
        unsupported.append('isomorphic')
    if ('pruning_ratio' not in params and 'ch_sparsity' not in params):
        unsupported.append('pruning_ratio/ch_sparsity')
    if 'max_pruning_ratio' not in params:
        unsupported.append('max_pruning_ratio')
    if 'round_to' not in params and args.round_to > 0:
        unsupported.append('round_to')
    if unsupported:
        print(f"[WARN] pruner ctor does not support: {', '.join(unsupported)}")

    formatted_kwargs = format_pruner_kwargs(kwargs)
    print(f"[info] using pruner: {pruner_cls.__name__}")
    print(f"[info] pruner kwargs: {json.dumps(formatted_kwargs, ensure_ascii=False)}")
    return pruner_cls(model, example_inputs, **kwargs), formatted_kwargs


def run_post_prune_finetune(
    model: nn.Module,
    args: argparse.Namespace,
    device: str,
    h: int,
    w: int,
) -> List[float]:
    if not args.finetune:
        return []

    print('[finetune] start lightweight post-prune finetune loop (dummy supervised).')
    print('[finetune][WARN] This is a recovery placeholder. For real accuracy recovery, run dataset-based finetune.')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.finetune_lr),
        weight_decay=float(args.finetune_weight_decay),
    )
    losses: List[float] = []

    model.train()
    for it in range(1, int(args.finetune_iters) + 1):
        inputs = torch.randn(int(args.finetune_batch_size), 3, h, w, device=device)
        output = seg_forward_fn(model, inputs, mode='tensor')
        logits, aux_logits = extract_logits_from_output(output)
        target = build_dummy_seg_target(
            batch_size=logits.shape[0],
            out_h=logits.shape[2],
            out_w=logits.shape[3],
            num_classes=logits.shape[1],
            device=logits.device,
        )
        loss = compute_seg_loss(
            logits=logits,
            aux_logits=aux_logits,
            target=target,
            loss_fn=args.loss_fn,
            aux_loss_weight=float(args.aux_loss_weight),
            ignore_index=int(args.dummy_ignore_index),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.finetune_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.finetune_grad_clip))
        optimizer.step()

        loss_val = float(loss.detach().item())
        losses.append(loss_val)
        if it == 1 or it % max(1, int(args.finetune_log_interval)) == 0 or it == int(args.finetune_iters):
            print(f'[finetune] iter={it}/{args.finetune_iters}, loss={loss_val:.6f}')

    model.eval()
    print('[finetune] done.')
    return losses


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
        print('[TIP] Consider using fvcore or ptflops for FLOPs/MACs counting.')
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
        print('[TIP] You can switch to fvcore/ptflops to reduce torch_pruning version compatibility issues.')
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

    tp_version = get_tp_version()
    print(f'[env] torch_pruning version: {tp_version}')
    if tp_version != 'unknown' and not version_at_least(tp_version, '1.3.0'):
        print('[WARN] torch_pruning>=1.3.0 is recommended to reduce constructor-compatibility risks.')

    model, _ = load_model_from_cfg(args.config, args.checkpoint)
    device = maybe_to_device(model, args.device)
    h, w = int(args.shape[0]), int(args.shape[1])
    example_inputs = torch.randn(1, 3, h, w, device=device)

    # Optional conservative profile for SegFormer
    ignore_keywords = list(args.ignore_keywords)
    pruning_ratio_dict: Dict[nn.Module, float] | None = None
    safe_profile_applied = False
    original_pruning_ratio = float(args.pruning_ratio)
    original_max_pruning_ratio = float(args.max_pruning_ratio)
    original_round_to = int(args.round_to)

    if args.segformer_safe:
        safe_profile_applied = True
        safe_stage_multipliers = [0.35, 0.55, 0.80, 1.00]

        if args.safe_target == 'latency':
            # Latency-oriented profile:
            # - keep stage3 (index=2, head=5) at a moderate pruning ratio to avoid bottleneck
            # - prune earlier/later stages with tensor-core friendly round_to
            ignore_keywords.extend(['decode_head', 'auxiliary_head', 'backbone.layers.2'])
            safe_stage_multipliers = [0.25, 0.70, 0.30, 1.00]
            if args.safe_round_to == 40:
                print('[safe-latency] override safe_round_to: 40 -> 8')
                args.safe_round_to = 8
            if args.safe_max_pruning_ratio > 0.30:
                print(f'[safe-latency] cap safe_max_pruning_ratio: {args.safe_max_pruning_ratio:.4f} -> 0.3000')
                args.safe_max_pruning_ratio = 0.30
        else:
            # Accuracy-oriented profile (existing behavior)
            ignore_keywords.extend(['decode_head', 'auxiliary_head', 'backbone.layers.3'])

        ignore_keywords = list(dict.fromkeys(ignore_keywords))

        if args.pruning_ratio > args.safe_pruning_ratio:
            print(f'[safe] cap pruning_ratio: {args.pruning_ratio:.4f} -> {args.safe_pruning_ratio:.4f}')
            args.pruning_ratio = float(args.safe_pruning_ratio)

        if args.max_pruning_ratio > args.safe_max_pruning_ratio:
            print(f'[safe] cap max_pruning_ratio: {args.max_pruning_ratio:.4f} -> {args.safe_max_pruning_ratio:.4f}')
            args.max_pruning_ratio = float(args.safe_max_pruning_ratio)

        if args.safe_round_to > 0 and args.round_to != args.safe_round_to:
            print(f'[safe] set round_to: {args.round_to} -> {args.safe_round_to}')
            args.round_to = int(args.safe_round_to)

        pruning_ratio_dict = build_segformer_stage_ratio_dict(
            model,
            float(args.pruning_ratio),
            multipliers=safe_stage_multipliers,
        )
        print(f'[safe] stage-wise pruning ratio dict enabled. target={args.safe_target}')

    ignored_layers, ignored_layer_names = collect_ignored_layers(model, ignore_keywords)
    importance = choose_importance(args.importance)
    pruner, pruner_kwargs = build_pruner(
        model,
        example_inputs,
        importance,
        args,
        ignored_layers,
        pruning_ratio_dict=pruning_ratio_dict,
        conv_only_root=bool(args.segformer_safe),
    )

    mode_desc = 'local'
    if args.global_pruning and args.isomorphic:
        mode_desc = 'global+isomorphic'
    elif args.global_pruning:
        mode_desc = 'global'

    history: List[Dict[str, Any]] = []
    attention_warnings: List[Dict[str, Any]] = []
    aborted_due_to_attention = False
    aborted_step: int | None = None
    base_params = param_count(model)
    base_macs = try_count_macs(model, example_inputs, enabled=args.count_macs)
    history.append({'step': 0, 'params': base_params, 'macs': base_macs})
    print(f'[step 0] params={base_params:,}' + (f', macs={base_macs:,}' if base_macs is not None else ''))

    for step in range(1, args.iterative_steps + 1):
        model.train()
        if args.importance == 'taylor':
            model.zero_grad(set_to_none=True)
            out = seg_forward_fn(model, example_inputs, mode='tensor')
            logits, aux_logits = extract_logits_from_output(out)
            dummy_target = build_dummy_seg_target(
                batch_size=logits.shape[0],
                out_h=logits.shape[2],
                out_w=logits.shape[3],
                num_classes=logits.shape[1],
                device=logits.device,
            )
            taylor_loss = compute_seg_loss(
                logits=logits,
                aux_logits=aux_logits,
                target=dummy_target,
                loss_fn=args.loss_fn,
                aux_loss_weight=float(args.aux_loss_weight),
                ignore_index=int(args.dummy_ignore_index),
            )
            taylor_loss.backward()
            print(f'[step {step}] taylor_loss={float(taylor_loss.detach().item()):.6f} (loss_fn={args.loss_fn})')

        model.eval()
        model_backup = copy.deepcopy(model) if args.fail_fast_attn_violation else None
        pruner.step()
        violations = validate_segformer_attention_dims(model)
        if violations:
            attention_warnings.append({'step': step, 'violations': violations})
            if args.fail_fast_attn_violation and model_backup is not None:
                print(
                    f'[FAIL-FAST] attention violations at step {step}; '
                    'rollback current step and stop pruning.'
                )
                model = model_backup
                aborted_due_to_attention = True
                aborted_step = step

                cur_params = param_count(model)
                cur_macs = try_count_macs(model, example_inputs, enabled=args.count_macs)
                history.append({
                    'step': step,
                    'params': cur_params,
                    'macs': cur_macs,
                    'rolled_back': True,
                    'aborted': True,
                })
                break

        cur_params = param_count(model)
        cur_macs = try_count_macs(model, example_inputs, enabled=args.count_macs)
        history.append({'step': step, 'params': cur_params, 'macs': cur_macs})
        print(
            f'[step {step}] params={cur_params:,}, '
            f'pruned={(base_params - cur_params):,}'
            + (f', macs={cur_macs:,}' if cur_macs is not None else '')
        )

    finetune_losses = run_post_prune_finetune(model, args, device, h, w)

    # Save artifacts
    save_model_path = ensure_parent(args.save_model, project_root) if args.save_model_object else None
    save_tp_state_path = ensure_parent(args.save_tp_state, project_root)
    save_state_dict_path = ensure_parent(args.save_state_dict, project_root)
    save_json_path = ensure_parent(args.save_json, project_root)

    model.zero_grad(set_to_none=True)
    if args.save_model_object and save_model_path is not None:
        torch.save(model, save_model_path)
    else:
        print('[info] skip full model object saving (recommended). Use --save-model-object to enable.')
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
        'ignored_layer_names': ignored_layer_names,
        'segformer_safe': bool(args.segformer_safe),
        'safe_profile_applied': bool(safe_profile_applied),
        'safe_target': str(args.safe_target),
        'original_pruning_ratio': original_pruning_ratio,
        'original_max_pruning_ratio': original_max_pruning_ratio,
        'original_round_to': original_round_to,
        'effective_pruning_ratio': float(args.pruning_ratio),
        'effective_max_pruning_ratio': float(args.max_pruning_ratio),
        'effective_round_to': int(args.round_to),
        'loss_fn': str(args.loss_fn),
        'aux_loss_weight': float(args.aux_loss_weight),
        'dummy_ignore_index': int(args.dummy_ignore_index),
        'torch_pruning_version': tp_version,
        'pruner_kwargs': pruner_kwargs,
        'attention_dim_warnings': attention_warnings,
        'fail_fast_attn_violation': bool(args.fail_fast_attn_violation),
        'aborted_due_to_attention': bool(aborted_due_to_attention),
        'aborted_step': int(aborted_step) if aborted_step is not None else None,
        'finetune_enabled': bool(args.finetune),
        'finetune_iters': int(args.finetune_iters),
        'finetune_losses': finetune_losses,
        'save_model_object': bool(args.save_model_object),
        'history': history,
        'save_model': str(save_model_path) if save_model_path is not None else None,
        'save_state_dict': str(save_state_dict_path),
        'save_tp_state': str(save_tp_state_path),
        'tp_state_saved': bool(tp_state_saved),
        'notes': {
            'macs_counter_tip': 'Recommend fvcore/ptflops when tp.utils.count_ops_and_params is unavailable or unstable.',
            'save_tip': 'Prefer state_dict for portability. Full model object can break if class import path changes.',
            'finetune_tip': 'Dummy finetune is only a placeholder; run real dataset-based finetune for meaningful mIoU recovery.',
        },
    }

    save_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('[OK] pruning done.')
    if save_model_path is not None:
        print(f'[OK] model      : {save_model_path}')
    else:
        print('[OK] model      : skipped (use --save-model-object to enable)')
    print(f'[OK] state_dict : {save_state_dict_path}')
    print(f'[OK] tp_state   : {save_tp_state_path} (saved={tp_state_saved})')
    print(f'[OK] summary    : {save_json_path}')


if __name__ == '__main__':
    main()
