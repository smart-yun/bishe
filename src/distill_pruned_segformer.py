# -*- coding: utf-8 -*-
"""Iterative structural pruning + MGD distillation for SegFormer (mmseg).

Implements a loop inspired by Algorithm 1:
1) compute per-step pruning ratio r = 1 - (1 - R)^(1/T)
2) iterative rounds: prune -> fine-tune with segmentation + MGD distillation -> evaluate
3) early stop when (mIoU_init - mIoU_current) > epsilon

Notes:
- Teacher keeps the original pre-trained structure and is frozen.
- Student is structurally pruned in-place each round.
- After each pruning step, optimizer is re-created (required because parameters change).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp

from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner

from mmseg.models import build_segmentor
from mmseg.utils import register_all_modules


def prepare_pythonpath(project_root: Path) -> None:
    src_dir = str(project_root / 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Iterative pruning + MGD distillation for SegFormer')
    p.add_argument('--config', required=True, help='mmseg config path')
    p.add_argument('--checkpoint', required=True, help='baseline checkpoint path')
    p.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    p.add_argument('--shape', type=int, nargs=2, default=[512, 512], help='example input H W')

    p.add_argument('--target-pruning-rate', type=float, default=0.12, help='global target pruning rate R in Algorithm 1')
    p.add_argument('--rounds', type=int, default=3, help='iterative rounds T in Algorithm 1')
    p.add_argument('--epsilon-miou-drop', type=float, default=2.0, help='early-stop threshold epsilon for mIoU drop')
    p.add_argument('--max-pruning-ratio', type=float, default=0.30, help='max pruning ratio per layer')
    p.add_argument('--round-to', type=int, default=40, help='channel round multiple to keep attention compatibility')

    p.add_argument('--round-iters', type=int, default=300, help='training iterations per round')
    p.add_argument('--lr', type=float, default=1e-5, help='optimizer lr for each round')
    p.add_argument('--weight-decay', type=float, default=0.01, help='optimizer weight decay')
    p.add_argument('--grad-clip', type=float, default=1.0, help='max grad norm, <=0 disables')
    p.add_argument('--log-interval', type=int, default=50, help='train logging interval')

    p.add_argument('--mgd-lambda', type=float, default=0.10, help='weight of MGD loss')
    p.add_argument('--mgd-mask-ratio', type=float, default=0.65, help='mask ratio in MGD')

    p.add_argument(
        '--ignore-keywords',
        nargs='*',
        default=['decode_head', 'auxiliary_head', 'backbone.layers.3'],
        help='name keywords for ignored layers during pruning',
    )

    p.add_argument('--work-dir', default='runs/rs19/distill_prune_loop', help='work dir')
    p.add_argument('--save-model', default='checkpoints/tp_distill_tiny_model.pth', help='best tiny model path')
    p.add_argument('--save-json', default='exports/tp_distill_prune_loop_summary.json', help='summary json path')
    return p.parse_args()


def maybe_to_device(model: nn.Module, device: str) -> str:
    if device.startswith('cuda') and torch.cuda.is_available():
        model.to(device)
        return device
    model.to('cpu')
    print('[WARN] CUDA unavailable, fallback to CPU')
    return 'cpu'


def load_model_from_cfg(config_path: str, checkpoint_path: str, device: str) -> tuple[nn.Module, Config]:
    cfg = Config.fromfile(config_path)
    model = build_segmentor(cfg.model)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if getattr(incompatible, 'missing_keys', None):
        print(f"[load] missing_keys: {len(incompatible.missing_keys)}")
    if getattr(incompatible, 'unexpected_keys', None):
        print(f"[load] unexpected_keys: {len(incompatible.unexpected_keys)}")

    model = revert_sync_batchnorm(model)
    model = model.to(device)
    return model, cfg


def build_runtime_cfg(cfg_path: str, work_dir: str) -> Config:
    cfg = Config.fromfile(cfg_path)
    cfg.launcher = 'none'
    cfg.work_dir = work_dir
    cfg.load_from = None
    cfg.resume = False
    cfg.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer',
    )
    return cfg


def seg_forward_fn(model: nn.Module, example_inputs: torch.Tensor):
    return model(example_inputs, data_samples=None, mode='tensor')


def to_percent(value: float) -> float:
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def evaluate_miou(cfg: Config, model: nn.Module) -> Dict[str, float]:
    runner = Runner.from_cfg(copy.deepcopy(cfg))
    runner.model = model
    metrics = runner.test()

    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            fv = float(v)
        except Exception:
            continue
        out[k] = to_percent(fv) if k == 'mIoU' else fv
    return out


def collect_ignored_layers(model: nn.Module, keywords: Sequence[str]) -> List[nn.Module]:
    ignored: List[nn.Module] = []
    for name, module in model.named_modules():
        if any(k in name for k in keywords):
            ignored.append(module)

    num_classes = None
    if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'num_classes'):
        num_classes = int(model.decode_head.num_classes)
    if num_classes is not None:
        for _, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == num_classes:
                ignored.append(module)
            elif isinstance(module, nn.Linear) and module.out_features == num_classes:
                ignored.append(module)

    uniq = {id(m): m for m in ignored}
    out = list(uniq.values())
    print(f'[info] ignored_layers: {len(out)}')
    return out


def validate_segformer_attention_dims(model: nn.Module) -> None:
    if not hasattr(model, 'backbone') or not hasattr(model.backbone, 'layers'):
        return
    violations: List[str] = []
    for stage_idx, stage in enumerate(model.backbone.layers, start=1):
        try:
            blocks = stage[1]
        except Exception:
            blocks = None
        if blocks is None or len(blocks) == 0:
            continue
        blk = blocks[0]

        num_heads = None
        if hasattr(blk, 'attn'):
            num_heads = getattr(blk.attn, 'num_heads', None)
            if num_heads is None and hasattr(blk.attn, 'attn'):
                num_heads = getattr(blk.attn.attn, 'num_heads', None)

        dim = None
        if hasattr(blk, 'norm1'):
            ns = getattr(blk.norm1, 'normalized_shape', None)
            if isinstance(ns, (tuple, list)) and len(ns) > 0:
                dim = int(ns[0])

        if num_heads is not None and dim is not None and dim % int(num_heads) != 0:
            violations.append(f'stage{stage_idx}: dim={dim}, heads={int(num_heads)}')

    if violations:
        raise RuntimeError('Invalid attention dims after pruning: ' + '; '.join(violations))


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


class MGDFeatureDistiller(nn.Module):
    """A lightweight MGD-style feature distiller for 2D feature maps."""

    def __init__(self, student_chs: Sequence[int], teacher_chs: Sequence[int], mask_ratio: float = 0.65):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.align = nn.ModuleList()
        self.gen = nn.ModuleList()

        for c_s, c_t in zip(student_chs, teacher_chs):
            self.align.append(nn.Conv2d(c_s, c_t, kernel_size=1, bias=False))
            self.gen.append(
                nn.Sequential(
                    nn.Conv2d(c_t, c_t, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_t, c_t, kernel_size=3, padding=1, bias=False),
                )
            )

    def forward(self, student_feats: Sequence[torch.Tensor], teacher_feats: Sequence[torch.Tensor]) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for i, (fs, ft) in enumerate(zip(student_feats, teacher_feats)):
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = torch.nn.functional.interpolate(fs, size=ft.shape[-2:], mode='bilinear', align_corners=False)

            x = self.align[i](fs)
            mask = (torch.rand_like(x[:, :1, :, :]) > self.mask_ratio).float()
            x_masked = x * mask
            pred = self.gen[i](x_masked)

            loss_i = torch.mean((pred - ft.detach()) ** 2)
            losses.append(loss_i)

        if not losses:
            return torch.tensor(0.0, device=student_feats[0].device)
        return sum(losses) / len(losses)


def infer_stage_channels(model: nn.Module, example_inputs: torch.Tensor) -> List[int]:
    model.eval()
    with torch.no_grad():
        feats = model.extract_feat(example_inputs)
    return [int(f.shape[1]) for f in feats]


def build_pruner(
    model: nn.Module,
    example_inputs: torch.Tensor,
    pruning_ratio: float,
    max_pruning_ratio: float,
    round_to: int,
    ignored_layers: List[nn.Module],
):
    importance = tp.importance.GroupMagnitudeImportance(p=2) if hasattr(tp.importance, 'GroupMagnitudeImportance') else tp.importance.MagnitudeImportance(p=2)

    pruner_cls = getattr(tp.pruner, 'MagnitudePruner', None)
    if pruner_cls is None:
        pruner_cls = getattr(tp.pruner, 'BasePruner', None)
    if pruner_cls is None:
        raise RuntimeError('No compatible TP high-level pruner found')

    kwargs: Dict[str, Any] = dict(
        importance=importance,
        pruning_ratio=float(pruning_ratio),
        iterative_steps=1,
        max_pruning_ratio=float(max_pruning_ratio),
        ignored_layers=ignored_layers,
        root_module_types=[nn.Conv2d, nn.Linear],
        forward_fn=seg_forward_fn,
    )
    if round_to > 0:
        kwargs['round_to'] = int(round_to)

    return pruner_cls(model, example_inputs, **kwargs)


def ensure_path(path_like: str, project_root: Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = project_root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    args = parse_args()
    if not (0.0 < args.target_pruning_rate < 1.0):
        raise ValueError('--target-pruning-rate must be in (0,1)')
    if args.rounds < 1:
        raise ValueError('--rounds must be >=1')
    if not (0.0 <= args.mgd_mask_ratio < 1.0):
        raise ValueError('--mgd-mask-ratio must be in [0,1)')

    project_root = Path(__file__).resolve().parents[1]
    prepare_pythonpath(project_root)
    register_all_modules(init_default_scope=True)

    runtime_cfg = build_runtime_cfg(args.config, args.work_dir)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
        print('[WARN] CUDA unavailable, fallback to CPU')

    # teacher / student initialization from the same baseline checkpoint
    teacher, _ = load_model_from_cfg(args.config, args.checkpoint, device)
    device = maybe_to_device(teacher, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student, _ = load_model_from_cfg(args.config, args.checkpoint, device)
    maybe_to_device(student, device)
    student.train()

    h, w = int(args.shape[0]), int(args.shape[1])
    example_inputs = torch.randn(1, 3, h, w, device=device)

    print('[eval] evaluating initial student (unpruned) mIoU...')
    init_metrics = evaluate_miou(runtime_cfg, student.eval())
    init_miou = float(init_metrics.get('mIoU', float('nan')))
    print(f'[eval] initial mIoU: {init_miou:.2f}')
    student.train()

    # per-step pruning ratio from target rate and rounds
    per_step_ratio = 1.0 - math.pow((1.0 - float(args.target_pruning_rate)), 1.0 / float(args.rounds))
    print(f'[algo] target R={args.target_pruning_rate:.4f}, T={args.rounds}, per-step r={per_step_ratio:.6f}')

    train_loader = Runner.build_dataloader(runtime_cfg.train_dataloader)
    train_iter = iter(train_loader)

    summary: Dict[str, Any] = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': device,
        'shape': [h, w],
        'target_pruning_rate': float(args.target_pruning_rate),
        'rounds': int(args.rounds),
        'per_step_pruning_ratio': float(per_step_ratio),
        'epsilon_miou_drop': float(args.epsilon_miou_drop),
        'initial_metrics': init_metrics,
        'records': [],
    }

    best_miou = -1.0
    best_round = -1
    best_model_path = ensure_path(args.save_model, project_root)

    early_stopped = False
    stop_reason = ''

    for round_idx in range(1, args.rounds + 1):
        print(f'\n================ Round {round_idx}/{args.rounds} ================')

        # 1) prune one step
        ignored_layers = collect_ignored_layers(student, args.ignore_keywords)
        pruner = build_pruner(
            model=student,
            example_inputs=example_inputs,
            pruning_ratio=per_step_ratio,
            max_pruning_ratio=float(args.max_pruning_ratio),
            round_to=int(args.round_to),
            ignored_layers=ignored_layers,
        )

        student.eval()
        before_params = param_count(student)
        pruner.step()
        validate_segformer_attention_dims(student)
        after_params = param_count(student)
        print(f'[prune] params: {before_params:,} -> {after_params:,} (delta={before_params-after_params:,})')

        # 2) build MGD distiller for current student shape
        student_chs = infer_stage_channels(student, example_inputs)
        teacher_chs = infer_stage_channels(teacher, example_inputs)
        distiller = MGDFeatureDistiller(student_chs, teacher_chs, mask_ratio=float(args.mgd_mask_ratio)).to(device)

        # 3) re-create optimizer AFTER pruning
        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(distiller.parameters()),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )

        # 4) fine-tune + distill this round
        student.train()
        distiller.train()
        round_seg_losses: List[float] = []
        round_mgd_losses: List[float] = []
        round_total_losses: List[float] = []

        for it in range(1, args.round_iters + 1):
            try:
                data_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                data_batch = next(train_iter)

            data = student.data_preprocessor(data_batch, True)
            inputs = data['inputs']
            data_samples = data['data_samples']

            # segmentation loss
            loss_dict = student(inputs, data_samples, mode='loss')
            if hasattr(student, 'parse_losses'):
                seg_loss, _ = student.parse_losses(loss_dict)
            else:
                ts = [v for v in loss_dict.values() if torch.is_tensor(v)]
                if not ts:
                    raise RuntimeError('No tensor losses found in student loss dict')
                seg_loss = sum(ts)

            # MGD distillation loss
            with torch.no_grad():
                teacher_feats = teacher.extract_feat(inputs)
            student_feats = student.extract_feat(inputs)
            mgd_loss = distiller(student_feats, teacher_feats)

            total_loss = seg_loss + float(args.mgd_lambda) * mgd_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(distiller.parameters()), max_norm=float(args.grad_clip))
            optimizer.step()

            round_seg_losses.append(float(seg_loss.detach().cpu()))
            round_mgd_losses.append(float(mgd_loss.detach().cpu()))
            round_total_losses.append(float(total_loss.detach().cpu()))

            if it == 1 or it % args.log_interval == 0 or it == args.round_iters:
                print(
                    f"[round {round_idx}] iter {it}/{args.round_iters} "
                    f"seg={round_seg_losses[-1]:.4f} mgd={round_mgd_losses[-1]:.4f} total={round_total_losses[-1]:.4f}"
                )

        # 5) evaluate and early-stop check
        student.eval()
        metrics = evaluate_miou(runtime_cfg, student)
        miou = float(metrics.get('mIoU', float('nan')))
        miou_drop = float(init_miou - miou)
        print(f'[eval] round {round_idx} mIoU={miou:.2f}, drop={miou_drop:.2f}')

        record = {
            'round': round_idx,
            'params_before': before_params,
            'params_after': after_params,
            'student_stage_channels': student_chs,
            'teacher_stage_channels': teacher_chs,
            'seg_loss_mean': statistics.mean(round_seg_losses) if round_seg_losses else None,
            'mgd_loss_mean': statistics.mean(round_mgd_losses) if round_mgd_losses else None,
            'total_loss_mean': statistics.mean(round_total_losses) if round_total_losses else None,
            'metrics': metrics,
            'miou_drop_from_init': miou_drop,
        }
        summary['records'].append(record)

        # save best snapshot
        if not math.isnan(miou) and miou > best_miou:
            best_miou = miou
            best_round = round_idx
            student.zero_grad(set_to_none=True)
            torch.save(student, best_model_path)
            print(f'[save] best model updated at round {round_idx}: {best_model_path}')

        # epsilon early stopping
        if (not math.isnan(miou_drop)) and miou_drop > float(args.epsilon_miou_drop):
            early_stopped = True
            stop_reason = (
                f'mIoU drop exceeded epsilon: drop={miou_drop:.2f} > epsilon={float(args.epsilon_miou_drop):.2f}'
            )
            print(f'[early-stop] {stop_reason}')
            break

    summary['best_miou'] = best_miou
    summary['best_round'] = best_round
    summary['best_model'] = str(best_model_path)
    summary['early_stopped'] = early_stopped
    summary['stop_reason'] = stop_reason

    save_json = ensure_path(args.save_json, project_root)
    save_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n================ Final Summary ================')
    print(f"initial mIoU: {init_miou:.2f}")
    print(f"best mIoU   : {best_miou:.2f} (round {best_round})")
    print(f"best model  : {best_model_path}")
    print(f"summary json: {save_json}")
    if early_stopped:
        print(f"early stop  : {stop_reason}")
    print('===============================================')


if __name__ == '__main__':
    main()
