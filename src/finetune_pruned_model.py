# -*- coding: utf-8 -*-
"""Short fine-tuning for a structurally pruned SegFormer full-model object.

This script loads a pruned model saved by torch.save(model, ...), performs
iter-based fine-tuning on the training split, optionally evaluates mIoU, and
saves the fine-tuned full model.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from mmengine.config import Config
from mmengine.runner import Runner

from mmseg.utils import register_all_modules


def prepare_pythonpath(project_root: Path) -> None:
    src_dir = str(project_root / 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


def _nearest_valid_num_heads(embed_dim: int, old_heads: int) -> int:
    """Pick a valid num_heads close to old_heads, preferring smaller change."""
    if embed_dim <= 0:
        return 1
    if old_heads > 0 and embed_dim % old_heads == 0:
        return old_heads

    for h in range(max(old_heads - 1, 1), 0, -1):
        if embed_dim % h == 0:
            return h

    for h in range(max(old_heads + 1, 2), embed_dim + 1):
        if embed_dim % h == 0:
            return h
    return 1


def sanitize_multihead_attention(model: torch.nn.Module) -> None:
    """Fix invalid MultiheadAttention settings created by structural pruning."""
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


def load_pruned_model(model_path: str, device: str) -> torch.nn.Module:
    try:
        model = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location='cpu')

    sanitize_multihead_attention(model)

    if device.startswith('cuda') and torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model.to('cpu')
    return model


def build_cfg_for_runtime(config_path: str, work_dir: str) -> Config:
    cfg = Config.fromfile(config_path)
    cfg.launcher = 'none'
    cfg.work_dir = work_dir
    cfg.load_from = None
    cfg.resume = False
    cfg.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer')
    return cfg


def evaluate_miou(cfg: Config, model: torch.nn.Module) -> Dict[str, float]:
    runner = Runner.from_cfg(copy.deepcopy(cfg))
    runner.model = model
    metrics = runner.test()

    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fine-tune pruned SegFormer model')
    p.add_argument('--config', required=True, help='mmseg config path')
    p.add_argument('--model-path', required=True, help='path to pruned full model (.pth)')
    p.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    p.add_argument('--work-dir', default='runs/rs19/pruned_finetune', help='work dir')

    p.add_argument('--iters', type=int, default=1000, help='fine-tuning iterations')
    p.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    p.add_argument('--weight-decay', type=float, default=0.01, help='weight decay')
    p.add_argument('--grad-clip', type=float, default=1.0, help='max grad norm, <=0 to disable')
    p.add_argument('--log-interval', type=int, default=50, help='log interval')

    p.add_argument('--eval-after', action='store_true', help='run full val mIoU after fine-tune')

    p.add_argument('--save-model', default='checkpoints/tp_local_iter_safe_model_ft.pth', help='save fine-tuned full model')
    p.add_argument('--save-json', default='exports/tp_local_iter_safe_finetune_summary.json', help='save training summary json')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    prepare_pythonpath(project_root)
    register_all_modules(init_default_scope=True)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('[WARN] CUDA unavailable, fallback to CPU.')
        device = 'cpu'

    os.makedirs(args.work_dir, exist_ok=True)
    cfg = build_cfg_for_runtime(args.config, args.work_dir)

    model = load_pruned_model(args.model_path, device)
    model.train()

    train_loader = Runner.build_dataloader(cfg.train_dataloader)
    train_iter = iter(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_meter = 0.0
    seen = 0
    trace = []

    for it in range(1, args.iters + 1):
        try:
            data_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data_batch = next(train_iter)

        data = model.data_preprocessor(data_batch, True)
        losses = model(data['inputs'], data['data_samples'], mode='loss')

        if hasattr(model, 'parse_losses'):
            loss, log_vars = model.parse_losses(losses)
            loss_value = float(loss.detach().cpu())
        else:
            ts = [v for v in losses.values() if torch.is_tensor(v)]
            if not ts:
                raise RuntimeError('No tensor loss found in loss dict.')
            loss = sum(ts)
            log_vars = {'loss': float(loss.detach().cpu())}
            loss_value = float(loss.detach().cpu())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        optimizer.step()

        loss_meter += loss_value
        seen += 1

        if it % args.log_interval == 0 or it == 1 or it == args.iters:
            avg_loss = loss_meter / max(seen, 1)
            msg = f"[ft] iter {it}/{args.iters} loss={loss_value:.4f} avg_loss={avg_loss:.4f}"
            if 'decode.loss_ce' in log_vars:
                msg += f" decode.loss_ce={float(log_vars['decode.loss_ce']):.4f}"
            print(msg)
            trace.append({'iter': it, 'loss': loss_value, 'avg_loss': avg_loss})

    model.eval()

    eval_metrics: Dict[str, Any] | None = None
    if args.eval_after:
        print('[ft] running full validation after fine-tune...')
        eval_metrics = evaluate_miou(cfg, model)
        print(f"[ft] eval metrics: {eval_metrics}")

    save_model = Path(args.save_model)
    if not save_model.is_absolute():
        save_model = project_root / save_model
    save_model.parent.mkdir(parents=True, exist_ok=True)
    model.zero_grad(set_to_none=True)
    torch.save(model, save_model)

    save_json = Path(args.save_json)
    if not save_json.is_absolute():
        save_json = project_root / save_json
    save_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'config': args.config,
        'source_model': args.model_path,
        'save_model': str(save_model),
        'device': device,
        'iters': args.iters,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'trace': trace,
        'eval_after': bool(args.eval_after),
        'eval_metrics': eval_metrics,
    }

    save_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] finetuned model: {save_model}')
    print(f'[OK] summary json   : {save_json}')


if __name__ == '__main__':
    main()
