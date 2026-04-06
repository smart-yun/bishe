# -*- coding: utf-8 -*-
"""Run A1/A2/A3 pruning experiments and summarize metrics.

Ablation plan:
- A1: stage4.ffn, 10%, finetune 5k
- A2: stage3.ffn, 10%, finetune 5k
- A3: stage3+4.ffn, 10%, finetune 10k

This script orchestrates calls to src/global_prune.py and aggregates outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class ExpSpec:
    exp_id: str
    target_stages: List[int]
    pruning_ratio: float
    finetune_iters: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run A1/A2/A3 pruning ablations for SegFormer-B0')
    parser.add_argument('--config', required=True, help='Path to mmseg config .py')
    parser.add_argument('--checkpoint', required=True, help='Path to baseline checkpoint .pth')
    parser.add_argument('--device', default='cuda:0', help='Device for global_prune.py')

    parser.add_argument('--conda-env', default='railseg', help='Conda env for running global_prune.py')
    parser.add_argument('--use-current-python', action='store_true', help='Use current python instead of conda run')

    parser.add_argument('--max-target-layers', type=int, default=0, help='0 means no truncation of candidate layers')
    parser.add_argument('--shape', type=int, nargs=2, default=[512, 512], help='Input shape H W')

    parser.add_argument('--finetune-lr', type=float, default=1e-5)
    parser.add_argument('--finetune-weight-decay', type=float, default=1e-2)
    parser.add_argument('--finetune-eval-interval', type=int, default=200)
    parser.add_argument('--finetune-log-interval', type=int, default=50)

    parser.add_argument('--skip-latency', action='store_true', help='Skip latency benchmark during pruning runs')
    parser.add_argument('--skip-miou', action='store_true', help='Skip mIoU evaluation during pruning runs')
    parser.add_argument('--skip-flops', action='store_true', help='Skip FLOPs/Params during pruning runs')

    parser.add_argument('--output-dir', default='exports', help='Directory to save per-exp and summary outputs')
    parser.add_argument('--checkpoints-dir', default='checkpoints', help='Directory to save per-exp checkpoints')

    parser.add_argument('--baseline-miou', type=float, default=59.75, help='Baseline best mIoU (%) for delta calc')
    parser.add_argument('--baseline-params-m', type=float, default=3.720051, help='Baseline Params (M)')
    parser.add_argument('--baseline-flops-g', type=float, default=7.956135936, help='Baseline FLOPs (G)')

    parser.add_argument('--dry-run', action='store_true', help='Only print commands without executing')
    return parser.parse_args()


def run_cmd(cmd: Sequence[str], dry_run: bool) -> int:
    print('[CMD]', ' '.join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def build_prefix(use_current_python: bool, conda_env: str) -> List[str]:
    if use_current_python:
        return [sys.executable]
    return ['conda', 'run', '-n', conda_env, 'python']


def extract_result_metrics(result_json: Dict) -> Dict:
    # Prefer post_finetune_eval, fallback to post_prune_eval.
    eval_block = result_json.get('post_finetune_eval') or result_json.get('post_prune_eval') or {}
    miou = eval_block.get('miou_metrics', {}).get('mIoU')
    params_m = eval_block.get('complexity', {}).get('params_m')
    flops_g = eval_block.get('complexity', {}).get('flops_g')

    ft = result_json.get('finetune', {}) if result_json.get('finetune_enabled') else {}

    return {
        'mIoU': miou,
        'params_m': params_m,
        'flops_g': flops_g,
        'best_miou_during_ft': ft.get('best_miou_during_ft'),
        'best_iter': ft.get('best_iter'),
    }


def to_pct_drop(new: float | None, base: float) -> float | None:
    if new is None:
        return None
    if base == 0:
        return None
    return (1.0 - new / base) * 100.0


def to_delta(new: float | None, base: float) -> float | None:
    if new is None:
        return None
    return new - base


def fmt(v: float | None, nd: int = 3) -> str:
    if v is None:
        return 'NA'
    return f'{v:.{nd}f}'


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoints_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = project_root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    prefix = build_prefix(args.use_current_python, args.conda_env)

    exps = [
        ExpSpec('A1_stage4_ffn_r10_ft5k', [4], 0.10, 5000),
        ExpSpec('A2_stage3_ffn_r10_ft5k', [3], 0.10, 5000),
        ExpSpec('A3_stage34_ffn_r10_ft10k', [3, 4], 0.10, 10000),
    ]

    summary_rows = []
    for exp in exps:
        exp_json = output_dir / f'{exp.exp_id}.json'
        save_best = ckpt_dir / f'{exp.exp_id}_best.pth'
        save_last = ckpt_dir / f'{exp.exp_id}_last.pth'
        pruned_ckpt = ckpt_dir / f'{exp.exp_id}_pruned.pth'

        cmd = [
            *prefix,
            str(project_root / 'src' / 'global_prune.py'),
            '--config', args.config,
            '--checkpoint', args.checkpoint,
            '--pruning-ratio', str(exp.pruning_ratio),
            '--target-stages', *[str(s) for s in exp.target_stages],
            '--max-target-layers', str(args.max_target_layers),
            '--shape', str(args.shape[0]), str(args.shape[1]),
            '--device', args.device,
            '--pruned-checkpoint', str(pruned_ckpt),
            '--output-json', str(exp_json),
            '--enable-finetune',
            '--finetune-iters', str(exp.finetune_iters),
            '--finetune-lr', str(args.finetune_lr),
            '--finetune-weight-decay', str(args.finetune_weight_decay),
            '--finetune-eval-interval', str(args.finetune_eval_interval),
            '--finetune-log-interval', str(args.finetune_log_interval),
            '--finetune-save-best', str(save_best),
            '--finetune-save-last', str(save_last),
        ]

        if args.skip_latency:
            cmd.append('--skip-latency')
        if args.skip_miou:
            cmd.append('--skip-miou')
        if args.skip_flops:
            cmd.append('--skip-flops')

        code = run_cmd(cmd, dry_run=args.dry_run)
        if code != 0:
            raise SystemExit(f'Experiment {exp.exp_id} failed with code {code}')

        if args.dry_run:
            continue

        result = json.loads(exp_json.read_text(encoding='utf-8'))
        m = extract_result_metrics(result)

        row = {
            'exp_id': exp.exp_id,
            'target_stages': exp.target_stages,
            'pruning_ratio': exp.pruning_ratio,
            'finetune_iters': exp.finetune_iters,
            'mIoU': m['mIoU'],
            'delta_mIoU_vs_baseline': to_delta(m['mIoU'], args.baseline_miou),
            'params_m': m['params_m'],
            'params_drop_pct_vs_baseline': to_pct_drop(m['params_m'], args.baseline_params_m),
            'flops_g': m['flops_g'],
            'flops_drop_pct_vs_baseline': to_pct_drop(m['flops_g'], args.baseline_flops_g),
            'best_miou_during_ft': m['best_miou_during_ft'],
            'best_iter': m['best_iter'],
            'result_json': str(exp_json),
        }
        summary_rows.append(row)

    if args.dry_run:
        print('[OK] dry-run done. No experiments executed.')
        return

    summary = {
        'baseline': {
            'mIoU': args.baseline_miou,
            'params_m': args.baseline_params_m,
            'flops_g': args.baseline_flops_g,
        },
        'experiments': summary_rows,
    }

    summary_json = output_dir / 'prune_a123_summary.json'
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    md_lines = [
        '| ExpID | Stages | Ratio | FT iters | mIoU | d_mIoU | Params(M) | Params_drop_pct | FLOPs(G) | FLOPs_drop_pct |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in summary_rows:
        md_lines.append(
            f"| {r['exp_id']} | {r['target_stages']} | {r['pruning_ratio']:.2f} | {r['finetune_iters']} | "
            f"{fmt(r['mIoU'],2)} | {fmt(r['delta_mIoU_vs_baseline'],2)} | {fmt(r['params_m'],3)} | "
            f"{fmt(r['params_drop_pct_vs_baseline'],2)} | {fmt(r['flops_g'],3)} | {fmt(r['flops_drop_pct_vs_baseline'],2)} |"
        )

    summary_md = output_dir / 'prune_a123_summary.md'
    summary_md.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')

    print(f'[OK] summary json: {summary_json}')
    print(f'[OK] summary md:   {summary_md}')


if __name__ == '__main__':
    main()
