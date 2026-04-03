# -*- coding: utf-8 -*-
"""Plot mIoU-vs-step curve from MMEngine line-json log.

Usage:
python src/plot_miou_curve.py \
  --log-json runs/rs19/segformer_b0_512x512_40000it/20260225_170857/vis_data/20260225_170857.json \
  --out-png results/curves/miou_curve_40k.png \
  --out-summary-json results/curves/miou_curve_40k_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def moving_avg(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < window:
        return values
    out: List[float] = []
    csum = 0.0
    q: List[float] = []
    for v in values:
        q.append(v)
        csum += v
        if len(q) > window:
            csum -= q.pop(0)
        out.append(csum / len(q))
    return out


def parse_mmengine_json(log_json: Path) -> Tuple[List[int], List[float], List[int], List[float]]:
    """Return (train_steps, train_losses, val_steps, val_mious)."""
    train_steps: List[int] = []
    train_losses: List[float] = []

    # Use dict for val so same step only keeps the latest value.
    val_map: Dict[int, float] = {}

    with log_json.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            step = obj.get('step', obj.get('iter', None))
            if step is None:
                continue

            if 'loss' in obj:
                train_steps.append(int(step))
                train_losses.append(float(obj['loss']))

            if 'mIoU' in obj:
                val_map[int(step)] = float(obj['mIoU'])

    val_steps = sorted(val_map.keys())
    val_mious = [val_map[s] for s in val_steps]
    return train_steps, train_losses, val_steps, val_mious


def nearest_prev_train_loss(train_steps: List[int], train_losses: List[float], step: int) -> float | None:
    best_idx = -1
    for i, s in enumerate(train_steps):
        if s <= step:
            best_idx = i
        else:
            break
    if best_idx < 0:
        return None
    return train_losses[best_idx]


def detect_overfit(
    train_steps: List[int],
    train_losses: List[float],
    val_steps: List[int],
    val_mious: List[float],
    miou_drop_threshold: float,
    loss_drop_threshold: float,
) -> Dict[str, float | bool | None]:
    if not val_steps:
        return {
            'overfit_warning': False,
            'best_step': None,
            'best_miou': None,
            'last_step': None,
            'last_miou': None,
            'miou_drop_after_best': None,
            'train_loss_at_best': None,
            'train_loss_last': None,
            'train_loss_drop_after_best': None,
        }

    best_idx = max(range(len(val_mious)), key=lambda i: val_mious[i])
    best_step = val_steps[best_idx]
    best_miou = val_mious[best_idx]

    last_step = val_steps[-1]
    last_miou = val_mious[-1]
    miou_drop = best_miou - last_miou

    loss_best = nearest_prev_train_loss(train_steps, train_losses, best_step)
    loss_last = nearest_prev_train_loss(train_steps, train_losses, last_step)

    if loss_best is None or loss_last is None:
        loss_drop = None
        overfit = False
    else:
        loss_drop = loss_best - loss_last
        overfit = (miou_drop >= miou_drop_threshold) and (loss_drop >= loss_drop_threshold)

    return {
        'overfit_warning': overfit,
        'best_step': best_step,
        'best_miou': round(best_miou, 4),
        'last_step': last_step,
        'last_miou': round(last_miou, 4),
        'miou_drop_after_best': round(miou_drop, 4),
        'train_loss_at_best': None if loss_best is None else round(loss_best, 6),
        'train_loss_last': None if loss_last is None else round(loss_last, 6),
        'train_loss_drop_after_best': None if loss_drop is None else round(loss_drop, 6),
    }


def detect_overfit_critical_step(
    train_steps: List[int],
    train_losses: List[float],
    val_steps: List[int],
    val_mious: List[float],
    miou_drop_threshold: float,
    loss_drop_threshold: float,
    consecutive_val_points: int,
) -> Dict[str, int | float | None | bool]:
    """Find first overfitting critical step with consecutive validation points.

    Criterion at each val point (after running-best appears):
      1) mIoU has dropped from running-best by >= miou_drop_threshold
      2) train loss has dropped from loss at running-best by >= loss_drop_threshold
    The first step satisfying the criterion for N consecutive val points is
    treated as overfitting critical step.
    """
    if not val_steps:
        return {
            'critical_found': False,
            'critical_step': None,
            'critical_best_step': None,
            'critical_best_miou': None,
            'critical_miou_drop': None,
            'critical_train_loss_drop': None,
            'consecutive_val_points': consecutive_val_points,
        }

    running_best_miou = float('-inf')
    running_best_step = None
    running_best_loss = None

    streak = 0
    streak_last_step = None
    critical_payload = None

    for step, miou in zip(val_steps, val_mious):
        loss_now = nearest_prev_train_loss(train_steps, train_losses, step)

        if miou > running_best_miou:
            running_best_miou = miou
            running_best_step = step
            running_best_loss = loss_now
            streak = 0
            streak_last_step = None
            continue

        if running_best_step is None:
            continue

        miou_drop = running_best_miou - miou
        if running_best_loss is None or loss_now is None:
            loss_drop = 0.0
        else:
            loss_drop = running_best_loss - loss_now

        if miou_drop >= miou_drop_threshold and loss_drop >= loss_drop_threshold:
            # consecutive by validation index spacing, not absolute iter distance
            if streak_last_step is None:
                streak = 1
            else:
                streak += 1
            streak_last_step = step

            if streak >= max(1, consecutive_val_points):
                critical_payload = {
                    'critical_found': True,
                    'critical_step': step,
                    'critical_best_step': running_best_step,
                    'critical_best_miou': round(running_best_miou, 4),
                    'critical_miou_drop': round(miou_drop, 4),
                    'critical_train_loss_drop': round(loss_drop, 6),
                    'consecutive_val_points': consecutive_val_points,
                }
                break
        else:
            streak = 0
            streak_last_step = None

    if critical_payload is not None:
        return critical_payload

    return {
        'critical_found': False,
        'critical_step': None,
        'critical_best_step': running_best_step,
        'critical_best_miou': None if running_best_miou == float('-inf') else round(running_best_miou, 4),
        'critical_miou_drop': None,
        'critical_train_loss_drop': None,
        'consecutive_val_points': consecutive_val_points,
    }


def plot_curve(
    train_steps: List[int],
    train_losses: List[float],
    val_steps: List[int],
    val_mious: List[float],
    out_png: Path,
    smooth_window: int,
    critical_step: int | None = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 5.6), dpi=150)

    # mIoU on left y-axis
    ax1.plot(val_steps, val_mious, marker='o', linewidth=1.8, label='val mIoU')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('mIoU (%)')
    ax1.grid(True, linestyle='--', alpha=0.35)

    if val_mious:
        best_idx = max(range(len(val_mious)), key=lambda i: val_mious[i])
        bx, by = val_steps[best_idx], val_mious[best_idx]
        ax1.scatter([bx], [by], s=50)
        ax1.annotate(f'best={by:.2f}@{bx}', (bx, by), xytext=(6, 6), textcoords='offset points')

    if critical_step is not None:
        ax1.axvline(critical_step, linestyle='--', linewidth=1.4, alpha=0.8, label=f'overfit critical@{critical_step}')

    # loss on right y-axis
    ax2 = ax1.twinx()
    if train_losses:
        smooth_loss = moving_avg(train_losses, smooth_window)
        ax2.plot(train_steps, smooth_loss, linewidth=1.0, alpha=0.65, label=f'train loss (ma{smooth_window})')
    ax2.set_ylabel('Train Loss')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('mIoU / Train Loss vs Iteration')
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot mIoU curve from MMEngine log json')
    parser.add_argument('--log-json', required=True, help='Path to line-by-line JSON log')
    parser.add_argument('--out-png', required=True, help='Output curve image path')
    parser.add_argument('--out-summary-json', required=True, help='Output analysis json path')
    parser.add_argument('--smooth-window', type=int, default=50, help='Moving-average window for train loss')
    parser.add_argument('--miou-drop-threshold', type=float, default=0.5, help='mIoU drop threshold for overfit warning')
    parser.add_argument('--loss-drop-threshold', type=float, default=0.05, help='Train-loss drop threshold for overfit warning')
    parser.add_argument('--consecutive-val-points', type=int, default=3, help='Consecutive val points required to confirm critical overfit')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_json = Path(args.log_json)
    out_png = Path(args.out_png)
    out_summary = Path(args.out_summary_json)

    train_steps, train_losses, val_steps, val_mious = parse_mmengine_json(log_json)

    if not val_steps:
        raise RuntimeError('No mIoU found in log. Please ensure this is a training log with validation metrics.')

    summary = detect_overfit(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_mious=val_mious,
        miou_drop_threshold=args.miou_drop_threshold,
        loss_drop_threshold=args.loss_drop_threshold,
    )

    critical = detect_overfit_critical_step(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_mious=val_mious,
        miou_drop_threshold=args.miou_drop_threshold,
        loss_drop_threshold=args.loss_drop_threshold,
        consecutive_val_points=args.consecutive_val_points,
    )

    plot_curve(
        train_steps=train_steps,
        train_losses=train_losses,
        val_steps=val_steps,
        val_mious=val_mious,
        out_png=out_png,
        smooth_window=args.smooth_window,
        critical_step=critical['critical_step'] if critical['critical_found'] else None,
    )

    summary['num_val_points'] = len(val_steps)
    summary['first_step'] = val_steps[0]
    summary.update(critical)

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[OK] curve saved: {out_png}')
    print(f'[OK] summary saved: {out_summary}')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
